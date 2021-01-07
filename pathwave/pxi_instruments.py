from typing import List, Callable, Union, Optional, Dict
import inspect
import os
import numpy as np
import h5py
from sd1_api import keysightSD1
from pathwave.HVIConfig import open_modules, define_instruction_compile_hvi


class PXI_Instruments():
    """ class that contains all the modules on the PXI chassis
    """
    def __init__(self, msmtInfoDict: dict, reloadFPGA: bool = True):
        self.msmtInfoDict = msmtInfoDict
        module_configs = msmtInfoDict["moduleConfig"]
        module_dict = open_modules()
        subbuff_used_list = []
        for module in module_dict:
            instrument = module_dict[module].instrument
            module_config_dict = module_configs.get(module,{})
            # check module type
            if instrument.getProductName() in ["M3202A", "M3201A"]: #AWG modules
                default_FPGA = None # to be updated
            elif instrument.getProductName() == "M3102A":  # digitizer module
                default_FPGA = "Dig_Original"
            else:
                raise NotImplementedError(f"module {module_dict[module].instrument.getProductName()} is not supported")
            # load FPGA
            FPGA = module_config_dict.get("FPGA")
            if reloadFPGA:
                if FPGA is None :
                    if default_FPGA is not None:
                        instrument.FPGAload(default_FPGA)
                        print (f"default FPGA {default_FPGA} is loaded to {module}")
                else:
                    module_config_dict.pop("FPGA")
                    instrument.FPGAload(FPGA)
                    print(f"FPGA {FPGA} is loaded to {module}")
            #check subbuffer usage for digitizer modules
            if instrument.getProductName() == "M3102A":
                subbuff_used_list.append(instrument.subbuffer_used)
            #configure module, offset, coupling etc
            instrument.moduleConfig(**module_config_dict)
        if len(np.unique(subbuff_used_list)) != 1:
            raise NotImplementedError("Digitizer modules with different FPGAs is not supported at this point")
        self.subbuffer_used = subbuff_used_list[0]
        self.module_dict = module_dict
        # config FPGA registers
        self.configFPGAs()

    def configFPGAs(self):
        for module_name, module in self.module_dict.items():
            inst = module.instrument
            if inst.configFPGA is not None:
                module_config_dict = self.msmtInfoDict.get("FPGAConfig", {}).get(module_name,{})
                for ch, ch_config_dict in module_config_dict.items():
                    try:
                        wf_data_file = ch_config_dict.pop("wf_data_file")
                        wf_data = getWeightFuncByName(wf_data_file)
                        ch_config_dict["wf_data"] = wf_data
                    except KeyError:
                        pass
                    config_func_args = inspect.getfullargspec(inst.configFPGA).args
                    ch_config_dict_1 = ch_config_dict.copy()
                    for k in ch_config_dict:
                        if k not in config_func_args:
                            ch_config_dict_1.pop(k)
                    inst.configFPGA(inst, int(ch[-1]), module_name=module_name, **ch_config_dict_1)

    def autoConfigAllDAQ(self, W, Q, triggerMode=keysightSD1.SD_TriggerModes.SWHVITRIG):
        self.Q = Q
        self.W = W
        avg_num = self.msmtInfoDict["sequeceAvgNum"]
        demod_length_dict = self.msmtInfoDict.get("demodConfig", {})
        # find the maximum trigger number per experiment among all channel of all the modules
        max_trig_num_per_exp = 0
        for trig_nums in Q.dig_trig_num_dict.values():
            max_in_module = np.max(np.fromiter(trig_nums.values(), dtype=int))
            max_trig_num_per_exp = np.max((max_in_module, max_trig_num_per_exp))
        # configure all dig modules
        nAvgPerHVI_list = []
        for dig_name, trig_nums in Q.dig_trig_num_dict.items():
            inst = self.module_dict[dig_name].instrument
            demodLengthList = demod_length_dict.get(dig_name,{}).get("demodLength")
            nAvgPerHVI_ = inst.DAQAutoConfig(trig_nums, avg_num, max_trig_num_per_exp,
                                             demodLengthList, triggerMode)
            if nAvgPerHVI_ is not None:
                nAvgPerHVI_list.append(nAvgPerHVI_)
        if len(np.unique(nAvgPerHVI_list)) != 1:
            raise ValueError("Error automatic configuring all DAQ")
        self.avg_num_per_hvi =  nAvgPerHVI_list[0]


    def uploadPulseAndQueue(self):
        for module_name, module in self.module_dict.items():
            if module.instrument.getProductName() != "M3102A":
                w_index = module.instrument.AWGuploadWaveform(getattr(self.W, module_name))
                module.instrument.AWGqueueAllChanWaveform(w_index, getattr(self.Q, module_name))
        pulse_general_dict = dict(relaxingTime=self.msmtInfoDict["sequenceRelaxingTime"], avgNum=self.avg_num_per_hvi)
        hvi = define_instruction_compile_hvi(self.module_dict, self.Q, pulse_general_dict, self.subbuffer_used)
        self.hvi = hvi
        return hvi

    def runExperiment(self, timeout=10):
        """
        Start running the hvi generated from W and Q, and receive data.

        :param timeout: in ms
        :return : for demodulate FPGA (no subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp, demod_length//10)
                for Qubit_MSMT firmware (subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp)
        """
        data_receive = {}
        cyc_list = []

        for dig_name in self.Q.dig_trig_num_dict:
            dig_module = self.module_dict[dig_name].instrument
            data_receive[dig_name] = {}
            ch_mask = ""
            for ch, (cyc, ppc) in dig_module.DAQ_config_dict.items():
                if (ppc != 0) and (cyc != 0):
                    cyc_list.append(cyc)
                    data_receive[dig_name][ch] = np.zeros(ppc * cyc)
                    ch_mask = '1' + ch_mask
                else:
                    ch_mask = '0' + ch_mask
            self.module_dict[dig_name].instrument.DAQstartMultiple(int(ch_mask, 2))

        if (len(np.unique(cyc_list)) != 1) and self.subbuffer_used:
            raise ValueError("All modules must have same number of DAQ cycles")

        cycles = cyc_list[0]

        print('hvi is running')
        if self.subbuffer_used:
            for i in range(cycles):
                print(f"hvi running {i + 1}/{cycles}")
                self.hvi.run(self.hvi.no_timeout)
        else:
            self.hvi.run(self.hvi.no_timeout)

        for module_name, channels in data_receive.items():
            for ch in channels:
                print(f"receive data from {module_name} channel {ch}")
                inst = self.module_dict[module_name].instrument
                data_receive[module_name][ch] = inst.DAQreadArray(int(ch[-1]), timeout, reshapeMode="nAvg")
        self.hvi.stop()

        return data_receive

    def releaseHviAndCloseModule(self):
        try:
            self.hvi.release_hw()
            print("Releasing HW...")
        except AttributeError:
            print('No hvi initial, close module only')
            pass
        for engine_name in self.module_dict:
            self.module_dict[engine_name].instrument.close()
        print("Modules closed\n")


def getWeightFuncByName(wf_name: str):
    """ find weight function file with the give name in the current working directory
    """
    if wf_name is None:
        return
    cwd = os.getcwd()
    filepath = cwd + f"\{wf_name}"
    try:
        f = h5py.File(filepath, 'r')
    except  OSError:
        try:
            f = h5py.File(wf_name, 'r')
        except OSError:
            raise FileNotFoundError(f"cannot find either {wf_name}, or {wf_name} in {cwd}")
    if len(f.keys()) !=1:
        raise NameError("weight function file should only contains one dataset that store the weight function array")
    wf_data = f[list(f.keys())[0]][()]
    print(f"{list(f.keys())[0]} in {filepath} will be used as weight function")
    return wf_data



# --------------------------------------- functions for flexible usage------------------------------------
def digReceiveData(digModule, hvi, pointPerCycle, cycles, chan="1111", timeout=10, subbuffer_used = False):
    data_receive = {}
    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            data_receive[str(chanNum)] = np.zeros(pointPerCycle * cycles)
        else:
            data_receive[str(chanNum)] = []
        chanNum -= 1
    digModule.instrument.DAQstartMultiple(int(chan, 2))

    print('hvi is running')
    if subbuffer_used:
        for i in range(cycles):
            print(f"hvi running {i+1}/{cycles}")
            hvi.run(hvi.no_timeout)

    else:
        hvi.run(hvi.no_timeout)

    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            print('receive data from channel' + str(chanNum))
            data_receive[str(chanNum)] = digModule.instrument.DAQreadArray(chanNum, timeout)
        chanNum -= 1
    hvi.stop()

    return data_receive


def autoConfigAllDAQ(module_dict, Q, avg_num, points_per_cycle_dict: Optional[Dict[str,List[int]]] = None,
                  triggerMode = keysightSD1.SD_TriggerModes.SWHVITRIG):
    if points_per_cycle_dict is None:
        points_per_cycle_dict = {}
    max_trig_num_per_exp = 0
    for trig_nums in Q.dig_trig_num_dict.values():
        max_in_module = np.max(np.fromiter(trig_nums.values(), dtype=int))
        max_trig_num_per_exp = np.max((max_in_module, max_trig_num_per_exp))
    nAvgPerHVI_list = []
    for dig_name, trig_nums in Q.dig_trig_num_dict.items():
        nAvgPerHVI_ = module_dict[dig_name].instrument.DAQAutoConfig(trig_nums, avg_num, max_trig_num_per_exp,
                                                                    points_per_cycle_dict.get(dig_name), triggerMode)
        if nAvgPerHVI_ is not None:
            nAvgPerHVI_list.append(nAvgPerHVI_)
    if len(np.unique(nAvgPerHVI_list)) == 1:
        return nAvgPerHVI_list[0]
    else:
        raise ValueError("Error automatic configuring all DAQ")



def runExperiment(module_dict, hvi, Q, timeout=10):
    # TODO: module_dict and subbuffer_used should be removed if this is a member function fo PXIModules
    data_receive = {}
    cyc_list = []
    subbuff_used_list = []
    for dig_name in Q.dig_trig_num_dict:
        dig_module = module_dict[dig_name].instrument
        data_receive[dig_name] = {}
        ch_mask = ""
        for ch, (cyc, ppc) in dig_module.DAQ_config_dict.items():
            if (ppc!=0) and (cyc != 0):
                cyc_list.append(cyc)
                data_receive[dig_name][ch] = np.zeros(ppc * cyc)
                ch_mask = '1' + ch_mask
            else:
                ch_mask = '0' + ch_mask
        module_dict[dig_name].instrument.DAQstartMultiple(int(ch_mask, 2))
        subbuff_used_list.append(module_dict[dig_name].instrument.subbuffer_used)

    if len(np.unique(cyc_list)) != 1:
        raise ValueError("All modules must have same number of DAQ cycles")
    if len(subbuff_used_list) != 1:
        raise NotImplementedError("Digitizer modules with different FPGAs is not supported at this point")

    cycles = cyc_list[0]
    subbuffer_used = subbuff_used_list[0]

    print('hvi is running')
    if subbuffer_used:
        for i in range(cycles):
            print(f"hvi running {i+1}/{cycles}")
            hvi.run(hvi.no_timeout)
    else:
        hvi.run(hvi.no_timeout)

    for module_name, channels in data_receive.items():
        for ch in channels:
            print(f"receive data from {module_name} channel {ch}")
            inst =  module_dict[module_name].instrument
            data_receive[module_name][ch] = inst.DAQreadArray(int(ch[-1]), timeout, reshapeMode = "nAvg")
    hvi.stop()

    return data_receive