from typing import List, Callable, Union, Optional, Dict
import inspect
import os
import time
import numpy as np
import h5py
from HaPiCodes.sd1_api.SD1AddOns import findFPGAbyName
from HaPiCodes.sd1_api import keysightSD1
from HaPiCodes.pathwave.HVIConfig import open_modules, define_instruction_compile_hvi
import logging
import sys
import time

try:
    from tqdm import tqdm
    PROGRESSBAR = tqdm
except ImportError:
    PROGRESSBAR = lambda *x, **kwargs: x[0]

def print_percent_done(index, total, bar_len=50, title='Please wait'):
    '''
    index is started from 0.
    '''
    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done, {index}/{total}', end='\r')

    if round(percent_done) == 100:
        print('\t✅')


class PXI_Instruments():
    """ class that contains all the modules on the PXI chassis
    """
    def __init__(self, msmtInfoDict: dict, reloadFPGA: bool = True, boards: Union[str, List]="user_defined"):
        self.msmtInfoDict = msmtInfoDict
        module_configs = msmtInfoDict["moduleConfig"]
        module_dict = open_modules(boards)

        # gather all the modules used in the yaml file
        self.usedModules = []
        for ch_combo in msmtInfoDict["combinedChannelUsage"].values():
            for mod_ch in ch_combo.values():
                self.usedModules.append(mod_ch[0])
        self.usedModules = list(set(self.usedModules))

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
            module_config_dict1 = module_config_dict.copy()
            FPGA = module_config_dict1.get("FPGA")
            if "FPGA" in module_config_dict1.keys():
                module_config_dict1.pop("FPGA")
            if FPGA is None :
                if default_FPGA is not None:
                    if reloadFPGA:
                        instrument.FPGAload(default_FPGA)
                        print (f"default FPGA {default_FPGA} is loaded to {module}")
                    else:
                        instrument.FPGA_file = findFPGAbyName(default_FPGA)
                        instrument.FPGAconfigureFromK7z(instrument.FPGA_file)
                        instrument.getFPGAconfig()
            else:
                if reloadFPGA:
                    instrument.FPGAload(FPGA)
                    print(f"FPGA {FPGA} is loaded to {module}")
                else:
                    instrument.FPGA_file = findFPGAbyName(FPGA)
                    print(f"{module} is using FPGA {FPGA}")
                    instrument.FPGAconfigureFromK7z(instrument.FPGA_file)
                    instrument.getFPGAconfig()
            # check subbuffer usage for digitizer modules
            if instrument.getProductName() == "M3102A":
                if module in self.usedModules:
                    subbuff_used_list.append(instrument.subbuffer_used)
            # configure module, offset, coupling etc
            instrument.moduleConfig(**module_config_dict1)

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
                    ch_config_dict_1 = ch_config_dict.copy()
                    try:
                        wf_data_file = ch_config_dict_1.pop("wf_data_file")
                        wf_data = getWeightFuncByName(wf_data_file)
                        ch_config_dict_1["wf_data"] = wf_data
                        wf_start = ch_config_dict_1.get("wf_start", None)
                        if wf_start is None:
                            ch_config_dict_1["wf_start"] = ch_config_dict_1["integ_start"]
                    except KeyError:
                        pass
                    config_func_args = inspect.getfullargspec(inst.configFPGA).args
                    for k in ch_config_dict:
                        if (k not in config_func_args) and (k in ch_config_dict_1):
                            ch_config_dict_1.pop(k)
                    inst.configFPGA(inst, int(ch[-1]), module_name=module_name, **ch_config_dict_1)

    def autoConfigAllDAQ(self, W, Q, DAQ_trigger_limit="default", triggerMode=keysightSD1.SD_TriggerModes.SWHVITRIG):
        """ automatically configure all the Data AcQuisition blocks, based on the average numer, trigger number per
            experiment, etc.
        :param DAQ_trigger_limit: maximum trigger number after each reconfigure of DAQ, the default value is in
            HaPiCodes.pathwave.sysInfo["sysConstants"]["DAQ_Trigger_Limit"] (=1000). Sometimes when there are so many
            data points per trigger, this value needs to be smaller.
        """
        self._MQ = Q._MQ
        self._MW = Q._MW
        avg_num = self.msmtInfoDict["sequeceAvgNum"]
        demod_length_dict = self.msmtInfoDict.get("demodConfig", {})
        # find the maximum trigger number per experiment among all channel of all the modules
        max_trig_num_per_exp = 0
        self.triggered_digs = []
        for dig_, trig_nums in self._MQ.dig_trig_num_dict.items():
            max_in_module = np.max(np.fromiter(trig_nums.values(), dtype=int))
            max_trig_num_per_exp = np.max((max_in_module, max_trig_num_per_exp))
            if max_in_module > 0:
                self.triggered_digs.append(dig_)

        # configure all DACs
        hviCyc_list = []
        nAvgPerHVI_ = 0
        for dig_name in self.triggered_digs:
            trig_nums = self._MQ.dig_trig_num_dict[dig_name]
            inst = self.module_dict[dig_name].instrument
            demodLengthList = demod_length_dict.get(dig_name,{}).get("demodLength")
            nAvgPerHVI_, nHVICyc_ = inst.DAQAutoConfig(trig_nums, avg_num, max_trig_num_per_exp,
                                             demodLengthList, triggerMode, DAQ_trigger_limit)
            if nAvgPerHVI_ is not None:
                hviCyc_list.append(nHVICyc_)
        if len(np.unique(hviCyc_list)) != 1:
            raise ValueError("Error automatic configuring all DAQ")
        self.avg_num_per_hvi =  nAvgPerHVI_
        self.hvi_cycles =  hviCyc_list[0]

    def uploadPulseAndQueue(self, timer = False):
        if timer:
            t0_ = time.time()
            print("uploading pulse and compile HVI")
        for module_name, module in self.module_dict.items():
            if module.instrument.getProductName() != "M3102A":
                w_index = module.instrument.AWGuploadWaveform(getattr(self._MW, module_name))
                module.instrument.AWGqueueAllChanWaveform(w_index, getattr(self._MQ, module_name))
        rt_ = self.msmtInfoDict["sequenceRelaxingTime"]
        try:
            relaxingTime_ns = eval(rt_)
        except TypeError:
            relaxingTime_ns = rt_
        if relaxingTime_ns < 100:
            logging.warning(f"relaxing time is very short ({relaxingTime_ns} ns). Make sure this is"
                            f"really what you need.")
        pulse_general_dict = dict(relaxingTime=relaxingTime_ns/1e3, avgNum=self.avg_num_per_hvi)
        hvi = define_instruction_compile_hvi(self.module_dict, self._MQ, pulse_general_dict, self.subbuffer_used)
        self.hvi = hvi
        self.relaxingTime_ns = relaxingTime_ns
        if timer:
            print(f"took {time.time()-t0_} s to upload pulse and compile HVI")
        return hvi

    def runExperiment(self, timeout=10, showProgress: bool = True, preAverage=False):
        """
        Start running the hvi generated from W and Q, and receive data.
        :param timeout: in ms
        :param preAverage: Average data after each HVI run. Only for subbuffer not used case.
        :return : for demodulate FPGA (no subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp, demod_length//10)
                for Qubit_MSMT firmware (subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp)
        """
        # generate an empty data receive dict
        data_receive = {}
        self.dig_trig_masks = {}
        for dig_name in self.triggered_digs:
            dig_module = self.module_dict[dig_name].instrument
            data_receive[dig_name] = {}
            ch_mask = ""
            for ch, (ppc, cyc) in dig_module.DAQ_config_dict.items():
                if (ppc != 0) and (cyc != 0):
                    data_receive[dig_name][ch] = []
                    ch_mask = '1' + ch_mask
                else:
                    ch_mask = '0' + ch_mask
            self.dig_trig_masks[dig_name] = ch_mask
        self.data_receive = data_receive

        if not showProgress:
            PROGRESSBAR_ = lambda *x, **kwargs: x[0]
        else:
            PROGRESSBAR_ = PROGRESSBAR
        # run HVI

        if self.subbuffer_used:
            if preAverage :
                raise NotImplementedError("Preaverage is not implemented for subbuffer mode. "
                                          "Actually this shouldn't be needed, since the subbuffer mode shouldn't take "
                                          "a lot of PC RAM")
            for dig_name, msk in self.dig_trig_masks.items():
                self.module_dict[dig_name].instrument.DAQstartMultiple(int(msk, 2))

            for i in PROGRESSBAR_(range(self.hvi_cycles), desc="HVI is running"):
                if PROGRESSBAR_.__name__ == '<lambda>' and showProgress:
                    print('HVI is running')
                    print_percent_done(i, self.hvi_cycles)
                self.hvi.run(self.hvi.no_timeout)

            # only receive data after all HVI run
            self.receiveDataFromAllDAQ(timeout)
            self.hvi.stop()

        else:
            for i in PROGRESSBAR_(range(self.hvi_cycles), desc="HVI is running"):
                if PROGRESSBAR_.__name__ == '<lambda>' and showProgress:
                    print('HVI is running')
                    print_percent_done(i, self.hvi_cycles)
                # start DAQs
                for dig_name, msk in self.dig_trig_masks.items():
                    self.module_dict[dig_name].instrument.DAQstartMultiple(int(msk, 2))
                    # config DAQ channels
                    for ch in self.data_receive[dig_name]:
                        self.module_dict[dig_name].instrument.reConfigDAQ(int(ch[-1]))

                self.hvi.run(self.hvi.no_timeout)
                self.receiveDataFromAllDAQ(timeout, mute=True, averageData=preAverage)
                self.hvi.stop()

        return self.data_receive

    def receiveDataFromAllDAQ(self, timeout:int, mute:bool = True, averageData=False):
        # receive data
        for module_name, channels in self.data_receive.items():
            for ch in channels:
                if not mute:
                    print(f"receive data from {module_name} channel {ch}")
                inst = self.module_dict[module_name].instrument
                try:
                    new_data_ = inst.DAQreadArray(int(ch[-1]), timeout, reshapeMode="nAvg")
                    if self.data_receive[module_name][ch] == []:
                        self.data_receive[module_name][ch] = new_data_/ self.hvi_cycles if averageData else new_data_
                    elif averageData: # accumulate the new data on top of old data for average.
                        self.data_receive[module_name][ch] = self.data_receive[module_name][ch] + new_data_ / self.hvi_cycles
                    else:# append data if there is already data in self.data_receive
                        self.data_receive[module_name][ch] = np.concatenate((self.data_receive[module_name][ch], new_data_))
                except Exception as e:
                    print("data receive error: ", e)
                    if "cannot reshape array of size" in str(e):
                        print("it seems that there are too many data points per DAQ trigger cycle, try lower "
                              "DAQ_trigger_limit in pxi.autoConfigAllDAQ")
        return self.data_receive

    def releaseHVI(self):
        try:
            self.hvi.release_hw()
            print("Releasing HW...")
        except AttributeError:
            print('No hvi initial')
            return "No HVI"

    def closeModule(self):
        for engine_name, module in self.module_dict.items():
            inst = module.instrument
            if inst.getProductName()[:3] == "M32":
                for i in range(1, inst._ch_num + 1):
                    inst.AWGstop(i)
            inst.close()
        print("Modules closed\n")

    def releaseHviAndCloseModule(self):
        hvi_str = self.releaseHVI()
        if hvi_str is not None:
            print('No hvi initial, close module only')
        self.closeModule()


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