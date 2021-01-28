from typing import List, Callable, Union, Optional, Dict
import inspect
import os
import time
import numpy as np
import h5py
from HaPiCodes.sd1_api.SD1AddOns import findFPGAbyName
from HaPiCodes.sd1_api import keysightSD1
from HaPiCodes.pathwave.HVIConfig import open_modules, define_instruction_compile_hvi

import sys
import time

try:
    from tqdm import tqdm
    PROGRESSBAR = tqdm
except ImportError:
    PROGRESSBAR = lambda x:x

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
            if "FPGA" in module_config_dict.keys():
                module_config_dict.pop("FPGA")
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
        hviCyc_list = []
        nAvgPerHVI_ = 0
        for dig_name, trig_nums in Q.dig_trig_num_dict.items():
            inst = self.module_dict[dig_name].instrument
            demodLengthList = demod_length_dict.get(dig_name,{}).get("demodLength")
            nAvgPerHVI_, nHVICyc_ = inst.DAQAutoConfig(trig_nums, avg_num, max_trig_num_per_exp,
                                             demodLengthList, triggerMode)
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
                w_index = module.instrument.AWGuploadWaveform(getattr(self.W, module_name))
                module.instrument.AWGqueueAllChanWaveform(w_index, getattr(self.Q, module_name))
        pulse_general_dict = dict(relaxingTime=self.msmtInfoDict["sequenceRelaxingTime"], avgNum=self.avg_num_per_hvi)
        hvi = define_instruction_compile_hvi(self.module_dict, self.Q, pulse_general_dict, self.subbuffer_used)
        self.hvi = hvi
        if timer:
            print(f"took {time.time()-t0_} s to upload pulse and compile HVI")
        return hvi

    def runExperiment(self, timeout=10, showProgress: bool = True):
        """
        Start running the hvi generated from W and Q, and receive data.
        :param timeout: in ms
        :return : for demodulate FPGA (no subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp, demod_length//10)
                for Qubit_MSMT firmware (subbuffer used) the return is np float array
                    with shape (avg_num, msmt_num_per_exp)
        """
        # generate an empty data receive dict
        data_receive = {}
        self.dig_trig_masks = {}
        for dig_name in self.Q.dig_trig_num_dict:
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
            PROGRESSBAR_ = lambda x: x
        else:
            PROGRESSBAR_ = PROGRESSBAR
        # run HVI
        print('HVI is running')
        if self.subbuffer_used:
            for dig_name, msk in self.dig_trig_masks.items():
                self.module_dict[dig_name].instrument.DAQstartMultiple(int(msk, 2))

            for i in PROGRESSBAR_(range(self.hvi_cycles)):
                if PROGRESSBAR_.__name__ == '<lambda>' and showProgress:
                    print_percent_done(i, self.hvi_cycles)
                self.hvi.run(self.hvi.no_timeout)

            # only receive data after all HVI run
            self.receiveDataFromAllDAQ(timeout)
            self.hvi.stop()

        else:
            for i in PROGRESSBAR_(range(self.hvi_cycles)):
                if PROGRESSBAR_.__name__ == '<lambda>' and showProgress:
                    print_percent_done(i, self.hvi_cycles)
                # start DAQs
                for dig_name, msk in self.dig_trig_masks.items():
                    self.module_dict[dig_name].instrument.DAQstartMultiple(int(msk, 2))
                    # config DAQ channels
                    for ch in self.data_receive[dig_name]:
                        self.module_dict[dig_name].instrument.reConfigDAQ(int(ch[-1]))

                self.hvi.run(self.hvi.no_timeout)
                if not self.subbuffer_used: # receive data and after each HVI run
                    self.receiveDataFromAllDAQ(timeout, mute=True)
                self.hvi.stop()

        return self.data_receive

    def receiveDataFromAllDAQ(self, timeout:int, mute:bool = True):
        # receive data
        for module_name, channels in self.data_receive.items():
            for ch in channels:
                if not mute:
                    print(f"receive data from {module_name} channel {ch}")
                inst = self.module_dict[module_name].instrument
                try:
                    new_data_ = inst.DAQreadArray(int(ch[-1]), timeout, reshapeMode="nAvg")
                    if self.data_receive[module_name][ch] == []:
                        self.data_receive[module_name][ch] = new_data_
                    else:# append data if there is already data in self.data_receive
                        self.data_receive[module_name][ch] = np.concatenate((self.data_receive[module_name][ch], new_data_))
                except Exception as e:
                    print("data receive error: ", e)
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