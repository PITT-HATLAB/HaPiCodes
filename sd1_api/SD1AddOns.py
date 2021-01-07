import time
import pathlib
import importlib
from typing import List, Callable, Union, Optional
import warnings
import logging

import numpy as np
import yaml

from sd1_api import keysightSD1 as SD1
import FPGA

class KeysightSD1APIError(Exception):
    """Exception raised for errors happened when calling the SD1 API functions.

    :param error_code: error code returned from SD1 API functions.
    """

    def __init__(self, error_code: Union[int, str]):
        if (type(error_code) is int):
            error_msg = SD1.SD_Error.getErrorMessage(error_code)
        else:
            error_msg = "Others: " + error_code
        super().__init__(error_msg)


def check_SD1_error(func: Callable):
    def inner(*args, **kwargs):
        err = func(*args, **kwargs)
        if (type(err) is int) and (err < 0):
            raise KeysightSD1APIError(err)
        else:
            return err

    return inner


def validate_FPGA_register(val):
    """ validate the value of a register that is going to be written to the FPGA on-board memory.
    float values will be converted to int with warning. values beyond int32 will raise error
    :param val: register value
    :return:
    """
    if val < -2**31 or val > 2**31-1: #
        raise KeysightSD1APIError("FPGA register value out of range. [-2**31, 2**31-1]")
    elif not isinstance(val, (int, np.integer)):
        print (val)
        warnings.warn("non-integer register values will be converted to integer")
    return int(val)


@check_SD1_error
def get_FPGA_register(module_: SD1.SD_Module, reg_name: str):
    reg_ = module_.FPGAgetSandBoxRegister(reg_name)
    return reg_


@check_SD1_error
def write_FPGA_register(module_: SD1.SD_Module, reg_name: str, reg_val: int):
    reg_val = validate_FPGA_register(reg_val)
    reg_ = get_FPGA_register(module_, reg_name)
    err = reg_.writeRegisterInt32(reg_val)
    return err


@check_SD1_error
def write_FPGA_memory(module_: SD1.SD_Module, mem_name: str, mem_data: List[int],
                      idx_offset: int = 0, addr_mode: SD1.SD_AddressingMode = SD1.SD_AddressingMode.AUTOINCREMENT,
                      access_mode: SD1.SD_AccessMode = SD1.SD_AccessMode.DMA,
                      double_check: bool = True, waiting_time: float = 0):
    """
    write data to an on-board FPGA memory
    :param module_: target module for memory writing
    :param mem_name: name of the on-board mem, can be find in the sandbox file
    :param mem_data: data to be written into the mem
    :param idx_offset: starting index of memory
    :param addr_mode: addressing mode, can be AUTOINCREMENT or FIXED
    :param access_mode: can be DMA or NONDMA, if double check, I think NONDMA is better.
    :param double_check: if True, will read the mem after writing and compare with the input mem_data
    :return:
    """
    mem_ = get_FPGA_register(module_, mem_name)
    for i in range(len(mem_data)):
        mem_data[i] = validate_FPGA_register(mem_data[i])
    err = mem_.writeRegisterBuffer(idx_offset, mem_data, addr_mode, access_mode)
    if waiting_time != 0:
        time.sleep(waiting_time)
    if not double_check:
        return err
    else:
        rd_mem = mem_.readRegisterBuffer(idx_offset, len(mem_data), addr_mode, access_mode)
        if not np.array(rd_mem == mem_).all:
            raise KeysightSD1APIError("Attention! Data written into memory does not match with the input data. "
                                      "Try using NONDMA mode, or add waiting time after data writing, or separate data "
                                      "into pieces and write piece by piece")
        return rd_mem

def findFPGAbyName(FPGA_name: str):
    """ Find the FPGA file with the given name in the HaPiCodes.FPGA folder.
    """
    fpga_path = pathlib.Path(FPGA.__path__[0])
    fpga_files = list(fpga_path.glob('**/*.k7z'))
    for f in fpga_files:
        if FPGA_name == f.name[:-4]:
            return str(f)
    raise FileNotFoundError(f"Cannot find FPGA file with name '{FPGA_name}' in HaPiCodes.FPGA")





class AIN(SD1.SD_AIN):
    def __init__(self):
        super(AIN, self).__init__()
        self.__hvi = None
        self.DAQ_config_dict = {}
        self.subbuffer_used = None
        self.FPGA_markup = {}
        self.configFPGA = None

    @check_SD1_error
    def openWithOptions(self, partNumber, nChassis, nSlot, options):
        """ get module channel number when option module self.__ch_num
        """
        id = super().openWithOptions(partNumber, nChassis, nSlot, options)
        self.__ch_num = int(self.getOptions("channels")[-1])
        for i in range(self.__ch_num):
            self.DAQ_config_dict = {f"ch{i + 1}": (0, 0) for i in range(self.__ch_num)}
        return id

    @check_SD1_error
    def FPGAload(self, FPGA_file):
        try:            
            FPGA_file = findFPGAbyName(FPGA_file)
        except FileNotFoundError as e:
            logging.warning(str(e) + " will take the FPGA_file as path to a user generated FPGA")
        err = super().FPGAload(FPGA_file)
        self.FPGA_file = FPGA_file
        # try to find the markup file
        try:
            self.FPGA_markup = yaml.safe_load(open(FPGA_file[:-4] + "_Markup.yaml", 'r'))
            self.subbuffer_used = self.FPGA_markup["subbuffer_used"]
        except FileNotFoundError:
            self.subbuffer_used = None
        # try to find FPGA config function
        config_module_path = FPGA_file[:-4]+ "_Config.py"
        config_module_name = config_module_path.split('\\')[-1][:-3]
        print(config_module_path, config_module_name)
        try:
            print (config_module_path, config_module_name)
            spec = importlib.util.spec_from_file_location(config_module_name, config_module_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            self.configFPGA = config.configFPGA
        except FileNotFoundError:
            self.configFPGA = None

        return err

    def DAQconfig(self, channel, pointsPerCycle, nCycles, triggerDelay, triggerMode):
        super().DAQconfig(channel, pointsPerCycle, nCycles, triggerDelay, triggerMode)
        self.DAQ_config_dict[f"ch{int(channel)}"] = (nCycles, pointsPerCycle)

    def moduleConfig(self, fullScale: List[float] = None, impedance: List[SD1.AIN_Impedance] = None,
                     coupling: List[SD1.AIN_Coupling] = None, prescaler: List[int] = None):
        """basic configure AIN module

        """
        if fullScale is None:
            fullScale = [2 for i in range(self.__ch_num)]
        if prescaler is None:
            prescaler = [0 for i in range(self.__ch_num)]
        if impedance is None:
            impedance = [SD1.AIN_Impedance.AIN_IMPEDANCE_50 for i in range(self.__ch_num)]
        if coupling is None:
            coupling = [SD1.AIN_Coupling.AIN_COUPLING_AC for i in range(self.__ch_num)]

        for i in range(1, self.__ch_num+1):
            self.channelInputConfig(i, fullScale[i - 1], impedance[i - 1], coupling[i - 1])
            self.channelPrescalerConfig(i, prescaler[i - 1])
        self.FPGAreset(SD1.SD_ResetMode.PULSE)



    def DAQAutoConfig(self, trig_nums:dict, avg_num:int, max_trig_num_per_exp: int  = None,
                      demod_length_list: Optional[List[int]] = None, triggerMode = SD1.SD_TriggerModes.SWHVITRIG):
        """
        :param trig_nums: dictionary that contains number of triggers for each channel in one experiment sequence
        :param avg_num: number of experiment runs to average
        :param max_trig_num_per_exp: this should be the be maximum trig number in one experiment sequence among all
            channels of all Dig modules, if None, the maximum trigger num of current module will be used
        :param demod_length_list: List of demodulation length for each channel, in ns. Should only be provided when
            subbuffer is not used
        :param triggerMode: see SD1 manual
        """
        if self.subbuffer_used is None:
            raise NotImplementedError("DAQAutoConfig not implemented for current FPGA")
        triggerDelay = self.FPGA_markup.get("DAQ_trig_delay", 0)
        dig_trig_num_list = np.fromiter(trig_nums.values(), dtype=int)
        if max_trig_num_per_exp is None:
            # find the maximum trigger number per experiment among all channels in the current module
            max_trig_num = np.max(dig_trig_num_list) * avg_num
        else:
            # use the provided value, should be provided from PXIModules.autoConfigAllDAQs
            max_trig_num = max_trig_num_per_exp * avg_num
        if max_trig_num_per_exp == 0:
            return

        if not self.subbuffer_used:
            if  demod_length_list is None:
                raise ValueError("Looks like you are using the demodulate FPGA, "
                                 "in this case, demod_length_list must be provided")
            else:
                ppc_list = np.array(demod_length_list)//10*5
                cyc_list = dig_trig_num_list * avg_num
                avg_num_per_hvi = avg_num
                self.avg_num = avg_num

        else:
            if demod_length_list is not None:
                logging.warning("demod_length_list is omitted for the auto configuration of DAQs with subbuffer used")
            min_cyc = int(np.ceil(max_trig_num / self.FPGA_markup["subbuffer_size"]))
            # try to find a cycle number that is not too large compared to the min_cyc needed, and is a factor of
            # avg_number. If it can't find good cycle number within [min_cyc, min_cyc*2], avg_num will be adjusted.
            cycles = min_cyc
            for cyc in range(min_cyc, min_cyc*2):
                if avg_num % cyc == 0:
                    cycles = cyc
                    break
            avg_num_per_cycle = np.ceil(avg_num/cycles)
            self.avg_num = int(cycles * avg_num_per_cycle)
            if self.avg_num != avg_num:
                logging.warning(f"For easier configuration of DAQ, the average number is adjusted to {self.avg_num}")
            ppc_list = dig_trig_num_list * avg_num_per_cycle * 5
            cyc_list = [cycles] * len(dig_trig_num_list)
            avg_num_per_hvi = int(avg_num_per_cycle)

        ppc_list = np.array(ppc_list, dtype=int).tolist()
        cyc_list = np.array(cyc_list, dtype=int).tolist()
        for ch_name in trig_nums:
            ch = int(ch_name[-1])
            if ppc_list[ch-1] != 0: # only configure channels that are going to be triggered
                self.DAQconfig(ch, ppc_list[ch-1], cyc_list[ch-1], triggerDelay, triggerMode)
        return avg_num_per_hvi

    def DAQreadArray(self, channel, timeOut = 10, reshapeMode = "nAvg"):
        """ reads all the data as specified in DAQconfig (pointsPerCycle * nCycles) and reaturn as an numpy float array
         user have to make sure enough number of triggers have been sent before read.

        :param channel: DAQ channel to read
        :param timeOut: timeout in ms
        :param reshapeMode: "nCycles" or "nAvg"

        :return: for demodulate FPGA (no subbuffer used) the return is np float array
                    with shape (nCycles, pointsPerCycle) or (avg_num, msmt_num_per_exp, demod_length//10)
                for Qubit_MSMT firmware (subbuffer used) the return is np float array
                    with shape (nCycles, pointsPerCycle) or (avg_num, msmt_num_per_exp)
        """
        if self.subbuffer_used is None:
            raise NotImplementedError("DAQreadArray not implemented for current FPGA")
        ch_ncyc = self.DAQ_config_dict[f"ch{int(channel)}"][0]
        ch_ppc = self.DAQ_config_dict[f"ch{int(channel)}"][1]
        raw_data = self.DAQread(channel, ch_ppc * ch_ncyc, timeOut)
        if reshapeMode == "nCycles":
            array_data = np.array(raw_data).astype(float).reshape(ch_ncyc, ch_ppc)
        elif reshapeMode == "nAvg":
            if self.subbuffer_used:
                array_data = np.array(raw_data).astype(float).reshape(self.avg_num, -1)
            else:
                array_data = np.array(raw_data).astype(float).reshape(self.avg_num, -1, ch_ppc)
        else:
            raise NameError("invalid reshapeMode, must be eigher 'nCycyles' or 'nAvg' ")
        return array_data

class AOU(SD1.SD_AOU):
    def __init__(self):
        super(AOU, self).__init__()
        self.__hvi = None
        self.configFPGA = None

    @check_SD1_error
    def FPGAload(self, FPGA_file):
        try:
            FPGA_file = findFPGAbyName(FPGA_file)
        except FileNotFoundError as e:
            logging.warning(str(e) + " will take the FPGA_file as path to a user generated FPGA")
        err = super().FPGAload(FPGA_file)
        return err

    @check_SD1_error
    def openWithOptions(self, partNumber, nChassis, nSlot, options):
        """ get module channel number when option module self.__ch_num
        """
        id = super().openWithOptions(partNumber, nChassis, nSlot, options)
        self.__ch_num = int(self.getOptions("channels")[-1])
        for i in range(self.__ch_num):
            self.DAQ_config_dict = {f"ch{i + 1}": (0, 0) for i in range(self.__ch_num)}
        return id

    def moduleConfig(self, offset: List[float] = None, amplitude: List[float] = None,
                     syncMode: List[SD1.SD_SyncModes] = None, queueMode: List[SD1.SD_QueueMode] = None):
        """basic configure AOU module to fitinto our purpose for AWG

        Args:
            offset (list, optional): DC offset to compensate any potential leakage
            amplitude (list, optional): full amplitude for output voltage
        """
        if offset is None:
            offset = [0 for i in range(self.__ch_num)]
        if amplitude is None:
            amplitude = [1.5 for i in range(self.__ch_num)]
        if syncMode is None:
            syncMode = [SD1.SD_SyncModes.SYNC_CLK10 for i in range(self.__ch_num)]
        if queueMode is None:
            queueMode = [SD1.SD_QueueMode.CYCLIC for i in range(self.__ch_num)]

        self.waveformFlush()  # memory flush
        self.channelPhaseResetMultiple(0b1111)
        for i in range(1, self.__ch_num + 1):
            self.AWGflush(i)
            self.AWGqueueSyncMode(i, syncMode[i-1])
            self.AWGqueueConfig(i, queueMode[i-1])
            self.channelWaveShape(i, SD1.SD_Waveshapes.AOU_AWG)
            self.channelAmplitude(i, amplitude[i - 1])
            self.channelOffset(i, offset[i - 1])

    def AWGuploadWaveform(self, w_dict: dict):
        """upload all waveform into AWG module and return the index for correspondin pulse

        Args:
            w_dict (dict): {pulseName (str): pulseArray (np.ndarray)}

        Returns:
            dict: {pulseName (str): index (int)}
        """
        w_index = {}
        paddingMode = SD1.SD_Wave.PADDING_ZERO
        index = 0
        for waveformName, waveformArray in w_dict.items():
            if waveformArray == []:
                pass
            else:
                tWave = SD1.SD_Wave()
                tWave.newFromArrayDouble(0, waveformArray)
                self.waveformLoad(tWave, index, paddingMode)
                w_index[waveformName] = index
                index += 1
        return w_index

    def AWGqueueAllChanWaveform(self, w_index: dict, queueCollection):
        """upload all queue into module from queueCollection

        Args:
            w_index (dict): the index corresponding to each waveform. Generated from AWGuploadWaveform
            queueCollection (queueCollection): queueCollection
        """
        triggerMode = SD1.SD_TriggerModes.SWHVITRIG
        for chan in range(1, 5):
            for seqOrder, seqInfo in getattr(queueCollection, f'ch{chan}').items():
                for singlePulse in seqInfo:
                    triggerDelay = 0
                    # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
                    self.AWGqueueWaveform(chan, w_index[singlePulse[0]], triggerMode, triggerDelay, 1,
                                          0)  # singlePulse = ['pulseName', timeDelay]
        self.AWGstartMultiple(0b1111)  # TODO: not necessary
