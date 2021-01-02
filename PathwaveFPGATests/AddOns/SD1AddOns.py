import time
from typing import List, Callable, Union
import warnings
import numpy as np
import matplotlib.pyplot as plt
import keysightSD1 as SD1

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
        if not (rd_mem == mem_).all:
            raise KeysightSD1APIError("Attention! Data written into memory does not match with the input data. "
                                      "Try using NONDMA mode, or add waiting time after data writing, or separate data "
                                      "into pieces and write piece by piece")
        return rd_mem


class AIN(SD1.SD_AIN):
    def __init__(self):
        super(AIN, self).__init__()
        self.__hvi = None
        self.DAQ_config_dict = {"ch1": (None, None),
                                "ch2": (None, None),
                                "ch3": (None, None),
                                "ch4": (None, None)
                                }

    def DAQconfig(self, channel, pointsPerCycle, nCycles, triggerDelay, triggerMode):
        super().DAQconfig(channel, pointsPerCycle, nCycles, triggerDelay, triggerMode)
        self.DAQ_config_dict[f"ch{int(channel)}"] = (nCycles, pointsPerCycle)

    def DAQreadArray(self, channel, timeOut = 0):
        """ reads all the data as specified in DAQconfig (pointsPerCycle * nCycles) and reaturn as an numpy float array
         with shape (nCycles, pointsPerCycle).
         user have to make sure "nCycles" number of triggers have been sent before read.

        :param channel: DAQ channel to read
        :param timeOut: timeout
        :return:
        """
        ch_ncyc = self.DAQ_config_dict[f"ch{int(channel)}"][0]
        ch_ppc = self.DAQ_config_dict[f"ch{int(channel)}"][1]
        raw_data = self.DAQread(channel, ch_ppc * ch_ncyc, timeOut)
        array_data = np.array(raw_data).astype(float).reshape(ch_ncyc, ch_ppc)
        return array_data

class AOU(SD1.SD_AOU):
    def __init__(self):
        super(AOU, self).__init__()
        self.__hvi = None

    def AWGconfig(self, offset: list = [0, 0, 0, 0], amplitude: list = [1.5, 1.5, 1.5, 1.5], numChan: int = 4):
        """basic configure AOU module to fitinto our purpose for AWG

        Args:
            offset (list, optional): DC offset to compensate any potential leakage
            amplitude (list, optional): full amplitude for output voltage
            numChan (int, optional): number of channel
        """
        syncMode = SD1.SD_SyncModes.SYNC_CLK10
        queueMode = SD1.SD_QueueMode.CYCLIC  # (ONE_SHOT / CYCLIC)
        self.waveformFlush()  # memory flush
        self.channelPhaseResetMultiple(0b1111)
        for i in range(1, numChan + 1):
            self.AWGflush(i)
            self.AWGqueueSyncMode(i, syncMode)
            self.AWGqueueConfig(i, queueMode)
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
            for seqOrder, seqInfo in getattr(queueCollection, f'chan{chan}').items():
                for singlePulse in seqInfo:
                    triggerDelay = 0
                    # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
                    self.AWGqueueWaveform(chan, w_index[singlePulse[0]], triggerMode, triggerDelay, 1,
                                          0)  # singlePulse = ['pulseName', timeDelay]
        self.AWGstartMultiple(0b1111)  # TODO: not necessary
