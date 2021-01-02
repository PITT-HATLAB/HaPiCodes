import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

# Set product details
product = 'M3102A'
chassis = 1
slot = 5
# open a module
module = SD1.SD_AIN()
moduleID = module.openWithSlot(product, chassis, slot)

if moduleID < 0:
    print("Module open error: ", moduleID)
else:
    print("Module is: ", moduleID)
# Loading FPGA sandbox using .K7z file
error = module.FPGAload(
    r'C:\PXI_FPGA\Projects\Tests\IQIntegratorTest\IQIntegratorTest.data\bin\IQIntegratorTest_2020-12-20T20_06_54\IQIntegratorTest.k7z')


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


def check_SD1Error(func: Callable):
    def inner(*args, **kwargs):
        err = func(*args, **kwargs)
        if (type(err) is int) and (err < 0):
            raise KeysightSD1APIError(err)
        else:
            return err

    return inner


@check_SD1Error
def get_FPGA_register(module_: SD1.SD_Module, reg_name: str):
    reg_ = module_.FPGAgetSandBoxRegister(reg_name)
    return reg_


@check_SD1Error
def write_FPGA_register(module_: SD1.SD_Module, reg_name: str, reg_val: int):
    reg_ = get_FPGA_register(module_, reg_name)
    err = reg_.writeRegisterInt32(reg_val)
    return err


@check_SD1Error
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


def config_mem_reader(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      mem_name: str = "Host_mem_1", delay_reg_name: str = "MemReaderRegisters_rd_trig_delay",
                      length_reg_name: str = "MemReaderRegisters_rd_length"):
    write_FPGA_memory(module_, mem_name, input_data)
    write_FPGA_register(module_, delay_reg_name, trig_delay)
    write_FPGA_register(module_, length_reg_name, rd_length)


def config_IQIntegrator(module_: SD1.SD_Module, integ_start: int, integ_stop: int,
                        truncate_demod: List[int] = None,
                        truncate_integ: List[int] = None):
    write_FPGA_register(module_, "IntegRegisters_integ_trig_delay", integ_start)
    write_FPGA_register(module_, "IntegRegisters_integ_length", integ_stop)

    if truncate_demod is None:
        truncate_demod = [18, 18, 18, 18]
        #TODO: This needs a validator. The possible value should be 0-18 (for current FPGA), and this value needs to be in a markup file)
    if truncate_integ is None:
        truncate_integ = [5, 7, 7, 7] #TODO: Again, validator. The possible value should be 0-15 (for current FPGA)
    for i in range(4):
        write_FPGA_register(module_, f"TruncationRegisters_truncate_demod_{i}", truncate_demod[i])
        write_FPGA_register(module_, f"TruncationRegisters_truncate_integ_{i}", truncate_integ[i])


# PUT DATA IN ON-BOARD MEMORY AS INPUT DATA
# MEM_DIN = list(range(500))
DIN_x = np.arange(0, 1000, 2)
# DIN_shape = np.exp(-(DIN_x - np.mean(DIN_x)) ** 2 / 200 ** 2)
# DIN_shape =  np.linspace(0, 1, len(DIN_x))**1
DIN_shape =  np.zeros(len(DIN_x))**3+0.9
MEM_DIN = 32767 * np.sin(np.pi / 10 * DIN_x + np.pi / 4) * DIN_shape
MEM_DIN = MEM_DIN.astype(int)
for i in range(5):
    config_mem_reader(module, MEM_DIN[i::5], 2, len(DIN_x) // 5, f"Host_mem_{i + 1}")


# WRITE WEIGHT FUNCTION
WEIGHT_data = np.zeros(len(DIN_x)//5, dtype=int) + 2**15-1
config_mem_reader(module, WEIGHT_data, 2, len(DIN_x) // 5, "Host_mem_6",
                  delay_reg_name="WeightReaderRegisters_rd_trig_delay",
                  length_reg_name="WeightReaderRegisters_rd_length")


# CONFIGURE DEMODULATOR and IQINTEGRATOR
config_IQIntegrator(module, 18, 50) # stop-start-1 points. start from start+2, the maximum integration cycle is 2**16-1, maximum trig delay is 2**31-1

# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(800)
CYCLES = 1
TRIGGER_DELAY = 0
for i in [1, 2, 3]:
    module.DAQconfig(i, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0111)

# DAQ Trigger
module.DAQtriggerMultiple(0b0111)

# PXI Trigger
'''
PXI7 = 7
module.PXItriggerWrite(PXI7, 1);
module.PXItriggerWrite(PXI7, 0);
time.sleep(0.001)
module.PXItriggerWrite(PXI7, 1);
'''

# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQread(1, POINTS_PER_CYCLE * CYCLES, TIMEOUT)
dataRead2 = module.DAQread(2, POINTS_PER_CYCLE * CYCLES, TIMEOUT)
dataRead3 = module.DAQread(3, POINTS_PER_CYCLE * CYCLES, TIMEOUT)

# exiting...
module.close()
print()
print("AIN closed")

# processing
# plt.figure()
# plt.title("input data")
# plt.plot(DIN_x, MEM_DIN)

demod_out_I = dataRead2[0::5]
demod_out_Q = dataRead2[1::5]
integ_out_I = dataRead2[2::5]
integ_out_Q = dataRead2[3::5]
daq1_trig_out = dataRead1[3::5]
weight_out = dataRead1[4::5]
mem_out = dataRead3
xdata = np.arange(POINTS_PER_CYCLE) * 2

plt.figure()
plt.plot(xdata, mem_out, label="input data")
plt.plot(xdata[::5], weight_out, label="weight func")
plt.plot(xdata[::5], demod_out_I, label="demod_I")
plt.plot(xdata[::5], demod_out_Q, label="demod_Q")
plt.plot(xdata[::5], integ_out_I, label="integ_I")
plt.plot(xdata[::5], integ_out_Q, label="integ_Q")
plt.plot(xdata[::5], daq1_trig_out*30000, label="DAQ1 trig")
plt.legend()


# plt.figure()
# plt.plot(xdata[::5], dac1_trig_out)

#FIND INTEG RANGE
result_I = integ_out_I[-1]
result_Q = integ_out_Q[-1]
demod_out_Q_weight = np.int64(demod_out_Q)*32767>>15

for i in range(len(xdata)//5):
    integ_I_temp = np.sum(demod_out_I[i:i + 50]) >> 5
    integ_Q_temp = np.sum(demod_out_Q_weight[i:i + 50]) >> 5
    # if (integ_I_temp == result_I) and (integ_Q_temp == result_Q) :
    if (integ_Q_temp == result_Q):
        print (i)


# TEST ADDON FUNCS
from AddOns.DigFPGAConfig import get_recommended_truncation
dt_, it_ = get_recommended_truncation(np.array([demod_out_I]), np.array([demod_out_Q]), 180, 680, 18)
print (dt_, it_)