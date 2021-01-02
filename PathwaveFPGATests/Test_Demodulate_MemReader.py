import sys
import time
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
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
    r'C:\PXI_FPGA\Projects\Tests\DemodelateTest\DemodelateTest.data\bin\DemodelateTest_2020-12-18T16_49_46\DemodelateTest.k7z')


class KeysightSD1APIError(Exception):
    """Exception raised for errors happened when calling the SD1 API functions.

    :param error_code: error code returned from SD1 API functions.
    """

    def __init__(self, error_code: int):
        error_msg = SD1.SD_Error.getErrorMessage(error_code)
        super().__init__(error_msg)


def check_SD1Error(func: Callable):
    def inner(*args, **kwargs):
        err = func(*args, **kwargs)
        if (type(err) is int) and (err < 0):
            raise KeysightSD1APIError(err)
        else:
            return err

    return inner


def config_mem_reader(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      target_mem: str = "Host_mem_1"):
    @check_SD1Error
    def get_reg(reg_name_):
        reg_ = module_.FPGAgetSandBoxRegister(reg_name_)
        return reg_

    reg_trig_delay = get_reg("MemReaderRegisters_rd_trig_delay")
    err = reg_trig_delay.writeRegisterInt32(trig_delay)
    check_SD1Error(err)

    reg_rd_length = get_reg("MemReaderRegisters_rd_length")
    err = reg_rd_length.writeRegisterInt32(rd_length)
    check_SD1Error(err)

    memory_map = get_reg(target_mem)
    err = memory_map.writeRegisterBuffer(0, input_data, SD1.SD_AddressingMode.AUTOINCREMENT, SD1.SD_AccessMode.NONDMA)
    check_SD1Error(err)
    # Read buffer from memory map to check
    c_value = memory_map.readRegisterBuffer(0, 1000, SD1.SD_AddressingMode.AUTOINCREMENT,
                                            SD1.SD_AccessMode.NONDMA)
    print(c_value)


def config_demodulator(module_: SD1.SD_Module, truncation_lower_bits: List[int] = [18, 18, 18, 18]):
    @check_SD1Error
    def get_reg(reg_name_):
        reg_ = module_.FPGAgetSandBoxRegister(reg_name_)
        return reg_

    for i in range(4):
        trunc_reg = get_reg(f"TruncationRegisters_truncate_lower_bit_{i}")
        err = trunc_reg.writeRegisterInt32(truncation_lower_bits[i])
        check_SD1Error(err)


# PUT DATA IN ON-BOARD MEMORY
# MEM_DIN = list(range(1000))
DIN_x = np.arange(0, 1000, 2)
DIN_shape = np.exp(-(DIN_x - np.mean(DIN_x)) ** 2 / 200 ** 2)
MEM_DIN = 32767 * np.sin(np.pi / 10 * DIN_x+np.pi/4) * DIN_shape
MEM_DIN = MEM_DIN.astype(int)
plt.figure()
plt.plot(DIN_x, MEM_DIN)
for i in range(5):
    config_mem_reader(module, MEM_DIN[i::5], 2, len(DIN_x)//5, f"Host_mem_{i + 1}")

# CONFIGURE DEMODULATOR
config_demodulator(module, [17, 4, 4, 4])

# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(500)
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
demod_out_I = dataRead1[0::5]
demod_out_Q = dataRead1[1::5]
dac1_trig_out = dataRead2[4::5]
dmod_out_mag = dataRead2[0::5]
mem_dout = dataRead3
xdata = np.arange(POINTS_PER_CYCLE)*2

plt.figure()
plt.plot(xdata[::5], demod_out_I)
plt.plot(xdata[::5], demod_out_Q)
plt.plot(xdata[::5], dmod_out_mag)



plt.figure()
plt.plot(xdata[::5], dac1_trig_out)
plt.plot(xdata, mem_dout)
