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
channel = 1
# open a module
module = SD1.SD_AIN()
moduleID = module.openWithSlot(product, chassis, slot)

if moduleID < 0:
    print("Module open error: ", moduleID)
else:
    print("Module is: ", moduleID)
# Loading FPGA sandbox using .K7z file
error = module.FPGAload(
    r'C:\PXI_FPGA\Projects\Tests\MemReader\MemReader.data\bin\MemReader_2020-12-05T17_50_05\MemReader.k7z')


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


def config_mem_reader(module_: SD1.SD_Module, mem_data: List[int], trig_delay: int, rd_length: int):
    @check_SD1Error
    def get_reg(reg_name_):
        reg_ = module_.FPGAgetSandBoxRegister(reg_name_)
        return reg_

    reg_trig_delay = get_reg("Register_Bank_rd_trig_delay")
    err = reg_trig_delay.writeRegisterInt32(trig_delay)
    check_SD1Error(err)

    reg_rd_length = get_reg("Register_Bank_rd_length")
    err = reg_rd_length.writeRegisterInt32(rd_length)
    check_SD1Error(err)

    memory_map = get_reg("Host_mem_1")
    err = memory_map.writeRegisterBuffer(0, mem_data, SD1.SD_AddressingMode.AUTOINCREMENT, SD1.SD_AccessMode.DMA)
    check_SD1Error(err)
    # Read buffer from memory map to check
    c_value = memory_map.readRegisterBuffer(0, 1000, SD1.SD_AddressingMode.AUTOINCREMENT,
                                            SD1.SD_AccessMode.NONDMA)
    print(c_value)


a = list(range(1000))
config_mem_reader(module, a, 2, 10)

# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(5e7)
CYCLES = 1
TRIGGER_DELAY = 0
module.DAQconfig(channel, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstart(channel)

# input("Press any key to provide trigger")
module.DAQtrigger(channel)

PXI7 = 7
module.PXItriggerWrite(PXI7, 1);
module.PXItriggerWrite(PXI7, 0);
time.sleep(0.001)
module.PXItriggerWrite(PXI7, 1);


# READ DATA
TIMEOUT = 1
dataRead = module.DAQread(channel, POINTS_PER_CYCLE * CYCLES, TIMEOUT)

# exiting...
module.close()
print()
print("AIN closed")


# processing
trig_data = dataRead[::5]
mem_data = dataRead[1::5]
xdata = np.arange(POINTS_PER_CYCLE/5)

plt.figure()
plt.plot(xdata, trig_data)
plt.plot(xdata, mem_data)
