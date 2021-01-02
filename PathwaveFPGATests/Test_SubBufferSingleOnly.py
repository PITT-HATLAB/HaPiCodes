import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

from AddOns.SD1AddOns import AIN, write_FPGA_register
from AddOns.DigFPGAConfig import *

# Set product details
product = 'M3102A'
chassis = 1
slot = 5
# open a module
module = AIN()
moduleID = module.openWithSlot(product, chassis, slot)
if moduleID < 0:
    print("Module open error: ", moduleID)
else:
    print("Module is: ", moduleID)
# Loading FPGA sandbox using .K7z file
error = module.FPGAload(
    r'C:\PXI_FPGA\Projects\Tests\SubBufferSingleTest\SubBufferSingle.data\bin\SubBufferSingle_2020-12-28T00_26_27\SubBufferSingle.k7z')



# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(150050)
CYCLES = 5
TRIGGER_DELAY = 45
module.DAQconfig(1, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0001)

data_in_list = np.arange(1, 30001, 1)
for j in range(CYCLES):
    for din in data_in_list:
        module.PXItriggerWrite(7, 1)
        module.PXItriggerWrite(7, 0)
        write_FPGA_register(module, "Register_Bank_data_in", din)
    module.DAQtriggerMultiple(0b0001)
    time.sleep(0.1)

# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQreadArray(1, TIMEOUT)


# exiting...
module.close()
print("AIN closed")


plt.figure()
for i in range(CYCLES):
    plt.plot(dataRead1[i,0::5])

