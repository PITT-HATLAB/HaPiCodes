import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

from AddOns.SD1AddOns import AIN
from AddOns.DigFPGAConfig import config_mem_reader, config_IQIntegrator


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
    r'C:\PXI_FPGA\Projects\Tests\IQIntegratorTest\IQIntegratorTest.data\bin\IQIntegratorTest_2020-12-20T20_06_54\IQIntegratorTest.k7z')



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
config_IQIntegrator(module, 18, 50) # stop-start-1 points. start from start+1, the maximum integration cycle is 2**16-1, maximum trig delay is 2**31-1

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
dataRead1 = module.DAQreadArray(1, TIMEOUT)
dataRead2 = module.DAQreadArray(2, TIMEOUT)
dataRead3 = module.DAQreadArray(3, TIMEOUT)

# exiting...
module.close()
print()
print("AIN closed")

# processing
# plt.figure()
# plt.title("input data")
# plt.plot(DIN_x, MEM_DIN)

demod_out_I = dataRead2[:, 0::5]
demod_out_Q = dataRead2[:, 1::5]
integ_out_I = dataRead2[:, 2::5]
integ_out_Q = dataRead2[:, 3::5]
daq1_trig_out = dataRead1[:, 3::5]
weight_out = dataRead1[:, 4::5]
mem_out = dataRead3
xdata = np.arange(POINTS_PER_CYCLE) * 2

plt.figure()
plt.plot(xdata, mem_out[0], label="input data")
plt.plot(xdata[::5], weight_out[0], label="weight func")
plt.plot(xdata[::5], demod_out_I[0], label="demod_I")
plt.plot(xdata[::5], demod_out_Q[0], label="demod_Q")
plt.plot(xdata[::5], integ_out_I[0], label="integ_I")
plt.plot(xdata[::5], integ_out_Q[0], label="integ_Q")
plt.plot(xdata[::5], daq1_trig_out[0]*30000, label="DAQ1 trig")
plt.legend()



# TEST ADDON FUNCS
from AddOns.DigFPGAConfig import get_recommended_truncation
dt_, it_ = get_recommended_truncation(demod_out_I, demod_out_Q, 180, 680, 18)
print (dt_, it_)