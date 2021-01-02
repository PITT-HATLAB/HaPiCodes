import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

from AddOns.SD1AddOns import AIN
from AddOns.DigFPGAConfig_Demodulate import config_demodulator, config_weight_func, get_recommended_truncation

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
# error = module.FPGAload(
#     r'C:\PXI_FPGA\Projects\Demodulate_showWeight\Demodulate_showWeight.data\bin\Demodulate_showWeight_2020-12-24T21_59_37\Demodulate_showWeight.k7z')

error = module.FPGAload(
    r'C:\PXI_FPGA\Projects\Origional\Origional.data\bin\Origional_2020-12-01T19_03_56\Origional.k7z')

DIN_x = np.arange(0, 1000, 2) ##########################
WEIGHT_data = np.zeros(len(DIN_x) // 5, dtype=int) + 2 ** 15 - 1
for i in range(4):
    config_weight_func(module, i+1, WEIGHT_data, 2, len(DIN_x)//5)
config_demodulator(module, [18, 18, 18, 18])


# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(500)
CYCLES = 1
TRIGGER_DELAY = 0
for i in [1, 2, 3]:
    module.DAQconfig(i, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0111)

# DAQ Trigger
module.DAQtriggerMultiple(0b0111)


# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQreadArray(1, TIMEOUT)
dataRead2 = module.DAQreadArray(2, TIMEOUT)
dataRead3 = module.DAQreadArray(3, TIMEOUT)
data_read_all = np.array([dataRead1, dataRead2, dataRead3])

# exiting...
module.close()
print()
print("AIN closed")

# processing
demod_I = data_read_all[:, :, 0::5]
demod_Q = data_read_all[:, :, 1::5]
mag2 = data_read_all[:, :, 2::5]
weight = data_read_all[:, :, 3::5]
HVI_trig = data_read_all[:, :, 4::5]*30000

def plot_result(ch, cyc):
    plt.figure()
    plt.plot(demod_I[ch, cyc])
    plt.plot(demod_Q[ch, cyc])
    plt.plot(mag2[ch, cyc])
    plt.plot(weight[ch, cyc])
    plt.plot(HVI_trig[ch, cyc])


plot_result(1,0)