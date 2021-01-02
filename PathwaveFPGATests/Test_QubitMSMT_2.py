import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

from AddOns.SD1AddOns import AIN
from AddOns.DigFPGAConfig_QubitMSMT import *

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
    r'C:\PXI_FPGA\Projects\Tests\QubitMSMTTest\QubitMSMTTest.data\bin\QubitMSMTTest_2020-12-28T01_32_22\QubitMSMTTest.k7z')


def write_mem_data(sig_phase, ref_phase):
    # PUT DATA IN ON-BOARD MEMORY AS INPUT DATA
    DIN_x = np.arange(0, 1000, 2)
    DIN_shape = np.zeros(len(DIN_x)) ** 3 + 0.9
    # MEM_DIN_s = 32767 * np.sin(np.pi / 10 * DIN_x + sig_phase) * DIN_shape
    MEM_DIN_s = 32767 *  DIN_shape
    MEM_DIN_r = 15000 * np.sin(np.pi / 10 * DIN_x + ref_phase) * (DIN_shape + np.random.rand(len(DIN_x)) * 0.5)
    MEM_DIN_s = MEM_DIN_s.astype(int)
    MEM_DIN_r = MEM_DIN_r.astype(int)
    for i in range(5):
        config_mem_reader(module, MEM_DIN_s[i::5], 2, len(DIN_x) // 5, f"Host_mem_{i + 1}")
        config_mem_reader_1(module, MEM_DIN_r[i::5], 2, len(DIN_x) // 5, f"Host_mem_{i + 6}")

    # WRITE WEIGHT FUNCTION
    WEIGHT_data = np.zeros(len(DIN_x) // 5, dtype=int) + 2 ** 15 - 1
    write_FPGA_memory(module, "WeightFunc_1", WEIGHT_data)
    write_FPGA_memory(module, "WeightFunc_2", WEIGHT_data)



# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(8000000)
CYCLES = 1
TRIGGER_DELAY = 0
for i in [3, 4]:
    module.DAQconfig(i, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b1100)

# CONFIGURE DEMODULATOR and IQINTEGRATOR
sig_config = FPGARegisters(18, 50, 2, 100, 18, 5)
ref_config = FPGARegisters(18, 50, 2, 100, 18, 6)
config_FPGA_registers(module, 1, sig_config)
config_FPGA_registers(module, 2, ref_config)
config_ref_channels(module, [2,2,2,2], [False, False, False, False])


ref_phase_list = np.random.rand(POINTS_PER_CYCLE)*2*np.pi
sig_phase_list = np.linspace(0, 2 * np.pi, POINTS_PER_CYCLE) #- ref_phase_list

for i in range(CYCLES):
    sp = sig_phase_list[i]
    rp = ref_phase_list[i]
    write_mem_data(sp, rp)
    module.DAQtriggerMultiple(0b1101)
    time.sleep(0.01)
    module.DAQtriggerMultiple(0b1101)
    time.sleep(1)


# module.DAQtriggerMultiple(0b0011)
# READ DATA
TIMEOUT = 1
dataRead3 = module.DAQreadArray(3, TIMEOUT)
dataRead4 = module.DAQreadArray(4, TIMEOUT)

# exiting...
module.close()
print("AIN closed")

# PROCESSING
xdata = np.arange(POINTS_PER_CYCLE) * 2
daq_1_trig = dataRead3[:, 0::5]*30000
daq_3_trig = dataRead3[:, 1::5]*20000
integ_I = dataRead3[:, 2::5]
integ_Q = dataRead3[:, 3::5]
PXI_trig_7 = dataRead3[:, 4::5] * 40000
data_out = dataRead4




plt.figure()
idx = 0
plt.plot(xdata[::5], daq_1_trig[idx], label="daq_1_trig")
plt.plot(xdata[::5], daq_3_trig[idx], label="daq_3_trig")
plt.plot(xdata[::5], integ_I[idx], label="integ_I")
plt.plot(xdata[::5], integ_Q[idx], label="integ_Q")
plt.plot(xdata[::5], PXI_trig_7[idx], label="PXIe7")
plt.plot(xdata, data_out[idx], label="data_out")
plt.legend()

# here LO phase reset is connected to integration trigger. This actually requires that the integ_trig must be in low during the integration range.