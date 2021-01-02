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
    MEM_DIN_s = 32767 * np.sin(np.pi / 10 * DIN_x + sig_phase) * DIN_shape
    MEM_DIN_r = 15000 * np.sin(np.pi / 10 * DIN_x + ref_phase) * (DIN_shape + np.random.rand(len(DIN_x)) * 0.1)
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
POINTS_PER_CYCLE = 100
CYCLES = 4
TRIGGER_DELAY = 45

# module.FPGAreset(2) ############################### RESET FPGA ########################################################

for i in [1, 2]:
    module.DAQconfig(i, POINTS_PER_CYCLE*5, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0011)

# CONFIGURE DEMODULATOR and IQINTEGRATOR
sig_config = FPGARegisters(18, 50, 2, 100, 18, 5)
ref_config = FPGARegisters(18, 50, 2, 100, 18, 5)
config_FPGA_registers(module, 1, sig_config)
config_FPGA_registers(module, 2, ref_config)
config_ref_channels(module, [2,2,2,2], [False, False, False, False])

np.random.seed(0)
ref_phase_list = np.linspace(0, 6 * np.pi, POINTS_PER_CYCLE * CYCLES)
sig_phase_list = np.linspace(0, 6 * np.pi, POINTS_PER_CYCLE * CYCLES) #+ ref_phase_list

for j in range(CYCLES):
    for i in range(POINTS_PER_CYCLE):
        sp = sig_phase_list[i + POINTS_PER_CYCLE * j]
        rp = ref_phase_list[i + POINTS_PER_CYCLE * j]
        write_mem_data(sp, rp)
        time.sleep(0.02)
        # module.DAQtriggerMultiple(0b1100)
        module.PXItriggerWrite(7, 1)
        module.PXItriggerWrite(7, 0)
        module.PXItriggerWrite(7, 1)
        time.sleep(0.1)

    module.DAQtriggerMultiple(0b0011)
    time.sleep(0.02)


# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQreadArray(1, TIMEOUT)
dataRead2 = module.DAQreadArray(2, TIMEOUT)

# exiting...
module.close()
print("AIN closed")

# PROCESSING

S_I = dataRead1[:, 0::5]
S_Q = dataRead1[:, 1::5]
S_I_rot = dataRead1[:, 2::5]
S_Q_rot = dataRead1[:, 3::5]
R_I = dataRead2[:, 0::5]
R_Q = dataRead2[:, 1::5]
R_I_rot = dataRead2[:, 2::5]
R_Q_rot = dataRead2[:, 3::5]


def plot_IQ(I_, Q_):
    fig, axs = plt.subplots(2, 2,figsize=(8,8))
    complex_data = I_ + 1j * Q_
    axs[0, 0].plot(I_, Q_, '*')
    axs[0, 1].plot(I_)
    axs[0, 1].plot(Q_)
    axs[1, 0].plot(np.abs(complex_data))
    axs[1, 1].plot(np.unwrap(np.angle(complex_data)) - sig_phase_list )

# idx = 1
# plot_IQ(S_I[idx], S_Q[idx])
# plot_IQ(S_I_rot[idx], S_Q_rot[idx])
# plot_IQ(R_I[idx], R_Q[idx])
# plot_IQ(R_I_rot[idx], R_Q_rot[idx])


S_I_all = (S_I).flatten()
S_Q_all = (S_Q).flatten()
R_I_all = (R_I).flatten()
R_Q_all = (R_Q).flatten()
S_I_rot_all = (S_I_rot).flatten()
S_Q_rot_all = (S_Q_rot).flatten()
plot_IQ(S_I_all, S_Q_all)
plot_IQ(R_I_all, R_Q_all)
