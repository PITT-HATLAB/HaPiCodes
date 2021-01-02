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
    r'C:\PXI_FPGA\Projects\Tests\PhaseCorrectionOnly\PhaseCorrectionOnly.data\bin\PhaseCorrectionOnly_2020-12-23T20_06_47\PhaseCorrectionOnly.k7z')


def write_mem_data(sig_phase_list, ref_phase_list, ref_amp_list):
    # PUT DATA IN ON-BOARD MEMORY AS INPUT DATA
    data_len = len(sig_phase_list)
    MEM_DIN_SI = (32700 * np.cos(sig_phase_list)).astype(int)
    MEM_DIN_SQ = (32700 * np.sin(sig_phase_list)).astype(int)
    MEM_DIN_RI = (ref_amp_list * np.cos(ref_phase_list)).astype(int)
    MEM_DIN_RQ = (ref_amp_list * np.sin(ref_phase_list)).astype(int)

    config_mem_reader(module, MEM_DIN_SI, 2, data_len, f"Host_mem_1")
    config_mem_reader(module, MEM_DIN_SQ, 2, data_len, f"Host_mem_2")
    config_mem_reader(module, MEM_DIN_RI, 2, data_len, f"Host_mem_3")
    config_mem_reader(module, MEM_DIN_RQ, 2, data_len, f"Host_mem_4")



# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(800)
CYCLES = 1
TRIGGER_DELAY = 0
for i in [1, 2, 3]:
    module.DAQconfig(i, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0111)




ref_phase_list = np.random.rand(POINTS_PER_CYCLE//5)*2*np.pi
ref_amp_list = 32767/ (np.random.rand(POINTS_PER_CYCLE//5) + 1)
sig_phase_list = np.linspace(0, 2 * np.pi, POINTS_PER_CYCLE//5) + ref_phase_list

write_mem_data(sig_phase_list, ref_phase_list, ref_amp_list)
write_FPGA_register(module, "SliceAfterPhaseCorrection_lower_bit", 8)


module.DAQtriggerMultiple(0b0111)

# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQreadArray(1, TIMEOUT)
dataRead2 = module.DAQreadArray(2, TIMEOUT)
dataRead3 = module.DAQreadArray(3, TIMEOUT)

# exiting...
module.close()
print("AIN closed")


data_out_Is = dataRead1[0, 0::5]
data_out_Qs = dataRead1[0, 1::5]
data_out_Ir = dataRead1[0, 2::5]
data_out_Qr = dataRead1[0, 3::5]


subtract_data_I = dataRead3[0, 0::5]
subtract_data_Q = dataRead3[0, 1::5]
sum_data_I = dataRead3[0, 2::5]
sum_data_Q = dataRead3[0, 3::5]



def plot_IQ(I_, Q_):
    fig, axs = plt.subplots(2, 2,figsize=(8,8))
    complex_data = I_ + 1j * Q_
    axs[0, 0].plot(I_, Q_, '*')
    axs[0, 1].plot(I_)
    axs[0, 1].plot(Q_)
    axs[1, 0].plot(np.abs(complex_data))
    axs[1, 1].plot(np.angle(complex_data))


plot_IQ(data_out_Is, data_out_Qs)
plot_IQ(data_out_Ir, data_out_Qr)

plot_IQ(subtract_data_I, subtract_data_Q)
plot_IQ(sum_data_I, sum_data_Q)


