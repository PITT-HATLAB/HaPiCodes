import sys
import time
from typing import List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

import keysightSD1 as SD1

from AddOns.SD1AddOns import AIN
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
    r'C:\PXI_FPGA\Projects\Tests\PhaseCorrectionTest\PhaseCorrectionTest.data\bin\PhaseCorrectionTest_2020-12-23T22_39_27\PhaseCorrectionTest.k7z')


def write_mem_data(sig_phase, ref_phase):
    # PUT DATA IN ON-BOARD MEMORY AS INPUT DATA
    DIN_x = np.arange(0, 1000, 2)
    DIN_shape = np.zeros(len(DIN_x)) ** 3 + 0.9
    MEM_DIN_s = 32767 * np.sin(np.pi / 10 * DIN_x + sig_phase) * DIN_shape
    MEM_DIN_r = 15000 * np.sin(np.pi / 10 * DIN_x + ref_phase) * (DIN_shape + np.random.rand(len(DIN_x)) * 0.5)
    MEM_DIN_s = MEM_DIN_s.astype(int)
    MEM_DIN_r = MEM_DIN_r.astype(int)
    for i in range(5):
        config_mem_reader(module, MEM_DIN_s[i::5], 2, len(DIN_x) // 5, f"Host_mem_{i + 1}")
        config_mem_reader_1(module, MEM_DIN_r[i::5], 2, len(DIN_x) // 5, f"Host_mem_{i + 7}")

    # WRITE WEIGHT FUNCTION
    WEIGHT_data = np.zeros(len(DIN_x) // 5, dtype=int) + 2 ** 15 - 1
    config_mem_reader(module, WEIGHT_data, 7, len(DIN_x) // 5, "Host_mem_6",
                      delay_reg_name="WeightReaderRegisters_rd_trig_delay",
                      length_reg_name="WeightReaderRegisters_rd_length")
    config_mem_reader_1(module, WEIGHT_data, 7, len(DIN_x) // 5, "Host_mem_12",
                        delay_reg_name="WeightReaderRegisters_1_rd_trig_delay",
                        length_reg_name="WeightReaderRegisters_1_rd_length")


# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = int(800)
CYCLES = 100
TRIGGER_DELAY = 0
for i in [1, 2, 3]:
    module.DAQconfig(i, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, SD1.SD_TriggerModes.SWHVITRIG)
module.DAQstartMultiple(0b0111)

# CONFIGURE DEMODULATOR and IQINTEGRATOR
config_IQIntegrator(module, 18, 50)  # stop-start-1 points. start from start+1, the maximum integration cycle is 2**16-1, maximum trig delay is 2**31-1
config_IQIntegrator_1(module, 18, 50)
write_FPGA_register(module, "SliceAfterPhaseCorrection_lower_bit", 8)

ref_phase_list = np.random.rand(CYCLES)*2*np.pi
sig_phase_list = np.linspace(0, 2 * np.pi, CYCLES) - ref_phase_list

for i in range(CYCLES):
    sp = sig_phase_list[i]
    rp = ref_phase_list[i]
    write_mem_data(sp, rp)
    module.DAQtriggerMultiple(0b0111)



# READ DATA
TIMEOUT = 1
dataRead1 = module.DAQreadArray(1, TIMEOUT)
dataRead2 = module.DAQreadArray(2, TIMEOUT)
dataRead3 = module.DAQreadArray(3, TIMEOUT)

# exiting...
module.close()
print("AIN closed")

# PROCESSING
xdata = np.arange(POINTS_PER_CYCLE) * 2
demod_out_Is = dataRead1[:, 0::5]
demod_out_Qs = dataRead1[:, 1::5]
integ_out_Is = dataRead1[:, 2::5]
integ_out_Qs = dataRead1[:, 3::5]
weight_outs = dataRead1[:, 4::5]
demod_out_Ir = dataRead2[:, 2::5]
demod_out_Qr = dataRead2[:, 3::5]
integ_out_Ir = dataRead2[:, 0::5]
integ_out_Qr = dataRead2[:, 1::5]
weight_outr = dataRead2[:, 4::5]

subtract_data_I = dataRead3[:, 0::5]
subtract_data_Q = dataRead3[:, 1::5]
sum_data_I = dataRead3[:, 2::5]
sum_data_Q = dataRead3[:, 3::5]

I_list_s = integ_out_Is[:, -1]
Q_list_s = integ_out_Qs[:, -1]
I_list_r = integ_out_Ir[:, -1]
Q_list_r = integ_out_Qr[:, -1]
I_list_subtract = subtract_data_I[:, -1]
Q_list_subtract = subtract_data_Q[:, -1]
I_list_sum = sum_data_I[:, -1]
Q_list_sum = sum_data_Q[:, -1]



def plot_IQ(I_, Q_):
    fig, axs = plt.subplots(2, 2,figsize=(8,8))
    complex_data = I_ + 1j * Q_
    axs[0, 0].plot(I_, Q_, '*')
    axs[0, 1].plot(I_)
    axs[0, 1].plot(Q_)
    axs[1, 0].plot(np.abs(complex_data))
    axs[1, 1].plot(np.angle(complex_data))

plot_IQ(I_list_s, Q_list_s)
plot_IQ(I_list_r, Q_list_r)

plot_IQ(I_list_subtract, Q_list_subtract)
plot_IQ(I_list_sum, Q_list_sum)


'''
plt.figure()
idx = 2
plt.plot(xdata[::5], demod_out_Is[idx], label="demod_I")
plt.plot(xdata[::5], demod_out_Qs[idx], label="demod_Q")
plt.plot(xdata[::5], integ_out_Is[idx], label="integ_I")
plt.plot(xdata[::5], integ_out_Qs[idx], label="integ_Q")
plt.legend()
'''



plt.figure()
idx = 2
plt.plot(xdata[::5], integ_out_Is[idx], label="integ_I")
plt.plot(xdata[::5], integ_out_Qs[idx], label="integ_Q")
plt.plot(xdata[::5], subtract_data_I[idx], label="sub_I")
plt.plot(xdata[::5], subtract_data_Q[idx], label="sub_Q")

plt.legend()