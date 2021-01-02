import numpy as np
import matplotlib.pyplot as plt
import package_pulse as pp
import time
import package_pathWaveAndHvi as pwh
import package_allMsmtPulses as amp
import package_dataProcess as dp
from collections import OrderedDict
import json
import keysightSD1
import warnings
import keysight_hvi as kthvi
from AddOns.DigFPGAConfig_QubitMSMT import *
from scipy.optimize import curve_fit as cf


triggerAwgDigDelay = 330 #info['sysConstants']['triggerAwgDigDelay']

pulse_general_dict = {'relaxingTime': 600, # float; precision: 0.01us; relax after the start of the firt pulse
                      'avgNum': 100}


timeArray = np.linspace(0, 300000, 101)[:100]
if __name__ == '__main__':

    pointPerCycle = 50000
    cycles = 20
    start = time.time()
    FPGA_file = r'C:\PXI_FPGA\Projects\Qubit_MSMT\Qubit_MSMT.data\bin\Qubit_MSMT_2020-12-28T14_28_28\Qubit_MSMT.k7z'


    module_dict = pwh.openAndConfigAllModules(FPGA_file)
    for i in range(1, 5):
        module_dict["D1"].instrument.DAQconfig(i, pointPerCycle, cycles, 45, 1)
    W, Q, hvi = pwh.definePulseAndUpload(amp.t1MsmtReal, module_dict, pulse_general_dict, subbuffer_used=1)


    WEIGHT_data = np.zeros(100 // 5, dtype=int) + 2 ** 13 - 1
    # WRITE WEIGHT FUNCTION
    write_FPGA_memory(module_dict["D1"].instrument, "WeightFunc_1", WEIGHT_data)
    write_FPGA_memory(module_dict["D1"].instrument, "WeightFunc_2", WEIGHT_data)

    sig_config = FPGARegisters(100, 300, 1, 100, 17, 5)
    ref_config = FPGARegisters(100, 300, 1, 100, 18, 7)
    config_FPGA_registers(module_dict["D1"].instrument, 1, sig_config)
    config_FPGA_registers(module_dict["D1"].instrument, 2, ref_config)
    config_ref_channels(module_dict["D1"].instrument, [2, 2, 2, 2], [False, False, False, False])


    dataReceive = pwh.digReceiveData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='0011', timeout=20000, subbuffer_used=1)

    Idata = np.average(dataReceive['1'], axis=0)[2::5]
    Qdata = np.average(dataReceive['1'], axis=0)[3::5]
    I = np.average(Idata.reshape([100, 100]), axis=0)
    Q = np.average(Qdata.reshape([100, 100]), axis=0)

    pwh.releaseHviAndCloseModule(hvi, module_dict)

    import fit_all as fa
    fa.t1_fit(I, Q, timeArray/1e3)
    # def t1fit(xdata, ydata):
    #     def exponetialDecay(x, tc, amp, b, x0):
    #         return amp * np.exp(-(x+x0)/tc) + b
    #     popt, pcov = cf(exponetialDecay, xdata[:], ydata[:], bounds=([0, 0, -30000, 0], [80e3, 33000, 30000, 20000]))
    #     plt.figure()
    #     plt.plot(xdata, ydata, '*', label='data')
    #     plt.plot(xdata, exponetialDecay(xdata, *popt), label='fit')
    #     plt.legend()
    #     return popt, pcov
    
    # popt, pcov = t1fit(timeArray, I)
    # print('T1 is', np.round(popt[0]/1e3, 5), 'us')
    
    