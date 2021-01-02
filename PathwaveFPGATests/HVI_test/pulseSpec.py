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

# with open(r"sysInfo.json") as file_:
#     info = json.load(file_)
triggerAwgDigDelay = 330 #info['sysConstants']['triggerAwgDigDelay']

pulse_general_dict = {'relaxingTime': 1000, # float; precision: 0.01us; relax after the start of the firt pulse
                      'avgNum': 2000}



if __name__ == '__main__':

    pointPerCycle = 10000
    cycles = 1
    start = time.time()
    FPGA_file = r'C:\PXI_FPGA\Projects\Qubit_MSMT\Qubit_MSMT.data\bin\Qubit_MSMT_2020-12-28T14_28_28\Qubit_MSMT.k7z'


    module_dict = pwh.openAndConfigAllModules(FPGA_file)
    for i in range(1, 5):
        module_dict["D1"].instrument.DAQconfig(i, pointPerCycle, cycles, 45, 1)
    W, Q, hvi = pwh.definePulseAndUpload(amp.pulseSpecReal, module_dict, pulse_general_dict, subbuffer_used=1)


    WEIGHT_data = np.zeros(100 // 5, dtype=int) + 2 ** 13 - 1
    # WRITE WEIGHT FUNCTION
    write_FPGA_memory(module_dict["D1"].instrument, "WeightFunc_1", WEIGHT_data)
    write_FPGA_memory(module_dict["D1"].instrument, "WeightFunc_2", WEIGHT_data)

    sig_config = FPGARegisters(100, 300, 1, 100, 17, 9)
    ref_config = FPGARegisters(100, 300, 1, 100, 18, 7)
    config_FPGA_registers(module_dict["D1"].instrument, 1, sig_config)
    config_FPGA_registers(module_dict["D1"].instrument, 2, ref_config)
    config_ref_channels(module_dict["D1"].instrument, [2, 2, 2, 2], [False, False, False, False])


    freqRange = 3.06764e9 + np.linspace(-5, 5, 101) * 1e6
    Ilist = np.zeros(len(freqRange))
    Qlist = np.zeros(len(freqRange))
    plt.figure()
    for i in range(len(freqRange)):
        SC_Qubit.set_frequency(freqRange[i])
        SC_Qubit.set_output_status(0)
        dataReceive = pwh.digReceiveData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='0011', timeout=20000, subbuffer_used=1)
        Idata = np.average(dataReceive['1'].flatten()[2::5])
        Qdata = np.average(dataReceive['1'].flatten()[3::5])
        plt.plot(freqRange[i], Idata, 'r*')
        plt.plot(freqRange[i], Qdata, 'r*')
    
    
        SC_Qubit.set_output_status(1)
        dataReceive = pwh.digReceiveData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='0011', timeout=20000, subbuffer_used=1)
        Idata = np.average(dataReceive['1'].flatten()[2::5])
        Qdata = np.average(dataReceive['1'].flatten()[3::5])
        Ilist[i] = Idata
        Qlist[i] = Qdata
        plt.plot(freqRange[i], Idata, 'b*')
        plt.plot(freqRange[i], Qdata, 'b*')
        plt.pause(0.1)

    # dataReceive = pwh.digReceiveData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='0011', timeout=20000,
    #                                  subbuffer_used=1)
    # plt.hist2d(dataReceive['1'].flatten()[2::5], dataReceive['1'].flatten()[3::5], bins=101)
    # plt.colorbar()

    pwh.releaseHviAndCloseModule(hvi, module_dict)
