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
from AddOns.DigFPGAConfig_Demodulate import configFPGA, get_recommended_truncation
# with open(r"sysInfo.json") as file_:
#     info = json.load(file_)
triggerAwgDigDelay = 330 #info['sysConstants']['triggerAwgDigDelay']
pulse_general_dict = {'relaxingTime': 50, # float; precision: 0.01us; relax after the start of the firt pulse
                      'avgNum': 10000}

def sin_gen(x, phase=0):
    return (32767 * np.sin(2 * np.pi * x / 10 + phase)).astype(int)

if __name__ == '__main__':

    pointPerCycle = 3000
    cycles = 10000
    start = time.time()
    FPGA_file = r'N:\Chao\HaPiCodes\PathwaveFPGATests\FPGA\Origional\Origional.data\bin\Origional_2020-12-01T19_03_56\Origional.k7z'
    module_dict = pwh.openAndConfigAllModules(FPGA_file)

    WEIGHT_data = np.zeros(100 // 5, dtype=int) + 2 ** 13 - 1
    for i in range(1, 5):
        module_dict["D1"].instrument.DAQconfig(i, pointPerCycle, cycles, 0, 1)
        # configFPGA(module_dict["D1"].instrument,i, 18)


    W, Q, hvi = pwh.definePulseAndUpload(amp.driveAndMsmt, module_dict, pulse_general_dict)

    dataReceive = pwh.digReceiveData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='1111', timeout=2000)
    pwh.releaseHviAndCloseModule(hvi, module_dict)

    dp.plotOriginal(dataReceive, 4, 0, weightAndTrig=0)
    dd_list = []
    for idx in range(cycles):
        xdata = np.arange(pointPerCycle)
        data = dataReceive['4'][idx]
        d1 = data * sin_gen(xdata)
        dd = [np.sum(d1[i*5: i*5+10])  for i in range(pointPerCycle//5-2)]
        dd = np.array(dd, dtype=np.int64) >> 18
        dd_list.append(dd)
    dd_avg = np.average( np.array(dd_list), axis=0)
    plt.figure()
    plt.plot(dd_avg)


    plt.show()