from package import package_PathWave as PW
from package import package_pulseGenerate as pG
import numpy as np
import matplotlib.pyplot as plt

digFPGAName = "C:\\PXI_FPGA\\Projects\\Origional\\Origional.data\\bin\\Origional_2020-12-01T19_03_56\\Origional.k7z"
pointPerCycle = 200
cycles = 5
chanNum = 4


def runMsmt():
    module_dict = PW.open_modules()
    for module in module_dict:
        if module_dict[module].instrument.getProductName() != "M3102A":
            PW.configAWG(module_dict[module])
        else:
            PW.configDig(module_dict[module], fpgaName=digFPGAName, manualReloadFPGA=0)
    W, Q = pG.piPulseTuneUp()

    for i in range(1, chanNum + 1):
        module_dict['D1'].instrument.DAQconfig(i, pointPerCycle, cycles, 0, 1)