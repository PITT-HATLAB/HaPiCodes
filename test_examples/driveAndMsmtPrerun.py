import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from pathwave.pxi_instruments import PXI_Instruments
from data_process import package_fittingAndDataProcess as f
from test_examples import msmtInfoSel
from data_process import package_dataProcess as dp

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
msmtInfoDict['moduleConfig']['D1']['FPGA'] = 'Demodulate_showWeight'
msmtInfoDict['sequeceAvgNum'] = 10000
f.yamlFile = msmtInfoSel.cwYaml

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt()
    pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    # demod_I, demod_Q, mag2 = f.processDataReceive(pxi.subbuffer_used, dataReceive, plot=1)

    IQdata = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=1)
    
