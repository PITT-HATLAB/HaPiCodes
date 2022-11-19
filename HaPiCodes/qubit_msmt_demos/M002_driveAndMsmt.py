import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.qubit_msmt_demos import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from HaPiCodes.qubit_msmt_demos import msmtInfoSel

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
f.yamlFile = msmtInfoSel.cwYaml

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = bmp.BasicExperiments(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt(driveQubit=True)
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    IQData = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=1)