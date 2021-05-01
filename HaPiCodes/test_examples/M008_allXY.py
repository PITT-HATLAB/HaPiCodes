import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from HaPiCodes.test_examples import msmtInfoSel




def allXY(yamlFile=msmtInfoSel.cwYaml, plot=0):
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = bmp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.allXY()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    Id, Qd = f.processDataReceive(pxi.subbuffer_used, dataReceive, plot=plot)
    f.rotateData(Id, Qd)
    return (W, Q, dataReceive, Id, Qd)

if __name__ == '__main__':
    msmt = allXY()
