import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from data_process import package_fittingAndDataProcess as f
from pathwave.pxi_instruments import PXI_Instruments
from test_examples import msmtInfoSel

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
f.yamlFile = msmtInfoSel.cwYaml


def t2RMsmt(plot=1):
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.t2RMsmt()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    Id, Qd = f.processDataReceive(pxi.subbuffer_used, dataReceive)
    t2R, detune = f.t2_ramsey_fit(Id, Qd, plot=plot)
    return (W, Q, dataReceive, Id, Qd, t2R, detune)


if __name__ == '__main__':
    msmt = t2RMsmt()