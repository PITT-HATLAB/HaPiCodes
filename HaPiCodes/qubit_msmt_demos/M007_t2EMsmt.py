import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.qubit_msmt_demos import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments
from HaPiCodes.qubit_msmt_demos import msmtInfoSel


def t2EMsmt(yamlFile=msmtInfoSel.cwYaml, plot=1):
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = bmp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.t2EMsmt()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    IQdata = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=1)
    Id, Qd = f.average_data(IQdata.I_rot, IQdata.Q_rot)
    t2E = f.t2_echo_fit(Id, Qd, plot=plot)
    return (W, Q, dataReceive, Id, Qd, t2E)


if __name__ == '__main__':
    msmt = t2EMsmt()
