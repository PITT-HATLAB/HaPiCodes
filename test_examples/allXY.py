import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from data_process import package_fittingAndDataProcess as f
from pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from test_examples import msmtInfoSel
from sd1_api.SD1AddOns import get_FPGA_register

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
f.yamlFile = msmtInfoSel.cwYaml


def allXY(plot=1):
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.allXY()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    Id, Qd = f.processDataReceive(pxi.subbuffer_used, dataReceive, plot=plot)
    f.rotateData(Id, Qd)

if __name__ == '__main__':
    allXY()
