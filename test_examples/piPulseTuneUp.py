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


def piPulseTuneUp(plot=1, update=0):
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.piPulseTuneUp()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    Id, Qd = f.processDataReceive(pxi.subbuffer_used, dataReceive)
    piPulseAmp = f.pi_pulse_tune_up(Id, Qd, updatePiPusle_amp=update, plot=plot)
    return (W, Q, dataReceive, Id, Qd, piPulseAmp)

if __name__ == '__main__':
    msmt = piPulseTuneUp()
