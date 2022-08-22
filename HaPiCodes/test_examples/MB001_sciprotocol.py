import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments
from HaPiCodes.data_process import postSelectionProcess as psdp
from HaPiCodes.test_examples import msmtInfoSel

def sciprotocol(yamlFile=msmtInfoSel.cwYaml, ampArray=np.linspace(0, 0.9, 100), plot=1, update=0):
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = bmp.BasicExperiments(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.sciprotocol(ampArray)
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    IQdata = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=1)
    Id, Qd = f.average_data(IQdata.I_rot, IQdata.Q_rot)
    piPulseAmp = f.pi_pulse_tune_up(Id, Qd, updatePiPusle_amp=update, plot=plot)

    selData = psdp.PostSelectionData(Id, Qd, msmtInfoDict, [1, 0], plotGauFitting=0)
    selMask = selData.mask_g_by_circle(sel_idx=0, plot=0)
    I_vld, Q_vld = selData.sel_data(selMask, plot=0)
    return (dataReceive, Id, Qd, piPulseAmp, I_vld, Q_vld)

if __name__ == '__main__':

    ampArray = np.linspace(0, 0.9, 100)

    dataReceive, Id, Qd, piPulseAmp, I_vld, Q_vld = sciprotocol(ampArray)

    N = len(I_vld)

    tomo_axis_i = np.zeros(N)
    initial_state_i = np.zeros(N)
    amp_i = np.zeros(N)

    for i in range(len(I_vld)):
        I_i = I_vld[i]
        Q_i = Q_vld[i]

        tomo_axis_i = i % 3
        initial_state_i = i % 9 // 3
        amp_i = ampArray[i // 9]








