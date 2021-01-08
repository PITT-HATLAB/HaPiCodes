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

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.selectionDriveAndMsmt()
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()

    Id, Qd = f.processDataReceiveWithMultipleMsmt(pxi.subbuffer_used, dataReceive, plot=1)
    data = np.array([np.array(Id).flatten(), np.array(Qd).flatten()])

    fitRes = f.fit_Gaussian(data)
    sigma = np.sqrt(fitRes[4] ** 2 + fitRes[5] ** 2)
    I_vld, Q_vld = f.post_sel(data, fitRes[0], fitRes[1], sigma, [0], plot_check=1)
    g_pct = f.cal_g_state([I_vld[0], Q_vld[0]], fitRes[0], fitRes[1], fitRes[2], fitRes[3], plot=1)
    print(g_pct)
