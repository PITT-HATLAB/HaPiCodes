import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments
from HaPiCodes.test_examples import msmtInfoSel

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

    Id, Qd = dataReceive["D1"]["ch1"][:, 2::5], dataReceive["D1"]["ch1"][:, 3::5]
    data = np.array([np.array(Id).flatten(), np.array(Qd).flatten()])

    fitRes = f.fit_Gaussian(data)
    sigma = np.sqrt(fitRes[4] ** 2 + fitRes[5] ** 2)
    I_vld, Q_vld = f.post_sel(Id, Qd, fitRes[0], fitRes[1], sigma, 2, plot_check=1)
    g_pct = f.cal_g_pct([I_vld[0], Q_vld[0]], fitRes[0], fitRes[1], fitRes[2], fitRes[3], plot=1)
    print(g_pct)
