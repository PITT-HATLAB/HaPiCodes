import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from test_examples import msmtInfoSel
from data_process import package_dataProcess as dp

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
msmtInfoDict['moduleConfig']['D1']['FPGA'] = 'Demodulate_showWeight'

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt()
    pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()

    demod_I, demod_Q, mag2 = dp.plotPreRrun(dataReceive['D1'], [1,2], sumRange=[1000, 4000])
    dp.get_recommended_truncation(dataReceive['D1']['ch1'][:, 0, 0::5], dataReceive['D1']['ch1'][:, 0, 1::5], 1000, 4000, 18)
    dp.get_recommended_truncation(dataReceive['D1']['ch2'][:, 0, 0::5], dataReceive['D1']['ch2'][:, 0, 1::5], 1000, 4000, 18)
    # plt.figure()
    # plt.plot(dataReceive['D1']['ch1'][0, 0, 4::5] * 30000)
    # plt.plot(dataReceive['D1']['ch1'][0, 0, 3::5])
    # plt.plot(dataReceive['D1']['ch1'][0, 0, 2::5])
    # #
    #
    #
    # Idata = np.average(dataReceive["D1"]['ch1'], axis=0)[2::5]
    # Qdata = np.average(dataReceive["D1"]['ch1'], axis=0)[3::5]
    # from data_process import fit_all as fa
    #
    # fa.t1_fit(Idata, Qdata, timeArray/1e3)