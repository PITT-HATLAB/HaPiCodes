import yaml
import numpy as np
from pulse import allMsmtPulses as amp
from pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
import h5py
from test_examples import msmtInfoSel

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))

timeArray = np.linspace(0, 300000, 101)[:100]
if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    W, Q = amp.t1MsmtReal(pxi.module_dict)

    # pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    # pxi.uploadPulseAndQueue()
    # dataReceive = pxi.runExperiment(timeout=20000)
    # pxi.releaseHviAndCloseModule()



    # Idata = np.average(dataReceive["D1"]['ch1'], axis=0)[2::5]
    # Qdata = np.average(dataReceive["D1"]['ch1'], axis=0)[3::5]
    # from data_process import fit_all as fa
    #
    # fa.t1_fit(Idata, Qdata, timeArray/1e3)