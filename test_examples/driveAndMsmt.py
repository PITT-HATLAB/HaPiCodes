import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from test_examples import msmtInfoSel
from sd1_api.SD1AddOns import get_FPGA_register

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt()
    pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)

    d1 = pxi.module_dict["D1"].instrument
    r1 = get_FPGA_register(d1, "FPGARegisters_1_integ_trunc")
    print(r1.readRegisterInt32())

    pxi.releaseHviAndCloseModule()
    #
    Idata = dataReceive['D1']['ch1'][:, 0::5].flatten()
    Qdata = dataReceive['D1']['ch1'][:, 1::5].flatten()
    plt.hist2d(Idata, Qdata, bins=101)
    plt.hist2d(dataReceive["D1"]['ch1'].flatten()[0::5], dataReceive["D1"]['ch1'].flatten()[1::5], bins=101)
    plt.colorbar()