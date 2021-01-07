import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pulse import allMsmtPulses as amp
from data_process import fit_all as f
from pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from test_examples import msmtInfoSel
from sd1_api.SD1AddOns import get_FPGA_register

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))

if __name__ == '__main__':

    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.piPulseTuneUp()
    pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)

    d1 = pxi.module_dict["D1"].instrument
    r1 = get_FPGA_register(d1, "FPGARegisters_1_integ_trunc")
    print(r1.readRegisterInt32())

    pxi.releaseHviAndCloseModule()


    Idata = dataReceive['D1']['ch1'][:, 0::5]
    Qdata = dataReceive['D1']['ch1'][:, 1::5]
    I = np.average(Idata, axis=0)
    Q = np.average(Qdata, axis=0)
    f.pi_pulse_tune_up(I, Q, np.linspace(-0.5, 0.5, 101)[:100])