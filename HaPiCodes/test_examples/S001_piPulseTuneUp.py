import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.simulation import basicSimulatePulse as bsp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel

yamlFile = msmtInfoSel.cwYaml
ampArray = np.linspace(-0.9, 0.9, 100)

if __name__ == '__main__':

    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    WQ = bsp.BasicExperiments(msmtInfoDict)
    W, Q = WQ.piPulseTuneUp(ampArray)
    WQ.generateDrivePulse()
    res = WQ.simulate()
    plt.figure()
    plt.plot(ampArray, res)

    piPulseAmp = f.pi_pulse_tune_up(res, np.zeros(len(res)), xdata=ampArray, updatePiPusle_amp=1, plot=1)