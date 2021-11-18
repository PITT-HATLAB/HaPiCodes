import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.simulation import basicSimulatePulse as bsp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel

yamlFile = msmtInfoSel.cwYaml
ampArray = np.linspace(-0.2, 0.2, 1000)

if __name__ == '__main__':

    ##
    #set ampArray size from a command line argument - used in benchmark.ipynb
    #call as python S001_piPulseTuneUp.py #ampArray
    import sys
    ampArray = np.linspace(-0.2, 0.2, int(sys.argv[1]))
    ##

    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    WQ = bsp.BasicExperiments(msmtInfoDict)
    W, Q = WQ.piPulseTuneUp(ampArray)
    WQ.generateDrivePulse()
    res = WQ.simulate()
    plt.figure()
    plt.plot(ampArray, res[:, 0])
    plt.xlabel('Amplitude (a.u.)', fontsize=12)
    plt.ylabel('<N_q>', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=1, color='r', linestyle='--', label = 'Excited')
    plt.axhline(y=0, color='b', linestyle='--', label = 'Ground')
    piPulseAmp = f.pi_pulse_tune_up(res[:, 0], np.zeros(len(res[:, 0])), xdata=ampArray, updatePiPusle_amp=1, plot=1)