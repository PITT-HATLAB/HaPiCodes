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
    W, Q = WQ.testPulse()
    WQ.generateDrivePulse()
    W()['piPulse_gau.x'].plotIdata('all Pulse', 'Gaussian')
    W()['piPulse_boxOnOff'].plotIdata('all Pulse', 'Box')
    W()['piPulse_box'].plotIdata('all Pulse', 'Tanh')
    W()['piPulse_hanning'].plotIdata('all Pulse', 'Hanning')

    W()['piPulse_gau.x'].fft('fft', 'Gaussian', log=1)
    W()['piPulse_boxOnOff'].fft('fft', 'Box', log=1)
    W()['piPulse_box'].fft('fft', 'Tanh', log=1)
    W()['piPulse_hanning'].fft('fft', 'Hanning', log=1)

