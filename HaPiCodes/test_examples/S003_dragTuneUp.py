import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.simulation import basicSimulatePulse as bsp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel

yamlFile = msmtInfoSel.cwYaml
xdata = np.linspace(-5, 5, 51)
dragArray = xdata.repeat(2)

if __name__ == '__main__':

    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    WQ = bsp.BasicExperiments(msmtInfoDict)
    W, Q = WQ.dragTuneUp(dragArray)
    WQ.generateDrivePulse()
    res = WQ.simulate()
    plt.figure()
    plt.plot(xdata, res[::2, 0], label='ypx9')
    plt.plot(xdata, res[1::2, 0], label='xpy9')
    plt.xlabel('DRAG coefficient', fontsize=12)
    plt.ylabel('<N_q>', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
