import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.simulation import basicSimulatePulse as bsp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel

yamlFile = msmtInfoSel.cwYaml
timeArray = np.linspace(0.1, 5.1, 101) * 1e3

if __name__ == '__main__':

    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    f.yamlFile = yamlFile
    WQ = bsp.BasicExperiments(msmtInfoDict)
    W, Q = WQ.exchangeBetweenQ1AndC1(timeArray)
    WQ.generateDrivePulse()

    # res = WQ.simulateSingleIndex(100)
    # print(res)
    res = WQ.simulate()
    plt.figure()
    plt.plot(timeArray, res)
