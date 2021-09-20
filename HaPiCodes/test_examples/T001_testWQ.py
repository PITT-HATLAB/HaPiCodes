import os
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.pulse.waveformAndQueue import ExperimentSequence
from HaPiCodes.test_examples import msmtInfoSel
from HaPiCodes.simulation.constants import *
from HaPiCodes.simulation.qutipPac import *
from qutip.parallel import parallel_map
from HaPiCodes.simulation.qutipSim import SimulationExperiments

module_dict = { "A0": None,
                "A1": None,
                "A2": None,
                "A3": None,
                "A4": None,
                "A5": None,
                "M0": None,
                "M1": None,
                "M2": None,
                "M3": None,
                "D1": None,
                "D2": None
               }

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))

if __name__ == '__main__':
    ampArray = np.linspace(-1, 1, 51)
    test = SimulationExperiments(module_dict, msmtInfoDict, 1)
    W, Q = test.piPulseTuneUp(ampArray)
    test.generateDrivePulse()
    res = test.simulate()
    plt.figure()
    plt.plot(ampArray, res)