import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import basicMsmtPulses as bmp
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments, getWeightFuncByName
from HaPiCodes.test_examples import msmtInfoSel

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))
f.yamlFile = msmtInfoSel.cwYaml

module_dict = { "A1": None,
                "A2": None,
                "A3": None,
                "A4": None,
                "A5": None,
                "M1": None,
                "M2": None,
                "M3": None,
                "D1": None,
                "D2": None
               }

if __name__ == '__main__':

    WQ = bmp.BasicExperiments(module_dict, msmtInfoDict, subbuffer_used=0)
    W, Q = WQ.driveAndMsmt(driveQubit=True)
