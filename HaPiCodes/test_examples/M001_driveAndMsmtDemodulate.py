import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import basicMsmtPulses as bmp
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel



def cavityResponse(yamlFile=msmtInfoSel.cwYaml, plot=1, driveQubit=True, DigModules = ["D1", "D2"]):
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    for module in DigModules:
        try:
            msmtInfoDict['moduleConfig'][module]['FPGA'] = 'Demodulate_showWeight'
        except KeyError :
            pass
    f.yamlFile = yamlFile
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = bmp.BasicExperiments(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt(driveQubit=driveQubit)
    pxi.autoConfigAllDAQ(W, Q)
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    IQdata = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=plot)
    return (W, Q, dataReceive, IQdata, pxi)

if __name__ == '__main__':
    msmt = cavityResponse(plot=1)

