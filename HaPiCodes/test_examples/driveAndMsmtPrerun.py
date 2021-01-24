import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.pulse import basicMsmtPulses as amp
from HaPiCodes.pathwave.pxi_instruments import PXI_Instruments
from HaPiCodes.data_process import fittingAndDataProcess as f
from HaPiCodes.test_examples import msmtInfoSel



def cavityResponse(yamlFile=msmtInfoSel.cwYaml, plot=1, qdrive=1):
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    msmtInfoDict['moduleConfig']['D1']['FPGA'] = 'Demodulate_showWeight'
    f.yamlFile = yamlFile
    pxi = PXI_Instruments(msmtInfoDict, reloadFPGA=True)
    WQ = amp.waveformAndQueue(pxi.module_dict, msmtInfoDict, subbuffer_used=pxi.subbuffer_used)
    W, Q = WQ.driveAndMsmt(qdrive=qdrive)
    pxi.autoConfigAllDAQ(W, Q) #PXIModules.autoConfigAllDAQ
    pxi.uploadPulseAndQueue()
    dataReceive = pxi.runExperiment(timeout=20000)
    pxi.releaseHviAndCloseModule()
    IQdata = f.processDataReceiveWithRef(pxi.subbuffer_used, dataReceive, plot=plot)
    return (W, Q, dataReceive, IQdata)

if __name__ == '__main__':

    msmt = cavityResponse(plot=1)

