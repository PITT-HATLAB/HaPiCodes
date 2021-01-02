import numpy as np
import matplotlib.pyplot as plt
from AddOns.DigFPGAConfig_Demodulate import config_demodulator, config_weight_func, get_recommended_truncation

def plotSingleCycle(dataReceive, ch, cyc, plotName="preRun", weightAndTrig=0):
    data_read_all = np.array([dataReceive['1'], dataReceive['2'], dataReceive['3'], dataReceive['4']])
    demod_I = data_read_all[:, :, 0::5]
    demod_Q = data_read_all[:, :, 1::5]
    mag2 = data_read_all[:, :, 2::5]
    weight = data_read_all[:, :, 3::5]
    HVI_trig = data_read_all[:, :, 4::5] * 30000.
    xdata = np.arange(len(data_read_all[ch-1, cyc, 0::5])) * 10
    plt.figure(plotName)
    plt.plot(xdata, demod_I[ch-1, cyc], label="I")
    plt.plot(xdata, demod_Q[ch-1, cyc],label="Q")
    plt.plot(xdata, mag2[ch-1, cyc],label="Mag2")
    if weightAndTrig:
        plt.plot(xdata, weight[ch-1, cyc],label="weight")
        plt.plot(xdata, HVI_trig[ch-1, cyc],label="HVI trig")
    plt.legend()
    return (demod_I, demod_Q, mag2)

def plotPreRrun(dataReceive, sigRefCh, plotName="preRun", sumRange=[100, 500]):
    demod_sigI = dataReceive[str(sigRefCh[0])][:, 0::5]
    demod_sigQ = dataReceive[str(sigRefCh[0])][:, 1::5]
    demod_sigMag = dataReceive[str(sigRefCh[0])][:, 2::5]
    demod_refI = dataReceive[str(sigRefCh[1])][:, 0::5]
    demod_refQ = dataReceive[str(sigRefCh[1])][:, 1::5]
    demod_refMag = np.average(dataReceive[str(sigRefCh[1])][:, 2::5])
    
    demod_I = (demod_sigI * demod_refI + demod_sigQ * demod_refQ)/demod_refMag
    demod_Q = (-demod_sigI * demod_refQ + demod_sigQ * demod_refI)/demod_refMag

    Itrace = np.average(demod_I, axis=0)
    Qtrace = np.average(demod_Q, axis=0)

    xdata = np.arange(len(Itrace)) * 10
    plt.figure(plotName, figsize=(8, 8))
    plt.subplot(221)
    plt.plot(xdata, Itrace, label="I")
    plt.plot(xdata, Qtrace, label="Q")
    plt.legend()
    plt.subplot(222)
    plt.plot(xdata, np.sqrt(Itrace**2 + Qtrace**2), label="Mag1")
    plt.plot(xdata, np.average(demod_sigMag, axis=0), label='Mag2')
    plt.legend()
    plt.subplot(223)
    plt.plot(Itrace, Qtrace)
    plt.subplot(224)
    plt.hist2d(np.sum(demod_I[:, sumRange[0]//10:sumRange[1]//10], axis=1), np.sum(demod_Q[:, sumRange[0]//10:sumRange[1]//10], axis=1), bins=101)

    return (demod_I, demod_Q, demod_sigMag)

def plotReal(dataReceive, ch, cyc, plotName="real", xdata=None):
    data_read_all = np.array([dataReceive['1'], dataReceive['2'], dataReceive['3'], dataReceive['4']])
    demod_I = data_read_all[:, :, 0::5]
    demod_Q = data_read_all[:, :, 1::5]
    demod_rotI = data_read_all[:, :, 2::5]
    demod_rotQ = data_read_all[:, :, 3::5]
    if xdata is None:
        xdata = np.arange(len(data_read_all[ch, cyc, 0::5]))
    plt.figure(plotName)
    plt.subplot(121)
    plt.plot(xdata, demod_I[ch-1, cyc], label="I")
    plt.plot(xdata, demod_Q[ch-1, cyc],label="Q")
    plt.subplot(122)
    plt.plot(xdata, demod_rotI[ch-1, cyc],label="rotI")
    plt.plot(xdata, demod_rotQ[ch-1, cyc],label="rotQ")
    plt.legend()
    return (demod_rotI, demod_rotQ)