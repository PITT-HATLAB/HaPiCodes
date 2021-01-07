from typing import List, Callable, Union, Tuple, Dict
import warnings

from nptyping import NDArray
import numpy as np
import matplotlib.pyplot as plt

def plotOriginal(dataReceive, ch, cyc, plotName="Original", weightAndTrig=0):
    data_read_all = np.array([dataReceive['1'], dataReceive['2'], dataReceive['3'], dataReceive['4']])
    dataPlot = data_read_all[:, :, :]
    xdata = np.arange(len(data_read_all[ch-1, cyc, :])) * 2
    plt.figure(plotName)
    plt.plot(xdata, dataPlot[ch-1, cyc], '*-', label="I")

    return


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
    plt.plot(xdata, demod_Q[ch-1, cyc], label="Q")
    plt.plot(xdata, mag2[ch-1, cyc],label="Mag2")
    if weightAndTrig:
        plt.plot(xdata, weight[ch-1, cyc],label="weight")
        plt.plot(xdata, HVI_trig[ch-1, cyc],label="HVI trig")
    plt.legend()
    return (demod_I, demod_Q, mag2)

def plotPreRrun(dataReceive, sigRefCh, plotName="preRun", sumRange=[100, 500]):
    demod_sigI = dataReceive['ch' + str(sigRefCh[0])][:, 0, 0::5]
    demod_sigQ = dataReceive['ch' + str(sigRefCh[0])][:, 0, 1::5]
    demod_sigMag = dataReceive['ch' + str(sigRefCh[0])][:, 0, 2::5]
    demod_refI = dataReceive['ch' + str(sigRefCh[1])][:, 0, 0::5]
    demod_refQ = dataReceive['ch' + str(sigRefCh[1])][:, 0, 1::5]
    demod_refMag = np.average(dataReceive['ch' + str(sigRefCh[1])][:, 0, 2::5])

    demod_I = (demod_sigI * demod_refI + demod_sigQ * demod_refQ) / demod_refMag
    demod_Q = (-demod_sigI * demod_refQ + demod_sigQ * demod_refI) / demod_refMag

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

def get_recommended_truncation(data_I: NDArray[float], data_Q:NDArray[float],
                               integ_start: int, integ_stop: int, current_demod_trunc: int = 19,
                               fault_tolerance_factor:float = 1.01) -> Tuple[int, int]:
    """ get recommended truncation point for both demodulation and integration from the cavity response data trace.

    :param data_I: cavity response I data. 1 pt/10 ns. The shape should be (DAQ_cycles, points_per_cycle).
    :param data_Q: cavity response Q data. 1 pt/10 ns. The shape should be (DAQ_cycles, points_per_cycle).
    :param integ_start: integration start point, unit: ns
    :param integ_stop: integration stop point (not integrated), unit: ns
    :param current_demod_trunc: the demodulation truncation point used to get data_I and data_Q
    :param fault_tolerance_factor: a factor that will be multiplied onto the data to make sure overflow will not happen
    :return: demod_trunc_point, integ_trunc_point
    """
    data_I = data_I.astype(float)
    data_Q = data_Q.astype(float) # to avoid overflow in calculation
    max_mag = np.max(np.sqrt(data_I ** 2 + data_Q ** 2)) * fault_tolerance_factor
    bits_available = 15 - int(np.ceil(np.log2(max_mag+1)))
    if bits_available < 0 :
        warnings.warn("Overflow might happen, increasing digitizer fullScale is recommended")
    if current_demod_trunc - bits_available < 0:
        warnings.warn("Input data too small, decreasing digitizer fullScale is recommended")
    demod_trunc = np.clip(current_demod_trunc - bits_available, 0, 19) #TODO: this 19 should come from markup actually
    data_I_new = data_I * 2 ** bits_available
    data_Q_new = data_Q * 2 ** bits_available

    #TODO: validate integ_start and integ_stop
    integ_I = np.sum(data_I_new[:, integ_start // 10: integ_stop // 10], axis=1)
    integ_Q = np.sum(data_Q_new[:, integ_start // 10: integ_stop // 10], axis=1)
    max_integ = np.max(np.sqrt(integ_I ** 2 + integ_Q ** 2)) * fault_tolerance_factor
    integ_trunc = np.clip(int(np.ceil(np.log2(max_integ+1))) - 15, 0, 16) #TODO: this 16 should also come from markup
    return demod_trunc, integ_trunc
