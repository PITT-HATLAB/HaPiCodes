from typing import List, Callable, Union, Tuple, Dict
from typing_extensions import Literal
import warnings
import operator
from functools import reduce
from inspect import getfullargspec

import matplotlib.pyplot as plt
import h5py
import lmfit as lmf
import math
import yaml
import os
import numpy as np
from nptyping import NDArray
import h5py
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib.patches import Circle, Wedge, Polygon
from scipy.ndimage import gaussian_filter as gf
from scipy.special import factorial

from HaPiCodes.data_process.IQdata import IQData, getIQDataFromDataReceive

yamlFile = ''


def processDataReceiveWithRef(subbuffer_used, dataReceive, digName='Dig', plot=0, reSampleNum=1):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    sig_ch = yamlDict['combinedChannelUsage'][digName]['Sig']
    try:
        ref_ch = yamlDict['combinedChannelUsage'][digName]['Ref']
    except KeyError:
        return processDataReceive(subbuffer_used, dataReceive, digName, plot, reSampleNum)
    dataReceiveProcess = dataReceive

    if reSampleNum != 1:
        sig_data_raw = dataReceive[sig_ch[0]][f"ch{sig_ch[1]}"]
        ref_data_raw = dataReceive[ref_ch[0]][f"ch{ref_ch[1]}"]

        oldShape_s = sig_data_raw.shape
        oldShape_r = ref_data_raw.shape
        sig_data_temp = sig_data_raw.reshape((*oldShape_s[:-1], -1, reSampleNum, 5))
        ref_data_temp = ref_data_raw.reshape((*oldShape_r[:-1], -1, reSampleNum, 5))
        sig_data_resampled = np.mean(sig_data_temp, axis=-2).reshape((*oldShape_s[:-1], -1))
        ref_data_resampled = np.mean(ref_data_temp, axis=-2).reshape((*oldShape_r[:-1], -1))

        dataReceiveProcess = {}
        dataReceiveProcess[sig_ch[0]] = {}
        dataReceiveProcess[sig_ch[0]][f"ch{sig_ch[1]}"] = sig_data_resampled
        if ref_ch[0] in dataReceiveProcess.keys():
            dataReceiveProcess[sig_ch[0]][f"ch{ref_ch[1]}"] = ref_data_resampled
        else:
            dataReceiveProcess.update({ref_ch[0]: {f"ch{ref_ch[1]}": ref_data_resampled}})
        
    sig_data = getIQDataFromDataReceive(dataReceiveProcess, *sig_ch, subbuffer_used)
    ref_data = getIQDataFromDataReceive(dataReceiveProcess, *ref_ch, subbuffer_used)
    
    if subbuffer_used:
        Iarray = sig_data.I_rot
        Qarray = sig_data.Q_rot
        I = np.average(Iarray, axis=0)
        Q = np.average(Qarray, axis=0)
        if plot:
            plt.figure(figsize=(9, 4))
            plt.subplot(121)
            plt.plot(I)
            plt.plot(Q)
            plt.subplot(122)
            plt.hist2d(Iarray.flatten(), Qarray.flatten(), bins=101, range=yamlDict['histRange'])

    else:
        sumStart = yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['integ_start']
        sumEnd = yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['integ_stop']
        sig_data.integ_IQ_trace(sumStart, sumEnd, ref_data, timeUnit=reSampleNum * 10)
        Itrace = np.average(sig_data.I_trace_rot, axis=(0,1))
        Qtrace = np.average(sig_data.Q_trace_rot, axis=(0,1))
        Mag_trace = np.average(sig_data.Mag_trace, axis=(0,1))
        sig_data.time_trace = np.arange(len(Itrace)) * reSampleNum * 10
        if plot:
            xdata = np.arange(len(Itrace)) * reSampleNum * 10
            plt.figure(figsize=(8, 8))
            plt.subplot(221)
            plt.plot(xdata, Itrace, label="I")
            plt.plot(xdata, Qtrace, label="Q")
            plt.legend()
            plt.subplot(222)
            plt.plot(xdata, np.sqrt(Itrace ** 2 + Qtrace ** 2), label="Mag_py")
            plt.plot(xdata, Mag_trace, label='Mag2_fpga')
            plt.legend()
            plt.subplot(223)
            plt.plot(Itrace, Qtrace)
            plt.subplot(224)

            plt.hist2d(sig_data.I_rot.flatten(), sig_data.Q_rot.flatten(), bins=101)

        sigI_trace_flat = sig_data.I_trace_raw.reshape(-1, sig_data.I_trace_raw.shape[-1])
        sigQ_trace_flat = sig_data.Q_trace_raw.reshape(-1, sig_data.Q_trace_raw.shape[-1])
        refI_trace_flat = ref_data.I_trace_raw.reshape(-1, ref_data.I_trace_raw.shape[-1])
        refQ_trace_flat = ref_data.Q_trace_raw.reshape(-1, ref_data.Q_trace_raw.shape[-1])

        if reSampleNum == 1:
            sigTruncInfo = get_recommended_truncation(sigI_trace_flat, sigQ_trace_flat, sumStart, sumEnd, current_demod_trunc=yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['demod_trunc'])
            refTruncInfo = get_recommended_truncation(refI_trace_flat, refQ_trace_flat, sumStart, sumEnd, current_demod_trunc=yamlDict['FPGAConfig'][ref_ch[0]]['ch' + str(ref_ch[1])]['demod_trunc'])

            print('sig truncation: ', sigTruncInfo)
            print('ref truncation: ', refTruncInfo)

    return sig_data


def processDataReceive(subbuffer_used, dataReceive, digName='Dig', plot=0, reSampleNum=1):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    sig_ch = yamlDict['combinedChannelUsage'][digName]['Sig']
    if "Ref" in yamlDict['combinedChannelUsage'][digName]:
        warnings.warn("reference channel is in ymal file, but not used in data processing")
    dataReceiveProcess = dataReceive

    if reSampleNum != 1:
        sig_data_raw = dataReceive[sig_ch[0]][f"ch{sig_ch[1]}"]

        oldShape_s = sig_data_raw.shape
        sig_data_temp = sig_data_raw.reshape((*oldShape_s[:-1], -1, reSampleNum, 5))
        sig_data_resampled = np.mean(sig_data_temp, axis=-2).reshape((*oldShape_s[:-1], -1))
        dataReceiveProcess = {}
        dataReceiveProcess[sig_ch[0]] = {}
        dataReceiveProcess[sig_ch[0]][f"ch{sig_ch[1]}"] = sig_data_resampled

    sig_data = getIQDataFromDataReceive(dataReceiveProcess, *sig_ch, subbuffer_used)

    if subbuffer_used:
        Iarray = sig_data.I_raw
        Qarray = sig_data.Q_raw
        I = np.average(Iarray, axis=0)
        Q = np.average(Qarray, axis=0)
        if plot:
            plt.figure(figsize=(9, 4))
            plt.subplot(121)
            plt.plot(I)
            plt.plot(Q)
            plt.subplot(122)
            plt.hist2d(Iarray.flatten(), Qarray.flatten(), bins=101, range=yamlDict['histRange'])
    else:
        sumStart = yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['integ_start']
        sumEnd = yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['integ_stop']

        sig_data.integ_IQ_trace(sumStart, sumEnd, None, timeUnit=reSampleNum * 10)
        Itrace = np.average(sig_data.I_trace_raw, axis=(0,1))
        Qtrace = np.average(sig_data.Q_trace_raw, axis=(0,1))
        Mag_trace = np.average(sig_data.Mag_trace, axis=(0,1))
        sig_data.time_trace = np.arange(len(Itrace)) * reSampleNum * 10
        if plot:
            xdata = np.arange(len(Itrace)) * reSampleNum * 10
            plt.figure(figsize=(8, 8))
            plt.subplot(221)
            plt.plot(xdata, Itrace, label="I")
            plt.plot(xdata, Qtrace, label="Q")
            plt.legend()
            plt.subplot(222)
            plt.plot(xdata, np.sqrt(Itrace ** 2 + Qtrace ** 2), label="Mag_py")
            plt.plot(xdata, Mag_trace, label='Mag2_fpga')
            plt.legend()
            plt.subplot(223)
            plt.plot(Itrace, Qtrace)
            plt.subplot(224)
            plt.hist2d(sig_data.I_raw.flatten(), sig_data.Q_raw.flatten(), bins=101)

        sigI_trace_flat = sig_data.I_trace_raw.reshape(-1, sig_data.I_trace_raw.shape[-1])
        sigQ_trace_flat = sig_data.Q_trace_raw.reshape(-1, sig_data.Q_trace_raw.shape[-1])

        if reSampleNum == 1:
            sigTruncInfo = get_recommended_truncation(sigI_trace_flat, sigQ_trace_flat, sumStart, sumEnd, current_demod_trunc=yamlDict['FPGAConfig'][sig_ch[0]]['ch' + str(sig_ch[1])]['demod_trunc'])
            print('sig truncation: ', sigTruncInfo)

    sig_data.I_rot = sig_data.I_raw
    sig_data.Q_rot = sig_data.Q_raw
    sig_data.I_trace_rot = sig_data.I_trace_raw
    sig_data.Q_trace_rot = sig_data.Q_trace_raw

    return sig_data



def average_data(data_I, data_Q, axis0_type:Literal["nAvg", "xData"] = "nAvg"):
    if axis0_type == "nAvg":
        I_avg = np.average(data_I, axis=0)
        Q_avg = np.average(data_Q, axis=0)
    elif axis0_type == "xData":
        I_avg = []
        Q_avg = []
        for i in range(len(data_I)):
            I_avg.append(np.average(data_I[i]))
            Q_avg.append(np.average(data_Q[i]))
        I_avg = np.array(I_avg)
        Q_avg = np.array(Q_avg)
    else:
        raise NameError("invalid axis0_type name, can only be 'nAvg' or 'xData' ")
    return  I_avg, Q_avg




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



# =================== Histogram Fitting=================================================================================
def twoD_Gaussian(rawTuple, amp, x0, y0, sigmaX, sigmaY, theta, offset):
    (x, y) = rawTuple
    xo = float(x0)
    yo = float(y0)
    a = (np.cos(theta)**2)/(2*sigmaX**2) + (np.sin(theta)**2)/(2*sigmaY**2)
    b = -(np.sin(2*theta))/(4*sigmaX**2) + (np.sin(2*theta))/(4*sigmaY**2)
    c = (np.sin(theta)**2)/(2*sigmaX**2) + (np.cos(theta)**2)/(2*sigmaY**2)
    g = offset + amp*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def two_blob(rawTuple, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
                       amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2):
    (x, y) = rawTuple
    return twoD_Gaussian((x,y), amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) \
            + twoD_Gaussian((x,y), amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2)

def three_blob(rawTuple, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
               amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2,
               amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3):
    (x, y) = rawTuple
    return twoD_Gaussian((x,y), amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) \
            + twoD_Gaussian((x,y), amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2) \
            + twoD_Gaussian((x,y), amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3)


def fit1_2DGaussian(x_, y_, z_, plot=1, mute=0, fitGuess: dict = None):
    fitGuess = {} if fitGuess is None else fitGuess
    gau_args = getfullargspec(twoD_Gaussian).args[1:]
    for k_ in fitGuess.keys():
        if k_ not in gau_args:
            raise NameError(f"fitGuess got unexpected key '{k_}'. Available keys are {gau_args}")

    p0_ = [0, 0, 0, 500, 500, 0, 0]
    z_ = gf(z_, [2, 2])
    xd, yd = np.meshgrid(x_[:-1], y_[:-1])

    max1xy = np.where(z_==np.max(z_))
    x1indx = int(max1xy[0])
    y1indx = int(max1xy[1])
    x1ini = x_[x1indx]
    y1ini = y_[y1indx]
    amp1 = np.max(z_)

    p0_[0] = amp1
    p0_[1] = x1ini
    p0_[2] = y1ini

    guess_params = dict(zip(gau_args, p0_))
    guess_params.update(fitGuess)
    popt, pcov = curve_fit(twoD_Gaussian, (xd, yd), z_.ravel(), p0=list(guess_params.values()),
                          bounds=[[0, -30000, -30000, 0, 0, -np.pi, -10], [np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10]], maxfev=int(1e6))
    data_fitted = twoD_Gaussian((xd, yd), *popt)

    x1, y1, sigma1x, sigma1y = popt[1:5]
    if not mute:
        print('max count', (popt[0]))
        print('gaussian1 xy', (popt[1:3]))
        print('sigma1 xy', (popt[3:5]))
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x_, y_, z_)
        ax.contour(xd, yd, data_fitted.reshape(101, 101), 3, colors='w')
    return (x1, y1, sigma1x, sigma1y, popt[0])

def fit2_2DGaussian(x_, y_, z_, plot=1, mute=0, fitGuess: dict = None):
    fitGuess = {} if fitGuess is None else fitGuess
    gau_args = getfullargspec(two_blob).args[1:]
    for k_ in fitGuess.keys():
        if k_ not in gau_args:
            raise NameError(f"fitGuess got unexpected key '{k_}'. Available keys are {gau_args}")

    p0_ = [0, 0, 0, 500, 500, 0, 0,
           0, 0, 0, 500, 500, 0, 0]
    z_ = gf(z_, [2, 2])
    xd, yd = np.meshgrid(x_[:-1], y_[:-1])

    max1xy = np.where(z_==np.max(z_))
    x1indx = int(max1xy[1])
    y1indx = int(max1xy[0])
    x1ini = x_[x1indx]
    y1ini = y_[y1indx]
    amp1 = np.max(z_)
    maskIndex = 15
    # print(max1xy)
    # print(x1ini, y1ini, maskIndex)
    mask1 = np.zeros((len(x_)-1, len(y_)-1))
    mask1[-maskIndex+y1indx:maskIndex+y1indx, -maskIndex+x1indx:maskIndex+x1indx] = 1
    z2_ = np.ma.masked_array(z_, mask=mask1)
    max2xy = np.where(z2_==np.max(z2_))
    x2indx = int(max2xy[1])
    y2indx = int(max2xy[0])
    x2ini = x_[x2indx]
    y2ini = y_[y2indx]
    amp2 = np.max(z2_)

    p0_[0] = amp1
    p0_[1] = x1ini
    p0_[2] = y1ini
    p0_[7] = amp2
    p0_[8] = x2ini
    p0_[9] = y2ini

    guess_params = dict(zip(gau_args, p0_))
    guess_params.update(fitGuess)
    popt, pcov = curve_fit(two_blob, (xd, yd), z_.ravel(), p0=list(guess_params.values()),
                           bounds=[[0, -30000, -30000, 0, 0, -np.pi, -10,
                                    0, -30000, -30000, 0, 0, -np.pi, -10],
                                   [np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10,
                                    np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10]], maxfev=int(1e5))

    data_fitted = two_blob((xd, yd), *popt)
    amp1, x1, y1, sigma1x, sigma1y = popt[0:5]
    amp2, x2, y2, sigma2x, sigma2y = popt[7:12]
    sigma1 = np.sqrt(sigma1x**2 + sigma1y**2)
    sigma2 = np.sqrt(sigma2x**2 + sigma2y**2)
    sigma = np.mean([sigma1, sigma2])

    sigma1Std = np.std([sigma1x, sigma1y])
    sigma2Std = np.std([sigma2x, sigma2y])
    if amp1 < amp2:
        [x1, y1, amp1, sigma1x, sigma1y, x2, y2, amp2, sigma2x, sigma2y] = [x2, y2, amp2, sigma2x, sigma2y, x1, y1, amp1, sigma1x, sigma1y]
    if not mute:
        print('max count', amp1, amp2)
        print('gaussian1 xy', x1, y1)
        print('gaussian2 xy', x2, y2)
        print('sigma1 xy', sigma1x, sigma1y)
        print('sigma2 xy', sigma2x, sigma2y)
        print('Im/sigma', np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/sigma)
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x_, y_, z_)
        ax.set_aspect(1)
        ax.contour(xd, yd, data_fitted.reshape(int(np.sqrt(len(data_fitted))), int(np.sqrt(len(data_fitted)))), 3, colors='w')
        ax.scatter([x1, x2], [y1, y2], c="r", s=0.7)
        ax.annotate("g", (x1, y1))
        ax.annotate("e", (x2, y2))



    return (x1, y1, x2, y2, sigma1x, sigma1y, sigma2x, sigma2y, amp1, amp2, np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/sigma)

def fit3_2DGaussian(x_, y_, z_, plot=1, mute=0, fitGuess: dict = None):
    fitGuess = {} if fitGuess is None else fitGuess
    gau_args = getfullargspec(three_blob).args[1:]
    for k_ in fitGuess.keys():
        if k_ not in gau_args:
            raise NameError(f"fitGuess got unexpected key '{k_}'. Available keys are {gau_args}")
        
    p0_ = [0, 0, 0, 500, 500, 0, 0,
           0, 0, 0, 500, 500, 0, 0,
           0, 0, 0, 500, 500, 0, 0]
    z_ = gf(z_, [2, 2])
    xd, yd = np.meshgrid(x_[:-1], y_[:-1])

    max1xy = np.where(z_==np.max(z_))
    x1indx = int(max1xy[1])
    y1indx = int(max1xy[0])
    x1ini = x_[x1indx]
    y1ini = y_[y1indx]
    amp1 = np.max(z_)
    maskIndex = 20
    # print(max1xy)
    # print(x1ini, y1ini, maskIndex)
    mask1 = np.zeros((len(x_)-1, len(y_)-1))
    mask1[-maskIndex+y1indx:maskIndex+y1indx, -maskIndex+x1indx:maskIndex+x1indx] = 1
    z2_ = np.ma.masked_array(z_, mask=mask1)
    max2xy = np.where(z2_==np.max(z2_))
    x2indx = int(max2xy[1])
    y2indx = int(max2xy[0])
    x2ini = x_[x2indx]
    y2ini = y_[y2indx]
    amp2 = np.max(z2_)

    mask2 = mask1
    mask2[-maskIndex+y2indx:maskIndex+y2indx, -maskIndex+x2indx:maskIndex+x2indx] = 1
    z3_ = np.ma.masked_array(z_, mask=mask2)
    max3xy = np.where(z3_==np.max(z3_))
    x3indx = int(max3xy[1])
    y3indx = int(max3xy[0])
    x3ini = x_[x3indx]
    y3ini = y_[y3indx]
    amp3 = np.max(z3_)


    p0_[0] = amp1
    p0_[1] = x1ini
    p0_[2] = y1ini
    p0_[7] = amp2
    p0_[8] = x2ini
    p0_[9] = y2ini
    p0_[14] = amp3
    p0_[15] = x3ini
    p0_[16] = y3ini

    guess_params = dict(zip(gau_args, p0_))
    guess_params.update(fitGuess)
    popt, pcov = curve_fit(three_blob, (xd, yd), z_.ravel(), p0=list(guess_params.values()),
                           bounds=[[0, -30000, -30000, 0, 0, -np.pi, -10,
                                    0, -30000, -30000, 0, 0, -np.pi, -10,
                                    0, -30000, -30000, 0, 0, -np.pi, -10],
                                   [np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10,
                                    np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10,
                                    np.sum(z_), 30000, 30000, 5000, 5000, np.pi, 10]], maxfev=int(1e5))

    data_fitted = three_blob((xd, yd), *popt)

    amp1, x1, y1, sigma1x, sigma1y = popt[0:5]
    amp2, x2, y2, sigma2x, sigma2y = popt[7:12]
    amp3, x3, y3, sigma3x, sigma3y = popt[14:19]
    sigma1 = np.sqrt(sigma1x**2 + sigma1y**2)
    sigma2 = np.sqrt(sigma2x**2 + sigma2y**2)
    sigma3 = np.sqrt(sigma3x**2 + sigma3y**2)
    sigma = np.mean([sigma1, sigma2, sigma3])

    sigma1Std = np.std([sigma1x, sigma1y])
    sigma2Std = np.std([sigma2x, sigma2y])
    sigma3Std = np.std([sigma3x, sigma3y])

    gIndex = np.argmax(np.array([amp1, amp2, amp3]))
    eIndex = np.argmax(np.ma.masked_values(np.array([amp1, amp2, amp3]), np.array([amp1, amp2, amp3])[gIndex]))
    fIndex = np.argmin(np.array([amp1, amp2, amp3]))
    gef_order = [gIndex, eIndex, fIndex]
    # if y1 < y2:
    #     [x1, y1, amp1, sigma1x, sigma1y, x2, y2, amp2, sigma2x, sigma2y] = [x2, y2, amp2, sigma2x, sigma2y, x1, y1, amp1, sigma1x, sigma1y]
    gef_xy = np.array([[x1, y1], [x2, y2], [x3, y3]])[gef_order]
    gef_sigma = np.array([[sigma1x, sigma1y], [sigma2x, sigma2y], [sigma3x, sigma3y]])[gef_order]
    gef_amp = np.array([amp1, amp2, amp3])[gef_order]
    if not mute:
        print('max count', gef_amp)
        print('gaussian1 xy', gef_xy[0])
        print('gaussian2 xy', gef_xy[1])
        print('gaussian3 xy', gef_xy[2])
        print('sigma1 xy', gef_sigma[0])
        print('sigma2 xy', gef_sigma[1])
        print('sigma3 xy', gef_sigma[2])
        # print('Im/sigma', np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/sigma)
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x_, y_, z_)
        ax.set_aspect(1)
        ax.contour(xd, yd, data_fitted.reshape(101, 101), 3, colors='w')
        ax.scatter(*gef_xy.transpose(), c="r", s=0.7)
        for i, txt in enumerate(["g","e","f"]):
            ax.annotate(txt, (gef_xy[i][0], gef_xy[i][1]))

    # return [popt[gIndex * 7: gIndex * 7 + 7], popt[eIndex * 7: eIndex * 7 + 7], popt[fIndex * 7: fIndex * 7 + 7]]
    return np.concatenate((gef_xy.flatten(), gef_sigma.flatten(), gef_amp.flatten()))

def fit_Gaussian(data, blob=2, plot=1, mute=0, fitGuess=None, histRange=None):
    if histRange is not None:
        z_, x_, y_ = np.histogram2d(data[0], data[1], bins=201, range=np.array(histRange))
    else:
        z_, x_, y_ = np.histogram2d(data[0], data[1], bins=101)
    z_ = z_.T
    if blob == 1:
        fitRes = fit1_2DGaussian(x_, y_, z_, plot=plot, mute=mute, fitGuess=fitGuess)
    elif blob == 2:
        fitRes = fit2_2DGaussian(x_, y_, z_, plot=plot, mute=mute, fitGuess=fitGuess)
    elif blob == 3:
        fitRes = fit3_2DGaussian(x_, y_, z_, plot=plot, mute=mute, fitGuess=fitGuess)
    '''
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    ################# Not sure if we're going to use ###################################
    twoBlobFitInfo = yamlDict["fitGaussian"]['twoBlobFit']
    s1x = np.abs(fitRes[4]) < 500 #- twoBlobFitInfo[4]) > twoBlobFitInfo[4] / 2 # sigma1x
    s1y = np.abs(fitRes[5]) < 500 # - twoBlobFitInfo[5]) > twoBlobFitInfo[5] / 2 # sigma1y
    s2x = np.abs(fitRes[6]) < 500 #- twoBlobFitInfo[6]) > twoBlobFitInfo[6] / 2 # sigma2x
    s2y = np.abs(fitRes[7]) < 500 # - twoBlobFitInfo[7]) > twoBlobFitInfo[7] / 2 # sigma2y
    criteria = s1x or s1y or s2x or s2y
    if criteria:
        fitRes = yamlDict["fitGaussian"]['twoBlobFit']
    else:
        yamlDict["fitGaussian"]['twoBlobFit'] = np.array(fitRes).tolist()
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(yamlDict, file, sort_keys=0, default_flow_style=None)
    #######################################################################################
    '''

    return fitRes
#=======================================================================================================================

def post_sel(data_I, data_Q, g_x, g_y, g_r, msmt_per_sel:int = 2, plot_check=0):
    """
    This function always assume the 0::msmt_per_sel data are for selection
    :return : a list that contains the selected data. Each element of the list is an array of indefinite length, which
        contains the valid selected measurement results at the corresponding xdata point.
    """
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    n_avg = len(data_I)
    pts_per_exp = len(data_I[0])
    sel_idxs = np.arange(pts_per_exp)[0::msmt_per_sel]

    I_sel = data_I[:, sel_idxs]
    Q_sel = data_Q[:, sel_idxs]
    I_exp = np.zeros((n_avg, len(sel_idxs), msmt_per_sel-1))
    Q_exp = np.zeros((n_avg, len(sel_idxs), msmt_per_sel-1))
    for i in range(n_avg):
        for j in range(len(sel_idxs)):
            I_exp[i][j] = data_I[i, j*msmt_per_sel+1: (j+1)*msmt_per_sel]
            Q_exp[i][j] = data_Q[i, j*msmt_per_sel+1: (j+1)*msmt_per_sel]

    mask = (I_sel - g_x) ** 2 + (Q_sel - g_y) ** 2 < g_r ** 2
    I_vld = []
    Q_vld = []
    for i in range(len(sel_idxs)):
        for j in range(msmt_per_sel-1):
            I_vld.append(I_exp[:, i, j][mask[:, i]])
            Q_vld.append(Q_exp[:, i, j][mask[:, i]])

    if plot_check:
    # Plot -----------------
        plt.figure(figsize=(7, 7))
        plt.title('g state selection range')
        plt.hist2d(I_sel.flatten(), Q_sel.flatten(), bins=101, range=yamlDict['histRange'])
        theta = np.linspace(0,2*np.pi,201)
        plt.plot(g_x+g_r*np.cos(theta),g_y+g_r*np.sin(theta),color='r')

        plt.figure(figsize=(7, 7))
        plt.title('experiment pts after selection')
        plt.hist2d(np.hstack(I_vld), np.hstack(Q_vld), bins=101, range=yamlDict['histRange'])

    return I_vld, Q_vld


def post_sel_byLine(data_I, data_Q, g_x, g_y, e_x, e_y, bias_factor = 1, msmt_per_sel:int = 2, plot_check=0):
    """
    This function always assume the 0::msmt_per_sel data are for selection
    :param bias_factor: distance from the middle point between g and e to the split line, in unit of half of the
        distance between g and e. positive direction is e->g
    :return : a list that contains the selected data. Each element of the list is an array of indefinite length, which
        contains the valid selected measurement results at the corresponding xdata point.
    """
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    n_avg = len(data_I)
    pts_per_exp = len(data_I[0])
    sel_idxs = np.arange(pts_per_exp)[0::msmt_per_sel]

    I_sel = data_I[:, sel_idxs]
    Q_sel = data_Q[:, sel_idxs]
    I_exp = np.zeros((n_avg, len(sel_idxs), msmt_per_sel-1))
    Q_exp = np.zeros((n_avg, len(sel_idxs), msmt_per_sel-1))
    for i in range(n_avg):
        for j in range(len(sel_idxs)):
            I_exp[i][j] = data_I[i, j*msmt_per_sel+1: (j+1)*msmt_per_sel]
            Q_exp[i][j] = data_Q[i, j*msmt_per_sel+1: (j+1)*msmt_per_sel]


    def split_line(x):
        center_x = (g_x + e_x) / 2
        center_y = (g_y + e_y) / 2
        k_ = -(g_x - e_x) / (g_y - e_y)
        x1 = center_x + bias_factor * 0.5 * (g_x - e_x)
        y1 = center_y + bias_factor * 0.5 * (g_y - e_y)
        return k_ * (x - x1) + y1

    if g_y < e_y:
        mask = Q_sel < split_line(I_sel)
    else:
        mask = Q_sel > split_line(I_sel)

    I_vld = []
    Q_vld = []
    for i in range(len(sel_idxs)):
        for j in range(msmt_per_sel-1):
            I_vld.append(I_exp[:, i, j][mask[:, i]])
            Q_vld.append(Q_exp[:, i, j][mask[:, i]])

    if plot_check:
    # Plot -----------------
        plt.figure(figsize=(7, 7))
        plt.title('g state selection range')
        h, xedges, yedges, image = plt.hist2d(I_sel.flatten(), Q_sel.flatten(), bins=101, range=yamlDict['histRange'])
        plt.plot(xedges,split_line(xedges),color='r')

        plt.figure(figsize=(7, 7))
        plt.title('experiment pts after selection')
        plt.hist2d(np.hstack(I_vld), np.hstack(Q_vld), bins=101, range=yamlDict['histRange'])

    return I_vld, Q_vld





def cal_g_pct(data_, g_x, g_y, e_x, e_y, plot=1):
    n_pts = float(len(data_[0]))
    def split_line(x):
        center_x = (g_x + e_x) / 2
        center_y = (g_y + e_y) / 2
        k_ = -(g_x - e_x) / (g_y - e_y)
        return k_ * (x - center_x) + center_y

    if g_y < split_line(g_x):
        g_linemask = data_[1] < split_line(data_[0])
    else:
        g_linemask = data_[1] > split_line(data_[0])

    g_data_x = data_[0][g_linemask]
    g_data_y = data_[1][g_linemask]

    g_percent = len(g_data_x) / n_pts
    # print(len(g_data_x) , n_pts)
    # print("g percentage: " + str(g_percent))

    if plot:
        plt.figure(figsize=(7, 7))
        plt.title('split line')
        h, xedges, yedges, image = plt.hist2d(data_[0], data_[1], bins=101)
        plt.plot(xedges, split_line(xedges), color='w')

        plt.figure(figsize=(7, 7))
        plt.title("selected g by line")
        plt.hist2d(g_data_x, g_data_y, bins=101)

    # return g_linemask, g_percent
    return g_percent






#====================================Fitting Related========================================================
def get_rot_info():
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    angle = yamlDict['fitParams']['angle']
    excitedDigV = yamlDict['fitParams']['excitedDigV']
    groundDigV = yamlDict['fitParams']['groundDigV']
    return angle, excitedDigV, groundDigV


def store_rot_info(angle, excited, ground, piPulse_amp):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    yamlDict['fitParams']['angle'] = float(np.round(angle, 4))
    yamlDict['fitParams']['excitedDigV'] = float(np.round(excited, 2))
    yamlDict['fitParams']['groundDigV'] = float(np.round(ground, 2))
    yamlDict['fitParams']['piPulse_amp'] = float(np.round(piPulse_amp, 4))
    with open(yamlFile, 'w') as file:
        yaml.safe_dump(yamlDict, file, sort_keys=0, default_flow_style=None)
    print('info successly stored')
    return


def rotate_complex(real_part, imag_part, angle):
    """
    rotate the complex number as rad units.
    """
    iq_new = (real_part + 1j * imag_part) * np.exp(1j * np.pi * angle)
    return iq_new


def get_rot_data(i_data, q_data, xdata):
    angle, excited_b, ground_b = get_rot_info()
    iq_new = rotate_complex(i_data, q_data, angle)
    return iq_new.real


def determine_ge_states(xdata, ydata):
    mid = ydata[int(len(ydata) / 2)]
    excited = round(ydata.max(), 2)
    ground = round(ydata.min(), 2)
    if np.abs(excited - mid) < np.abs(ground - mid):
        excited, ground = ground, excited
    print('The Excited State DAC is', excited)
    print('The Ground State DAC is', ground)
    return excited, ground


def hline():
    agnle, excited, ground = get_rot_info()
    plt.axhline(y=excited, color='r', linestyle='--', label = 'Excited')
    plt.axhline(y=ground, color='b', linestyle='--', label = 'Ground')
    plt.axhline(y=(excited + ground) / 2.0, color='y', linestyle='--')
    plt.legend()


def _residuals(params, model, xdata, ydata):
    model_data = model(params, xdata)
    return model_data - ydata


def cos_model(params, xdata):
    value = params.valuesdict()
    amp = value['amp']
    offset = value['offset']
    freq = value['freq']
    phase = value['phase']
    ydata = amp * np.cos(2 * np.pi * freq * xdata + phase) + offset
    return ydata.view(np.float)



def cos_fit(xdata, ydata, plot=True, plotName=None, mute=True, **fixParams):
    # print(np.max(ydata), np.min(ydata))
    offset = (np.max(ydata) + np.min(ydata)) / 2.0
    amp = np.abs(np.max(ydata) - np.min(ydata)) / 2.0
    fourier_transform = np.fft.fft(ydata)
    max_point = np.argmax(np.abs(fourier_transform[1: len(fourier_transform) // 2]))
    normVec = np.abs(fourier_transform[1: len(fourier_transform) // 2])/np.linalg.norm(np.abs(fourier_transform[1: len(fourier_transform) // 2]))
    time_spacing = xdata[1] - xdata[0]
    f_array = np.fft.fftfreq(len(fourier_transform), d=time_spacing)[1:len(fourier_transform) // 2]
    order = np.sort(normVec)[::-1]
    firstValIndex = np.where(normVec==order[0])
    secondValIndex = np.where(normVec==order[1])
    freq = (f_array[firstValIndex] * normVec[firstValIndex])[0]# + f_array[secondValIndex] * normVec[secondValIndex])[0]
    period = 1. / freq
    phase = np.angle(fourier_transform[max_point + 1]) + np.pi
    if not mute:
        print(amp, offset, freq, phase)


    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp, min=amp * 0.8, max=amp * 1.2, vary=True)
    fit_params.add('offset', value=offset, min=offset*0.8, max=offset*1.2, vary=True)
    fit_params.add('phase', value=phase, min=-np.pi, max=np.pi, vary=True)
    fit_params.add('freq', value=freq, min=0, vary=True)
    for fixName, value in fixParams.items():
        fit_params[fixName] = lmf.Parameter(name=fixName, value=value, vary=False)

    out = lmf.minimize(_residuals, fit_params, method='powell', args=(cos_model, xdata, ydata))
    if plot:
        if plotName is not None:
            plt.figure(plotName)
        else:
            plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        plt.plot(xdata, cos_model(out.params, xdata), '-', label='fit period/2:' + str(np.round(1.0 / out.params['freq'] / 2.0, 5)) + ' unit')
        plt.legend()
    return out


def exponetialDecay_model(params, xdata):
    value = params.valuesdict()
    amp = value['amp']
    offset = value['offset']
    t1Fit = value['t1Fit']
    ydata = (np.exp(-xdata / t1Fit)) * amp + offset
    return ydata.view(np.float)


def exponetialDecay_fit(xdata, ydata, CI=False, plot=True):
    offset_ = (ydata[-1])
    amp_ = (ydata[0]) - (ydata[-1])
    t1Fit_ = (1.0 / 3.0) * (xdata[-1] - xdata[0])
    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp_, vary=True)
    fit_params.add('offset', value=offset_, vary=True)
    fit_params.add('t1Fit', value=t1Fit_, min=0, vary=True)
    mini = lmf.Minimizer(_residuals, fit_params, fcn_kws={'model':exponetialDecay_model, 'xdata': xdata, 'ydata': ydata})
    out = mini.leastsq()
    # out = lmf.minimize(_residuals, fit_params, method='powell', args=(exponetialDecay_model, xdata, ydata), nan_policy='omit')
    if plot:
        plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        plt.plot(xdata, exponetialDecay_model(out.params, xdata), '-', label=r"fit $\tau$: " + str(np.round(out.params['t1Fit'].value, 3)) + ' unit')
        plt.legend()
    if CI is True:
        ci, tr = lmf.conf_interval(mini, out, trace=True)
        # lmf.report_ci(ci)
        return out, ci
    return out


def exponetialDecayWithCos_model(params, xdata):
    value = params.valuesdict()
    amp = value['amp']
    offset = value['offset']
    t2Fit = value['t2Fit']
    freq = value['freq']
    phase = value['phase']
    ydata = amp * np.cos(freq * np.pi * 2 * xdata + phase) * np.exp(-xdata / t2Fit) + offset
    return ydata.view(np.float)


def exponetialDecayWithCos_fit(xdata, ydata, plot=True, legend=True, freqGuess=None, **paramDict):
    # amp = (np.max(ydata) - np.min(ydata)) / 2.0
    amp = ydata[0]-ydata[-1]
    t2Fit = (1 / 4.0) * (xdata[-1] - xdata[0])
    offset = ydata[-1]
    fourier_transform = np.fft.fft(ydata)
    max_point = np.argmax(np.abs(fourier_transform[1: len(fourier_transform) // 2]))
    time_spacing = xdata[1] - xdata[0]
    f_array = np.fft.fftfreq(len(fourier_transform), d=time_spacing)
    if freqGuess is None:
        freq = f_array[max_point]
    else:
        freq = freqGuess
    phase = np.arctan2(np.imag(fourier_transform[max_point]), np.real(fourier_transform[max_point]))
    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp, vary=True)
    fit_params.add('offset', value=offset, vary=True)
    fit_params.add('t2Fit', value=t2Fit, min=0, vary=True)
    fit_params.add('phase', value=phase, min=-2 * np.pi, max=2 * np.pi, vary=True)
    fit_params.add('freq', value=freq, min=0, max=0.5/time_spacing, vary=True)
    for k, v in paramDict.items():
        fit_params[k] = v
    out = lmf.minimize(_residuals, fit_params, method='powell', args=(exponetialDecayWithCos_model, xdata, ydata))
    if plot:
        plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        fit_x = np.linspace(np.min(xdata), np.max(xdata), 1001)
        plt.plot(fit_x, exponetialDecayWithCos_model(out.params, fit_x), '-', label='fit T2: ' + str(np.round(out.params['t2Fit'].value, 3)) + ' unit')
        if legend:
            plt.legend()
        print('freq is: ',out.params['freq'].value)
    return out


def linear_model(params, xdata):
    value = params.valuesdict()
    k = value['k']
    b = value['b']
    ydata = k * xdata + b
    return ydata.view(np.float)

def linear_fit(xdata, ydata, plot=True, **kwargs):
    npts = len(xdata)
    x1, y1 = np.average(xdata[:npts//2]), np.average(ydata[:npts//2])
    x2, y2 = np.average(xdata[npts//2:]), np.average(ydata[npts//2:])

    k = (y1-y2) / (x1 - x2)
    b = y1 - k * x1

    fit_params = lmf.Parameters()
    fit_params.add('k', value=k, vary=True)
    fit_params.add('b', value=b, vary=True)
    out = lmf.minimize(_residuals, fit_params, args=(linear_model, xdata, ydata))
    kfit = np.round(out.params['k'].value, 6)
    bfit = np.round(out.params['b'].value, 6)
    if plot:
        plt.figure(kwargs.get('plotName', 'linearFit'))
        plt.plot(xdata, ydata, '*', label='data')

        plt.plot(xdata, linear_model(out.params, xdata), '-', label= f"fit: {kfit} x + {bfit} ")
        plt.legend()
        print(f'k is: {kfit}, b is: {bfit}')
    return out, kfit, bfit
############################## For specific fitting object ################################################

def findBestAngle(i_data, q_data):
    deriv = []
    for i in range(2001):
        angle = 0.001 * i
        iq_temp = rotate_complex(i_data, q_data, angle)
        yvalue = iq_temp.imag
        line_fit = np.zeros(len(yvalue)) + yvalue.mean()
        deriv_temp = ((yvalue - line_fit) ** 2).sum()
        deriv.append(deriv_temp)
    final = 0.001 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
    rotation_angle = final.ravel()[0]
    return rotation_angle

def pi_pulse_tune_up(i_data, q_data, xdata=None, updatePiPusle_amp=0, plot=1):
    """
    fitting pi_pulse_tune_up as a sin function
    """
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata is None:
        piPulseAmpInfo = yamlDict['regularMsmtPulseInfo']['piPulseTuneUpAmp']
        xdata = np.linspace(piPulseAmpInfo[0], piPulseAmpInfo[1], piPulseAmpInfo[2] + 1)[:100]
    deriv = []
    for i in range(2001):
        angle = 0.001 * i
        iq_temp = rotate_complex(i_data, q_data, angle)
        yvalue = iq_temp.imag
        line_fit = np.zeros(len(yvalue)) + yvalue.mean()
        deriv_temp = ((yvalue - line_fit) ** 2).sum()
        deriv.append(deriv_temp)
    final = 0.001 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
    rotation_angle = final.ravel()[0]
    print('The rotation angle is', rotation_angle, 'pi')

    iq_new = rotate_complex(i_data, q_data, rotation_angle)
    out = cos_fit(xdata, iq_new.real, plot=plot)
    freq = out.params.valuesdict()['freq']
    period = 1.0 / freq
    pi_pulse_amp = period / 2.0
    print('Pi pulse amp is ', pi_pulse_amp, 'V')
    fit_result = cos_model(out.params, xdata)
    excited_b, ground_b = determine_ge_states(xdata, fit_result)
    if plot:
        plt.plot(xdata, iq_new.imag)
        hline()
    if updatePiPusle_amp==1:
        store_rot_info(rotation_angle, excited_b, ground_b, pi_pulse_amp)
        with open(yamlFile) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        info['pulseParams']['piPulse_gau']['amp'] = float(np.round(pi_pulse_amp, 4))
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    elif updatePiPusle_amp==2:
        if float(np.round(pi_pulse_amp, 4)) < 1:
            store_rot_info(rotation_angle, excited_b, ground_b, pi_pulse_amp)
            with open(yamlFile) as file:
                info = yaml.load(file, Loader=yaml.FullLoader)
            info['pulseParams']['piPulse_gau']['amp'] = float(np.round(pi_pulse_amp, 4))
            with open(yamlFile, 'w') as file:
                yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    else:
        pass
    return pi_pulse_amp


def rotateData(i_data, q_data, xdata=[], angleUse=None, plot=1):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    angle, excited_b, ground_b = get_rot_info()
    if angleUse is not None:
        angle = angleUse
    iq_new = rotate_complex(i_data, q_data, angle)
    if plot:
        plt.figure()
        if xdata == []:
            plt.plot(iq_new.real)
        else:
            plt.plot(xdata, iq_new.real)
        hline()
    return iq_new.real, iq_new.imag

def findAngleAndRotateData(i_data, q_data, plot=1, figName=None):
    deriv = []
    for i in range(2001):
        angle = 0.001 * i
        iq_temp = rotate_complex(i_data, q_data, angle)
        yvalue = iq_temp.imag
        line_fit = np.zeros(len(yvalue)) + yvalue.mean()
        deriv_temp = ((yvalue - line_fit) ** 2).sum()
        deriv.append(deriv_temp)
    final = 0.001 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
    rotation_angle = final.ravel()[0]
    iq_new = rotate_complex(i_data, q_data, rotation_angle)
    if plot:
        if figName is not None:
            plt.figure(figName)
        else:
            plt.figure()
        plt.plot(iq_new.real)
        plt.plot(iq_new.imag)
    return iq_new.real, iq_new.imag

def t1_fit(i_data, q_data, xdata=[], plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == []:
        t1MsmtInfo = yamlDict['regularMsmtPulseInfo']['T1MsmtTime']
        xdata = np.linspace(t1MsmtInfo[0], t1MsmtInfo[1], t1MsmtInfo[2] + 1)[:t1MsmtInfo[2]]
        xdata *= 1000
    angle, excited_b, ground_b = get_rot_info()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecay_fit(xdata, iq_new.real, plot=plot)
    print('qubit T1 is ' + str(np.round(out.params.valuesdict()['t1Fit'], 3)) + 'ns')
    r2 = 1-np.sum(out.residual**2)/np.sum((iq_new.real-np.average(iq_new.real))**2)
    print('r-squared is ' + str(r2))
    if plot:
        hline()
    return out.params.valuesdict()['t1Fit'], r2


def t2_ramsey_fit(i_data, q_data, xdata=[], plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == []:
        t2MsmtInfo = yamlDict['regularMsmtPulseInfo']['T2MsmtTime']
        xdata = np.linspace(t2MsmtInfo[0], t2MsmtInfo[1], t2MsmtInfo[2] + 1)[:t2MsmtInfo[2]]
        xdata *= 1000
    angle, excited_b, ground_b = get_rot_info()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecayWithCos_fit(xdata, iq_new.real, plot=plot)
    f_detune = np.round(out.params.valuesdict()['freq'], 9)
    t2R = np.round(out.params.valuesdict()['t2Fit'], 3)
    print('qubit T2R is ' + str(t2R) + 'ns')
    print('The qubit drive frequency has been detuned', f_detune, ' GHz')
    if plot:
        hline()
    return t2R, f_detune


def t2_echo_fit(i_data, q_data, xdata=[], plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == []:
        t2MsmtInfo = yamlDict['regularMsmtPulseInfo']['T2MsmtTime']
        xdata = np.linspace(t2MsmtInfo[0], t2MsmtInfo[1], t2MsmtInfo[2] + 1)[:t2MsmtInfo[2]]
        xdata *= 1000
    angle, excited_b, ground_b = get_rot_info()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecay_fit(xdata, iq_new.real, plot=plot)
    t2E = np.round(out.params.valuesdict()['t1Fit'], 3)
    print('qubit T2E is ' + str(t2E) + 'ns')
    
    if plot:
        hline()
    return t2E

def DRAGTuneUp_fit(i_data, q_data, xdata, update_dragFactor=False, plot=True):
    angle, excited_b, ground_b = get_rot_info()
    iq_new = rotate_complex(i_data, q_data, angle)

    out, k, b = linear_fit(xdata, iq_new.real[::2]-iq_new.real[1::2], plot=plot)
    kfit = np.round(out.params['k'].value, 3)
    bfit = np.round(out.params['b'].value, 3)
    x0 = -bfit/kfit
    print('DRAG factor is ' + str(x0) )
    if plot:
        plt.plot(xdata, np.zeros(len(xdata)))
    if update_dragFactor:
        with open(yamlFile) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        info['pulseParams']['piPulse_gau']['dragFactor'] = float(np.round(x0, 4))
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    return x0

def DRAGTuneUp_fitGpct(g_pct, xdata, update_dragFactor=False, plot=True):
    out, k, b = linear_fit(xdata, g_pct[::2]-g_pct[1::2], plot=plot)
    kfit = np.round(out.params['k'].value, 3)
    bfit = np.round(out.params['b'].value, 3)
    x0 = -bfit/kfit
    print('DRAG factor is ' + str(x0) )
    if plot:
        plt.plot(xdata, np.zeros(len(xdata)))
    if update_dragFactor:
        with open(yamlFile) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        info['pulseParams']['piPulse_gau']['dragFactor'] = float(np.round(x0, 4))
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    return x0

def RB_fit(g_pct, n_gates, CI=True, plot=True):
    out, ci = exponetialDecay_fit(n_gates, g_pct, CI=CI, plot=False)
    lmf.report_ci(ci)
    pfit = np.round(np.exp(-1 / out.params['t1Fit'].value), 7)
    pfit_p = np.round(np.exp(-1 / ci['t1Fit'][2][1]), 7)
    pfit_n = np.round(np.exp(-1 / ci['t1Fit'][4][1]), 7)
    afit = np.round(out.params['amp'].value, 9)
    ofit = np.round(out.params['offset'].value, 9)
    print((ci['t1Fit'][4][1], ci['t1Fit'][3][1]))
    print('p is ' + str([pfit_p, pfit, pfit_n]) + '; sigma is' + str(pfit_n-pfit))
    if plot:
        plt.figure()
        plt.plot(n_gates, g_pct, 'o', label='data')
        plt.plot(n_gates, exponetialDecay_model(out.params, n_gates), '-', label=r"fit p: " + str(pfit) )
        plt.legend()
    return out.params

def findExtremeByPolyFitting(xdata: Union[list, np.array], ydata: Union[list, np.array],
                             poly_order: int = 4, plot=True, figureName=None):
    """
    find the extreme point of experiment data by fitting to polynomial
    :param xdata: x data
    :param ydata: y data
    :param poly_order: order of polynomial to fit
    :param plot:
    :return:
    """
    fit_params = np.polyfit(xdata, ydata, poly_order)
    fit_func = np.poly1d(fit_params)
    fit_func_extremes = fit_func.deriv(1).roots
    extreme_condition = (fit_func_extremes > np.min(xdata)) & (fit_func_extremes < np.max(xdata)) \
                        & (np.imag(fit_func_extremes)==0)
    extremes_in_xrange = np.real(fit_func_extremes[extreme_condition])
    if plot:
        if figureName is not None:
            plt.figure(figureName)
        else:
            plt.figure()
        plt.title(f"fit to {poly_order}th order polynomial")
        plt.plot(xdata, ydata)
        plt.plot(xdata, fit_func(xdata))
        plt.plot(extremes_in_xrange, fit_func(extremes_in_xrange), "*", color="g")
    print(extremes_in_xrange)
    print(fit_func(extremes_in_xrange))
    return extremes_in_xrange, fit_func(extremes_in_xrange)

## coherentfit

def lineFit(x, k, b):
    return x * k + b

def coherent(alpha, n):
    number = np.arange(n)
    mag = np.zeros(n)
    for i in range(n):
        mag[i] = np.exp(-alpha**2/2) * alpha**i/np.sqrt(np.math.factorial(i))
    return number, mag

def lorentzian(x, x0, a, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def multi_lorentz(x, params):
    off = params[0]
    paramsRest = params[1:]
    assert not (len(paramsRest) % 3)
    return off + sum([lorentzian(x, *paramsRest[i:i+3]) for i in range(0, len(paramsRest), 3)])

def numberSel(xdata, piAmp, offset, alpha, chi, kappa):
    n = 10
    number, mag = coherent(alpha, n)
    params = [offset]
    x0List = 0.1 + np.arange(n) * chi
    for i in range(n):
        params += [x0List[i], -mag[i] * piAmp, kappa]
    return multi_lorentz(xdata, params)

def gAlphaStateMSMT(n, alpha, prepareErr, msmtErr_g, msmtErr_e): # assume pipulse error is symetric
    pn = np.exp(-alpha**2) * alpha**(2*n)/factorial(n) # possibility of n photon in SCav
    pn_g = pn * ((1-prepareErr) * msmtErr_e + prepareErr * (1-msmtErr_g)) # possibility of qubit in g when n photon in Scav
    pNotN_g = (1-pn) * ((1-prepareErr) *  (1-msmtErr_g) + prepareErr * msmtErr_e) # possibility of qubit in g when not n photon in Scav
    return pn_g + pNotN_g

def fitCoherent(xdata, ydata, **kwargs):
    offset0 = kwargs.get("offset0", 1)
    alpha0 = kwargs.get("alpha0", 1)
    chi0 = kwargs.get("chi0", 2e-3)
    kappa0 = kwargs.get("kappa0", 1e-4)
    piAmp0 = kwargs.get("piAmp0", 0.5)
    popt, pcov = curve_fit(numberSel, xdata, ydata, p0=(piAmp0, offset0, alpha0, chi0, kappa0), sigma=ydata**2, bounds=[[piAmp0*0.5, 0.5, 0, chi0*0.5, kappa0*0.1], [piAmp0*2, 1.1, 5, chi0*1.5, kappa0*10]], maxfev=100000, xtol=1e-10, ftol=1e-10)
    print(popt)
    plt.figure('fitting')
    plt.plot(xdata, ydata, '*')
    xdataplot = np.linspace(min(xdata), max(xdata), 1001)
    plt.plot(xdataplot, numberSel(xdataplot, *popt), '-', label=f"alpha={popt[2]}")
    plt.legend()
    alpha = popt[2]
    chi = popt[3]
    kappa = popt[4]
    print('alpha: ', popt[2])
    print('chi: ', popt[3] * 1e3, 'MHz')
    return popt

def fitCoherentWithFixParams(xdata, ydata, **params):
    offset0 = params.get('offset0', 1)
    alpha0 = params.get('alpha0', 1)
    chi0 = params.get('chi0', 0.001)
    kappa0 = params.get('kappa0', 1e-4)
    piAmp0 = params.get('piAmp0', 0.5)
    popt, pcov = curve_fit(numberSel, xdata, ydata, p0=(piAmp0, offset0, alpha0, chi0, kappa0), sigma=ydata**2, bounds=[[piAmp0-1e-9, offset0-1e-9, 0, chi0 - 1e-9, kappa0-1e-9], [piAmp0+1e-9, offset0+1e-9, 5, chi0 + 1e-9, kappa0+1e-9]], maxfev=100000, xtol=1e-10, ftol=1e-10)
    print(popt)
    plt.figure('fitting')
    plt.plot(xdata, ydata, '*')
    xplot = np.linspace(xdata[0], xdata[-1], 101)
    plt.plot(xplot, numberSel(xplot, *popt), '-')
    alpha = popt[2]
    chi = popt[3]
    kappa = popt[4]
    print('alpha: ', popt[2])
    print('chi: ', popt[3] * 1e3, 'MHz')
    return alpha

def fitCoherentWithPeaks(nList, gPctList, plot=True, **params):
    alpha0 = params.get('alpha0', 1)
    prepareErr0 = params.get('prepareErr0', 0.05)
    msmtErr_g0 = params.get('msmtErr0', 0.05)
    msmtErr_e0 = params.get('msmtErr0', 0.1)
    popt, pcov = curve_fit(gAlphaStateMSMT, nList, gPctList, p0=(alpha0, prepareErr0, msmtErr_g0, msmtErr_e0),
                           bounds=[[0, 0, 0, 0], [5, 0.1, 0.1, 0.3]], maxfev=100000, xtol=1e-10, ftol=1e-10)
    if plot:
        plt.figure('fitting_g_n')
        l_ = plt.plot(nList, gPctList, '*')[0]
        plt.plot(nList, gAlphaStateMSMT(nList, *popt), '+-', color=l_.get_color(), label=f"alpha={popt[0]}")
        plt.legend()
    return popt



#=========================helpers============================================================
def updateYAML(newParamDict: dict):
    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root, items, value):
        """Set a value in a nested object in root by item sequence."""
        data_type =  type(get_by_path(root, items))
        get_by_path(root, items[:-1])[items[-1]] = data_type(value)

    def type_convert(obj):
        """convert numpy type to native python types"""
        try:
            converted_value = getattr(obj, "tolist", lambda: value)()
            converted_value = getattr(obj, "tolist", lambda: obj)()
        except NameError:
            converted_value = obj
        return converted_value

    with open(yamlFile) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
        for s, val in newParamDict.items():
            set_by_path(info, s.split("."), type_convert(val))
    with open(yamlFile, 'w') as file:
        yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)

def saveExperimentYaml(saveDir: str, fileName: str, yamlfile: dict):
    duplicate = 0
    saveSuccess = 0
    saveName = fileName
    while saveSuccess == 0:
        if os.path.isfile(saveDir+saveName+".yaml"):
            saveName = fileName + '_' + str(duplicate)
            duplicate += 1
        else:
            with open(saveDir+saveName+".yaml", 'w') as file:
                yaml.dump(yamlfile, file)
            saveSuccess = 1


if __name__ == '__main__':
    params = lmf.Parameters()
    params.add('amp', value=5000)
    params.add('offset', value=30000)
    params.add('phase', value=0)
    params.add('freq', value=1.2)
    xdata = np.linspace(-1, 1, 101)[:100]
    ydata = (cos_model(params, xdata)) + np.random.rand(len(xdata)) * 3000.0
    res = cos_fit(xdata, ydata)
    print(res.params)
    print(res.init_values)


    params = lmf.Parameters()
    params.add('amp', value=5000)
    params.add('offset', value=30000)
    params.add('phase', value=0)
    params.add('t2Fit', value=20)
    params.add('freq', value=0.03)
    xdata = np.linspace(0, 200, 101)[:100]
    ydata = (exponetialDecayWithCos_model(params, xdata)) + np.random.rand(len(xdata)) * 100.0
    res = exponetialDecayWithCos_fit(xdata, ydata)
    print(res.params)
    print(res.init_values)
    print(res)


    plt.show()