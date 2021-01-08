import numpy as np
import matplotlib.pyplot as plt
import h5py
import lmfit as lmf
import math
import yaml
from typing import List, Callable, Union, Tuple, Dict
import warnings
from nptyping import NDArray

yamlFile = '1224Q5_info.yaml'

def processDataReceive(subbuffer_used, dataReceive, plot=0):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)

    if subbuffer_used:
        Iarray = dataReceive['D1']['ch1'][:, 2::5]
        Qarray = dataReceive['D1']['ch1'][:, 3::5]
        I = np.average(Iarray, axis=0)
        Q = np.average(Qarray, axis=0)
        if plot:
            plt.figure(figsize=(9, 4))
            plt.subplot(121)
            plt.plot(I)
            plt.plot(Q)
            plt.subplot(122)
            plt.hist2d(Iarray.flatten(), Qarray.flatten(), bins=101, range=yamlDict['histRange'])
        return I, Q

    else:
        dig = yamlDict['combinedChannelUsage']['Dig']
        demod_sigI = dataReceive[dig['Sig'][0]]['ch' + str(dig['Sig'][1])][:, 0, 0::5]
        demod_sigQ = dataReceive[dig['Sig'][0]]['ch' + str(dig['Sig'][1])][:, 0, 1::5]
        demod_sigMag = dataReceive[dig['Sig'][0]]['ch' + str(dig['Sig'][1])][:, 0, 2::5]
        demod_refI = dataReceive[dig['Ref'][0]]['ch' + str(dig['Ref'][1])][:, 0, 0::5]
        demod_refQ = dataReceive[dig['Ref'][0]]['ch' + str(dig['Ref'][1])][:, 0, 1::5]
        demod_refMag = np.average(dataReceive[dig['Ref'][0]]['ch' + str(dig['Ref'][1])][:, 0, 2::5])

        demod_I = (demod_sigI * demod_refI + demod_sigQ * demod_refQ) / demod_refMag
        demod_Q = (-demod_sigI * demod_refQ + demod_sigQ * demod_refI) / demod_refMag

        Itrace = np.average(demod_I, axis=0)
        Qtrace = np.average(demod_Q, axis=0)

        if plot:
            xdata = np.arange(len(Itrace)) * 10
            plt.figure(figsize=(8, 8))
            plt.subplot(221)
            plt.plot(xdata, Itrace, label="I")
            plt.plot(xdata, Qtrace, label="Q")
            plt.legend()
            plt.subplot(222)
            plt.plot(xdata, np.sqrt(Itrace ** 2 + Qtrace ** 2), label="Mag1")
            plt.plot(xdata, np.average(demod_sigMag, axis=0), label='Mag2')
            plt.legend()
            plt.subplot(223)
            plt.plot(Itrace, Qtrace)
            plt.subplot(224)
            sumStart = yamlDict['FPGAConfig'][dig['Sig'][0]]['ch' + str(dig['Sig'][1])]['integ_start']
            sumEnd = yamlDict['FPGAConfig'][dig['Sig'][0]]['ch' + str(dig['Sig'][1])]['integ_stop']

            plt.hist2d(np.sum(demod_I[:, sumStart // 10:sumEnd // 10], axis=1),
                       np.sum(demod_Q[:, sumStart // 10:sumEnd // 10], axis=1), bins=101)

        sigTruncInfo = get_recommended_truncation(demod_sigI, demod_sigQ, sumStart, sumEnd, current_demod_trunc=yamlDict['FPGAConfig'][dig['Sig'][0]]['ch' + str(dig['Sig'][1])]['demod_trunc'])
        refTruncInfo = get_recommended_truncation(demod_refI, demod_refQ, sumStart, sumEnd, current_demod_trunc=yamlDict['FPGAConfig'][dig['Ref'][0]]['ch' + str(dig['Ref'][1])]['demod_trunc'])

        print('sig truncation: ', sigTruncInfo)
        print('ref truncation: ', refTruncInfo)


        return (demod_I, demod_Q, demod_sigMag)


def processDataReceiveWithMultipleMsmt(subbuffer_used, dataReceive, msmtTime=2, plot=0):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    Ilist = []
    Qlist = []
    if subbuffer_used:
        indexInterval = msmtTime * 5
        for i in range(msmtTime):
            Iarray = dataReceive['D1']['ch1'][:, (5 * i + 2)::indexInterval]
            Qarray = dataReceive['D1']['ch1'][:, (5 * i + 3)::indexInterval]
            if plot:
                plt.figure()
                plt.hist2d(Iarray.flatten(), Qarray.flatten(), bins=101, range=yamlDict['histRange'])
            Ilist.append(Iarray)
            Qlist.append(Qarray)
    else:
        raise TypeError('Could not process multiple msmt without subbuffer used.')
    return Ilist, Qlist


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


def get_data():
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    angle = yamlDict['fitParams']['angle']
    excitedDigV = yamlDict['fitParams']['excitedDigV']
    groundDigV = yamlDict['fitParams']['groundDigV']
    return angle, excitedDigV, groundDigV


def info_store(angle, excited, ground, piPulse_amp):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    yamlDict['fitParams']['angle'] = float(np.round(angle), 4)
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


def get_rot_data(i_data, q_data, xdata, plot=True):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    return iq_new.real


def determin_ge_states(xdata, ydata):
    mid = ydata[int(len(ydata) / 2)]
    excited = round(ydata.max(), 2)
    ground = round(ydata.min(), 2)
    if np.abs(excited - mid) < np.abs(ground - mid):
        excited, ground = ground, excited
    print('The Excited State Voltage is', excited)
    print('The Ground State Voltage is', ground)
    return excited, ground


def hline():
    agnle, excited, ground = get_data()
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


def cos_fit(xdata, ydata, plot=True):
    offset = np.average(ydata)
    amp = (np.max(np.abs(ydata)) - np.min(np.abs(ydata))) / 2.0
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
    print(amp, offset, freq, phase)
    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp, min=amp * 0.8, max=amp * 1.2, vary=True)
    fit_params.add('offset', value=offset, min=offset*0.8, max=offset*1.2, vary=True)
    fit_params.add('phase', value=phase, min=-np.pi, max=np.pi, vary=True)
    fit_params.add('freq', value=freq, min=0, vary=True)
    out = lmf.minimize(_residuals, fit_params, method='powell', args=(cos_model, xdata, ydata))
    if plot:
        plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        plt.plot(xdata, cos_model(out.params, xdata), '-', label='fit period/2:' + str(np.round(1.0 / out.params['freq'] / 2.0, 3)) + ' unit')
        plt.legend()
    return out


def exponetialDecay_model(params, xdata):
    value = params.valuesdict()
    amp = value['amp']
    offset = value['offset']
    t1Fit = value['t1Fit']
    ydata = (np.exp(-xdata / t1Fit)) * amp + offset
    return ydata.view(np.float)


def exponetialDecay_fit(xdata, ydata, plot=True):
    offset_ = (ydata[-1])
    amp_ = (ydata[0]) - (ydata[-1])
    t1Fit_ = (1.0 / 3.0) * (xdata[-1] - xdata[0])
    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp_, vary=True)
    fit_params.add('offset', value=offset_, vary=True)
    fit_params.add('t1Fit', value=t1Fit_, min=0, vary=True)
    out = lmf.minimize(_residuals, fit_params, method='powell', args=(exponetialDecay_model, xdata, ydata))
    if plot:
        plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        plt.plot(xdata, exponetialDecay_model(out.params, xdata), '-', label='fit T1: ' + str(np.round(out.params['t1Fit'].value, 3)) + ' unit')
        plt.legend()
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


def exponetialDecayWithCos_fit(xdata, ydata, plot=True):
    amp = (np.max(ydata) - np.min(ydata)) / 2.0
    t2Fit = (1 / 4.0) * (xdata[-1] - xdata[0])
    offset = ydata[-1]
    fourier_transform = np.fft.fft(ydata)
    max_point = np.argmax(np.abs(fourier_transform[1: len(fourier_transform) // 2]))
    time_spacing = xdata[1] - xdata[0]
    f_array = np.fft.fftfreq(len(fourier_transform), d=time_spacing)
    freq = f_array[max_point]
    phase = np.arctan2(np.imag(fourier_transform[max_point]), np.real(fourier_transform[max_point]))
    fit_params = lmf.Parameters()
    fit_params.add('amp', value=amp, vary=True)
    fit_params.add('offset', value=offset, vary=True)
    fit_params.add('t2Fit', value=t2Fit, min=0, vary=True)
    fit_params.add('phase', value=phase, min=-2 * np.pi, max=2 * np.pi, vary=True)
    fit_params.add('freq', value=freq, min=0, vary=True)
    out = lmf.minimize(_residuals, fit_params, method='powell', args=(exponetialDecayWithCos_model, xdata, ydata))
    if plot:
        plt.figure()
        plt.plot(xdata, ydata, '*', label='data')
        plt.plot(xdata, exponetialDecayWithCos_model(out.params, xdata), '-', label='fit T2: ' + str(np.round(out.params['t2Fit'].value, 3)) + ' unit')
        plt.legend()
    return out


############################## For specific fitting object ################################################


def pi_pulse_tune_up(i_data, q_data, xdata=None, updatePiPusle_amp=0, plot=1):
    """
    fitting pi_pulse_tune_up as a sin function
    """
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == None:
        piPulseAmpInfo = yamlDict['regularMsmtPulseInfo']['piPulseTuneUpAmp']
        xdata = np.linspace(piPulseAmpInfo[0], piPulseAmpInfo[1], piPulseAmpInfo[2] + 1)[:100]
    deriv = []
    for i in range(7001):
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
    excited_b, ground_b = determin_ge_states(xdata, fit_result)
    info_store(rotation_angle, excited_b, ground_b, pi_pulse_amp)
    if plot:
        plt.plot(xdata, iq_new.imag)
        hline()
    if updatePiPusle_amp:
        with open(yamlFile) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        info['pulseParams']['piPulse_gau']['amp'] = float(np.round(pi_pulse_amp, 4))
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    return pi_pulse_amp


def rotateData(i_data, q_data, plot=1):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    if plot:
        plt.figure()
        plt.plot(iq_new.real)
        hline()
    return iq_new.real, iq_new.imag


def t1_fit(i_data, q_data, xdata=None, plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == None:
        t1MsmtInfo = yamlDict['regularMsmtPulseInfo']['T1MsmtTime']
        xdata = np.linspace(t1MsmtInfo[0], t1MsmtInfo[1], t1MsmtInfo[2] + 1)[:100]
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecay_fit(xdata, iq_new.real, plot=plot)
    print('qubit T1 is ' + str(np.round(out.params.valuesdict()['t1Fit'], 3)) + 'ns')
    if plot:
        hline()
    return out.params.valuesdict()['t1Fit']


def t2_ramsey_fit(i_data, q_data, xdata=None, plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == None:
        t2MsmtInfo = yamlDict['regularMsmtPulseInfo']['T2MsmtTime']
        xdata = np.linspace(t2MsmtInfo[0], t2MsmtInfo[1], t2MsmtInfo[2] + 1)[:100]
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecayWithCos_fit(xdata, iq_new.real, plot=plot)
    f_detune = np.round(out.params.valuesdict()['freq'], 6)
    t2R = np.round(out.params.valuesdict()['t2Fit'], 3)
    print('qubit T2R is ' + str(t2R) + 'ns')
    print('The qubit drive frequency has been detuned', f_detune, ' MHz')
    if plot:
        hline()
    return t2R, f_detune


def t2_echo_fit(i_data, q_data, xdata=None, plot=True):
    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata == None:
        t2MsmtInfo = yamlDict['regularMsmtPulseInfo']['T2MsmtTime']
        xdata = np.linspace(t2MsmtInfo[0], t2MsmtInfo[1], t2MsmtInfo[2] + 1)[:100]
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    out = exponetialDecay_fit(xdata, iq_new.real, plot=plot)
    t2E = np.round(out.params.valuesdict()['t1Fit'], 3)
    print('qubit T2E is ' + str(t2E) + 'ns')
    if plot:
        hline()
    return t2E

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