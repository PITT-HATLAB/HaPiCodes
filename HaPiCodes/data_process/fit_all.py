# -*- coding: utf-8 -*-
"""
Created for all the data we get from the qubit experiment

@author: Pinlei Lu (Hat Lab)
email: pil9@pitt.edu
phone: 412-515-5602
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import lmfit as lmf
import math
import yaml

yamlFile = ''


def read_value(name):
    """
    read the h5py file.
    """
    data_temp = h5py.File(name, 'r')
    i_data = data_temp['I_data'].value
    q_data = data_temp['Q_data'].value
    xdata = data_temp['xdata'].value
    data_temp.close()
    return i_data, q_data, xdata


def rotate_complex(real_part, imag_part, angle):
    """
    rotate the complex number as rad units.
    """
    iq_new = (real_part + 1j * imag_part) * np.exp(1j * np.pi * angle)
    return iq_new


def get_data():
    with open(yamlFile) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    angle = info['fitParams']['angle']
    excitedDigV = info['fitParams']['excitedDigV']
    groundDigV = info['fitParams']['groundDigV']
    return angle, excitedDigV, groundDigV


########################################################################

def residuals(params, sin_fit, x, y):
    return sin_fit(params, x) - y


def sin_fit(params, x, do_float=True):
    value = params.valuesdict()
    A = value['A']
    B = value['B']
    freq = value['freq']
    phase = value['phase']

    fit = A * np.sin(2 * np.pi * freq * x + phase) + B
    if not do_float:
        return fit
    return fit.view(np.float)


def sin(x, y):
    A = (np.max(y) - np.min(y)) / 2.0
    B = np.average(y)
    fourier_transform = np.fft.fft(y)
    max_point = np.argmax(np.abs(fourier_transform[1: len(fourier_transform) // 2]))

    time_spacing = x[1] - x[0]
    freq_array = np.fft.fftfreq(len(fourier_transform), d=time_spacing)

    outs = []
    for i in range(3):
        f_max = freq_array[max_point + i]
        phase = np.arctan2(np.imag(fourier_transform[max_point + i]), np.real(fourier_transform[max_point + i]))

        fit_params = lmf.Parameters()
        fit_params.add('A', value=A, vary=True, min=0)
        fit_params.add('B', value=B, vary=True)
        fit_params.add('freq', value=f_max, vary=True, min=0)
        fit_params.add('phase', value=phase, vary=True, min=0, max=2 * np.pi)
        outs += [lmf.minimize(residuals, fit_params, args=(sin_fit, x, y))]
    minchi = None
    index_minchi = None
    for j in range(len(outs)):
        rchi = outs[j].redchi
        if minchi is None or minchi > rchi:
            minchi = rchi
            index_minchi = j
    return outs[index_minchi]


def determin_ge_states(tWave, fitwave, IQNew, plot=False):
    mid = fitwave[int(len(fitwave) / 2)]
    excited = round(fitwave.max(), 2)
    ground = round(fitwave.min(), 2)
    errorE = round(IQNew.real.max() - excited, 2)
    errorG = round(ground - IQNew.real.min(), 2)
    if np.abs(excited - mid) < np.abs(ground - mid):
        excited, ground = ground, excited
        errorE, errorG = errorG, errorE
    print('The Excited State Voltage is', excited, '+/-', errorE)
    print('The Ground State Voltage is', ground, '+/-', errorG)
    if plot:
        plt.figure()
        plt.xlabel('DAC_value')
        plt.ylabel('volts')
        plt.title('Fit_Pi_pulse')
        plt.plot(tWave, IQNew.real, 'b+')
        plt.plot(tWave, IQNew.imag)
        plt.plot(tWave, fitwave, 'r')
    return excited, ground


def info_store(angle, excited, ground, piPulse_amp):
    with open(yamlFile) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    info['fitParams']['angle'] = float(angle)
    info['fitParams']['excitedDigV'] = float(excited)
    info['fitParams']['groundDigV'] = float(ground)
    info['fitParams']['piPulse_amp'] = float(piPulse_amp)
    with open(yamlFile, 'w') as file:
        yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)

    print('info successly stored')
    return


def ef_pi_pulse_tune_up(i_data, q_data, xdata):
    """
    fitting ef_pi_pulse_tune_up as a sin function
    """
    deriv = []
    for i in range(101):
        angle = 0.01 * i
        iq_temp = rotate_complex(i_data, q_data, angle)
        yvalue = iq_temp.imag
        line_fit = np.zeros(len(yvalue)) + yvalue.mean()
        deriv_temp = ((yvalue - line_fit) ** 2).sum()
        deriv.append(deriv_temp)
    final = 0.01 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
    rotation_angle = final.ravel()[0]
    print('The rotation angle is', rotation_angle, 'pi')
    iq_new = rotate_complex(i_data, q_data, rotation_angle)
    out = sin(xdata, iq_new.real)
    freq = out.params.valuesdict()['freq']
    period = 1.0 / freq
    pi_pulse_amp = period / 2.0
    pi_2_pulse_amp = period / 4.0
    print('ef Pi pulse amp is ', pi_pulse_amp, 'V')
    print('ef Pi over 2 pulse amp is ', pi_2_pulse_amp, 'V')
    fit_result = sin_fit(out.params, xdata)
    excited_b, ground_b = determin_ge_states(xdata, fit_result, iq_new,
                                             plot=True)
    #    hline(excited_b, ground_b)
    #    info_store(rotation_angle, excited_b, ground_b, pi_pulse_amp,
    #               pi_2_pulse_amp)
    return


def fit(freq, real, imag, plot=True, method=None):
    if method == 't1':
        out = t1(freq, real, imag)
        F0 = 0
        print('!!! T1 is', float(out.params.valuesdict()['t1']), 'us')
        if plot:
            result = t1_model(out.params, freq)

    elif method == 't2R':
        out = t2(freq, real, imag)
        print('!!! T2 Ramsey is', float(out.params.valuesdict()['T2']), 'us')
        F0 = float(out.params.valuesdict()['f_max'])
        if plot:
            result = t2_model(out.params, freq)

    elif method == 't2E':
        out = t1(freq, real, imag)
        print('!!! T2 Echo is', float(out.params.valuesdict()['t1']), 'us')
        F0 = 0
        if plot:
            result = t1_model(out.params, freq)
    if plot:
        plot_fit(freq, real, imag, result, method=method)

    return F0, out


def plot_fit(freq, real, imag, result, method):
    '''
    This function plots the recieved data against the fit.

    Input:
    ------

        **freq**:
        An array of floats of size n that represents frequencies.

        **real**:
        An array of floats of size n that represents the real component of that which is fitted.

        **imag**:
        An array of floats of size n that represents the imag component of that which is fitted.

        **result**:
        An array of floats that contains the representation of the final fit function.

    Output:
    ------
        None

    Raises:
    -------
        None
    '''

    plt.figure()
    plt.plot(freq, real)
    plt.plot(freq, imag)
    plt.plot(freq, result[0:-1:2])
    plt.xlabel("Time(us)")
    plt.ylabel("Volts")

    if method == 't1':
        plt.title("T1")

    if method == 't2R':
        plt.title("T2_Ramsey")

    if method == 't2E':
        plt.title("T2_Echo")


def _residuals(params, model, real, imag, omega):
    '''Used internally (should be private but NO)'''
    model_data = model(params, omega)
    sample_data = real + imag * 1j
    return model_data - np.array(sample_data).view(np.float)


def t1_model(params, f, do_float=True):
    value = params.valuesdict()
    A = value['A']
    Ao = value['Ao']
    t1 = value['t1']
    S_21 = (np.exp(-f / t1)) * A + Ao + 0 * 1j
    if not do_float:
        return S_21
    return S_21.view(np.float)


def t1(freq, data_real, data_imag):
    A = (data_real[0]) - (data_real[-1])
    Ao = (data_real[-1])
    t1 = (1.0 / 3.0) * (freq[-1] - freq[0])
    fit_params = lmf.Parameters()
    fit_params.add('A', value=A, vary=True)
    fit_params.add('Ao', value=Ao, vary=True)
    fit_params.add('t1', value=t1, min=0, vary=True)
    out = lmf.minimize(_residuals, fit_params, args=(t1_model, data_real, data_imag, freq))
    return out


def t2_model(params, t, do_float=True):
    value = params.valuesdict()
    A = value['A']
    f_max = value['f_max']
    B = value['B']
    T2 = value['T2']
    D = value['D']
    S_21 = A * np.cos(f_max * np.pi * 2 * t + B) * np.exp(-t / T2) + D + 0 * 1j
    if not do_float:
        return S_21
    return S_21.view(np.float)


def t2(freq, data_real, data_imag):
    A = (np.max(data_real) - np.min(data_real)) / 2.0
    T2 = (1 / 4.0) * (freq[-1] - freq[0])
    D = data_real[-1]
    fourier_transform = np.fft.fft(data_real)
    max_point = np.argmax(np.abs(fourier_transform[1: len(fourier_transform) // 2]))
    time_spacing = freq[1] - freq[0]
    f_array = np.fft.fftfreq(len(fourier_transform), d=time_spacing)
    f_max = f_array[max_point]
    B = np.arctan2(np.imag(fourier_transform[max_point]), np.real(fourier_transform[max_point]))
    fit_params = lmf.Parameters()
    fit_params.add('A', value=A, vary=True)
    fit_params.add('D', value=D, vary=True)
    fit_params.add('T2', value=T2, min=0, vary=True)
    fit_params.add('B', value=B, min=-2 * np.pi, max=2 * np.pi, vary=True)
    fit_params.add('f_max', value=f_max, min=0, vary=True)
    out = lmf.minimize(_residuals, fit_params, args=(t2_model, data_real, data_imag, freq))
    return out


def t1_phase_fit(i_data, q_data, xdata, plot=True):
    phase = np.unwrap(np.angle(i_data + 1j * q_data))
    F0, out = fit(xdata, phase, np.zeros(len(phase)), plot=plot, method='t1')
    return out.params.valuesdict()['t1']


def allxy(i_data, q_data, xdata, plot_in_one_fig=False, plt_options={}):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    if plot_in_one_fig:
        plt.figure('AllXY')
    else:
        plt.figure()
    plt.plot(iq_new.real, 'o-', **plt_options)
    hline(excited_b, ground_b)
    return iq_new.real


def get_rot_data(i_data, q_data, xdata, plot=True):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    return iq_new.real


def hline(excited, ground):
    plt.axhline(y=excited, color='r', linestyle='--')  # , label = 'Excited')
    plt.axhline(y=ground, color='b', linestyle='--')  # , label = 'Ground')
    plt.axhline(y=(excited + ground) / 2.0, color='y', linestyle='--')
    # plt.legend(loc=5)
    plt.show()


#######################################################################################
#######################################################################################
#######################################################################################


def pi_pulse_tune_up(i_data, q_data, xdata):
    """
    fitting pi_pulse_tune_up as a sin function
    """
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
    out = sin(xdata, iq_new.real)
    freq = out.params.valuesdict()['freq']
    period = 1.0 / freq
    pi_pulse_amp = period / 2.0
    pi_2_pulse_amp = period / 4.0
    print('Pi pulse amp is ', pi_pulse_amp, 'V')
    fit_result = sin_fit(out.params, xdata)
    excited_b, ground_b = determin_ge_states(xdata, fit_result, iq_new,
                                             plot=True)
    hline(excited_b, ground_b)
    info_store(rotation_angle, excited_b, ground_b, pi_pulse_amp)
    return


def t1_fit(i_data, q_data, xdata, plot=True):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    F0, out = fit(xdata, iq_new.real, iq_new.imag, plot=plot, method='t1')
    if plot:
        hline(excited_b, ground_b)
    return out.params.valuesdict()['t1']


def t2_ramsey_fit(i_data, q_data, xdata, plot=True):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    f_detune, out = fit(xdata, iq_new.real, iq_new.imag,
                        plot=plot, method='t2R')
    print('The qubit drive frequency has been detuned', f_detune, ' MHz')
    if plot:
        hline(excited_b, ground_b)
    return out.params.valuesdict()['T2'], f_detune


def t2_echo_fit(i_data, q_data, xdata, plot=True):
    angle, excited_b, ground_b = get_data()
    iq_new = rotate_complex(i_data, q_data, angle)
    F0, out = fit(xdata, iq_new.real, iq_new.imag, plot=plot, method='t2E')
    if plot:
        hline(excited_b, ground_b)
    return out.params.valuesdict()['t1']


if __name__ == '__main__':
    print(get_data())
    info_store(2, 3, 4 , 8 )