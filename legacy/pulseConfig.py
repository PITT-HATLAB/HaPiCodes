import numpy as np
import warnings
import logging
from scipy import signal
import matplotlib.pyplot as plt
from HaPiCodes.sd1_api import keysightSD1

class modulesWaveformCollection(object):
    def __init__(self, module_dict):
        for module in module_dict.keys():
            setattr(self, str(module), {})
        self.module_dict = module_dict
        self.waveInfo = {}
        return

    def __dir__(self):
        return self.module_dict.keys()


class queueCollection(object):

    """queue collections for each module (AWG)
    chan{int}: dict. The queue(dictionary) for each channel in the module. Eg: chan1, chan2, chan3, chan4
    """

    def __init__(self, chanNum=4):
        for i in range(1, chanNum + 1):
            setattr(self, f'ch{i}', {})
        return


class modulesQueueCollection(object):
    def __init__(self, module_dict):
        self.dig_trig_num_dict = {}
        for module_name, module in module_dict.items():
            try:
                module_ch_num = int(module.instrument.getOptions("channels")[-1])
                if isinstance(module.instrument, keysightSD1.SD_AIN):
                    self.dig_trig_num_dict[module_name] = {f"ch{i + 1}": 0 for i in range(module_ch_num)}
            except AttributeError:
                module_ch_num = 4 #dummy module has 4 channels by default

            setattr(self, str(module_name), queueCollection(module_ch_num))
        self.module_dict = module_dict
        self.maxIndexNum = 0

        return

    def add(self, module: str, channel: int, waveIndex: int, pulse, timeDelay: int, msmt:bool = False):
        module_ = getattr(self, module)
        try:
            q = getattr(module_, f'ch{channel}')
        except AttributeError:
            raise AttributeError("Check the channel number of AWG module")
        if str(waveIndex) not in q.keys():
            getattr(module_, f'ch{channel}')[str(waveIndex)] = []
            self.maxIndexNum = np.max([int(waveIndex), self.maxIndexNum])
        timeDelay = np.ceil(timeDelay/10)*10
        getattr(module_, f'ch{channel}')[str(waveIndex)].append([pulse, timeDelay])

        if (module in self.dig_trig_num_dict.keys()) and msmt:
            self.dig_trig_num_dict[module][f"ch{channel}"] += 1

    def addTwoChan(self, AWG: str, channel: list, waveIndex: list, pulse: list, timeDelay: int):
        if len(channel) != 2:
            raise KeyError("channel number must be two!")
        for i in range(2):
            self.add(AWG, channel[i], waveIndex, pulse[i], timeDelay)


class Pulse(object):  # Pulse.data_list, Pulse.I_data, Pulse.Q_data
    def __init__(self, width, ssb_freq=0, iqscale=1, phase=0, skew_phase=0):
        self.vmax = 1.0                              # The max voltage that AWG is using
        self.width = width                           # How long the pulse is going to be. It is an integer number.
        self.ssb_freq = ssb_freq                     # The side band frequency, in order to get rid of the DC leakage from the mixer. Units: GHz.
        self.iqscale = iqscale                       # The voltage scale for different channels (i.e. the for I and Q signals). It is a floating point number.
        self.phase = phase / 180. * np.pi            # The phase difference between I and Q channels.
        self.skew_phase = skew_phase / 180. * np.pi
        self.Q_data = None                           # The I and Q data that will has the correction of IQ scale
        self.I_data = None                           # and phase. Both of them will be an array with floating number.
        xdataM = np.zeros(int(self.width + 20)) + 1.0
        xdataM[:10] = np.linspace(0, 1, 10)
        xdataM[-10:] = np.linspace(1, 0, 10)
        self.mark_data = xdataM

    def IQ_generator(self, data, amp):
        # This method is taking "raw pulse data" and then adding the correction of IQ scale and phase to it.
        # The input is an array of floating point number.
        # For example, if you are making a Gaussain pulse, this will be an array with number given by amp*exp(-((x-mu)/2*sigma)**2)
        # It generates self.Q_data and self.I_data which will be used to create waveform data in the .AWG file
        # For all the pulse that needs I and Q correction, the method needs to be called after doing in the data_generator after
        # you create the "raw pulse data"

        # Making I and Q correction
        tempx = np.arange(self.width)
        self.I_data = amp * data * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase)
        self.Q_data = amp * data * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale

    def DRAG_generator(self, data, amp, factor):
        tempx = np.arange(self.width)

        InPhase_shape = data
        Quadrature_shape = -np.gradient(data) * factor
        self.I_data = InPhase_shape * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase) + \
                      Quadrature_shape * np.sin(tempx * self.ssb_freq * 2. * np.pi + self.phase)  # noqa: E127
        self.Q_data = InPhase_shape * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale + \
                      Quadrature_shape * np.sin(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale  # noqa: E127

        max_dac = np.max([np.max(np.abs(self.I_data)), np.max(np.abs(self.Q_data))])
        if max_dac * amp > 1:
            raise TypeError("awg DAC>1")

        self.I_data = amp * self.I_data
        self.Q_data = amp * self.Q_data

    def anyPulse_generator(self, InData, QuData):
        self.width = len(InData)
        tempx = np.arange(len(InData))

        InPhase_shape = InData
        Quadrature_shape = QuData

        self.I_data = InPhase_shape * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase) + \
                      Quadrature_shape * np.sin(tempx * self.ssb_freq * 2. * np.pi + self.phase)  # noqa: E127
        self.Q_data = InPhase_shape * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale + \
                      Quadrature_shape * np.sin(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale  # noqa: E127


class combinePulse(Pulse):
    """ The pulse start point is always defined as the start of the first pulse in the list, all the time in the timelist
    should be defined relative to that time.
    """
    def __init__(self, pulseList, pulseTimeList):  # len(pulseTmeList) = len(pulseList) -1
        if len(pulseTimeList) != len(pulseList) - 1:
            raise TypeError("in valid time list definition, see class description")
        pulse_num = len(pulseList)
        pulseStartTimeList = np.append([0], pulseTimeList)
        pulseEndTimeList = [pulseStartTimeList[i] + pulseList[i].width - 1 for i in range(pulse_num)]

        self.width = np.max(pulseEndTimeList)+1
        self.I_data = np.zeros(self.width)
        self.Q_data = np.zeros(self.width)


        for i in range(pulse_num):
            pulse_ = pulseList[i]
            pad_front = pulseStartTimeList[i]
            pad_rear = self.width -pulseEndTimeList[i]-1
            padded_pulse_I = np.pad(pulse_.I_data, (pad_front, pad_rear), 'constant', constant_values=(0, 0))
            padded_pulse_Q = np.pad(pulse_.Q_data, (pad_front, pad_rear), 'constant', constant_values=(0, 0))
            self.I_data += padded_pulse_I
            self.Q_data += padded_pulse_Q


        max_dac = np.max([np.max(np.abs(self.I_data)), np.max(np.abs(self.Q_data))])
        if max_dac > 1:
            raise TypeError("awg DAC>1")

        self.marker = Marker(self.width + 20)
        xdataM = np.zeros(self.width + 20) + 1.0
        xdataM[:10] = np.linspace(0, 1, 10)
        xdataM[-10:] = np.linspace(1, 0, 10)
        self.mark_data = xdataM

class Zeros(Pulse):
    def __init__(self, width):
        super(Zeros, self).__init__(width, ssb_freq=0, iqscale=1, phase=0, skew_phase=0)
        self.Q_data = np.zeros(int(self.width))
        self.I_data = np.zeros(int(self.width))




class smoothBox(Pulse):
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase, height, ramp_slope, cut_factor=3, drag = 0):
        super(smoothBox, self).__init__(width, ssb_freq, iqscale, phase, skew_phase)
        x = np.arange(width)
        self.data_list = 0.5 * (np.tanh(ramp_slope * x - cut_factor) -
                                         np.tanh(ramp_slope * (x - width) + cut_factor))
        if self.data_list[len(self.data_list) // 2] < 0.9 * height:
            warnings.warn('wave peak is much shorter than desired amplitude')
        self.DRAG_generator(self.data_list, height, drag)


class hanning(Pulse):
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase, height, drag = 0):
        super(hanning, self).__init__(width, ssb_freq, iqscale, phase, skew_phase)
        x = np.arange(width)
        self.data_list = 1/2 * (1 - np.cos(np.pi / (width//2) * x))
        if self.data_list[len(self.data_list) // 2] < 0.9 * height:
            warnings.warn('wave peak is much shorter than desired amplitude')
        self.DRAG_generator(self.data_list, height, drag)


class Gaussian(Pulse):
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase, amp, deviation, drag=0):
        super(Gaussian, self).__init__(width, ssb_freq, iqscale, phase, skew_phase)
        self.data_list = signal.gaussian(width, deviation)
        self.DRAG_generator(self.data_list, amp, drag)


class Marker(Pulse):
    def __init__(self, width):
        super(Marker, self).__init__(width, 0, 1, 0, 0)
        x = np.zeros(width) + 1.0
        x[:5] = np.linspace(0, 1, 5)
        x[-5:] = np.linspace(1, 0, 5)
        self.data_list = x
        self.I_data = self.data_list
        self.Q_data = self.data_list


class MarkerOff(Pulse):
    def __init__(self, width):
        super(MarkerOff, self).__init__(width, 0, 1, 0, 0)
        x = np.zeros(width) + 0.01
        x[:5] = np.linspace(0, 1, 5)
        x[-5:] = np.linspace(1, 0, 5)
        self.data_list = x
        self.I_data = self.data_list
        self.Q_data = self.data_list


class gau():
    def __init__(self, gauCondition: dict):
        self.amp = gauCondition.get('amp', 0.1)
        self.ssbFreq = gauCondition.get('ssbFreq', 0.1)
        self.iqScale = gauCondition.get('iqScale', 1)
        self.phase = gauCondition.get('phase', 0)
        self.skewPhase = gauCondition.get('skewPhase', 0)
        self.sigma = gauCondition.get('sigma', 10)
        self.sigmaMulti = gauCondition.get('sigmaMulti', 6)
        self.dragFactor = gauCondition.get('dragFactor', 0)
        self.width = self.sigma * self.sigmaMulti

    def x(self):
        self.width = self.sigma * self.sigmaMulti
        self.x_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp, self.sigma, drag=self.dragFactor)
        return self.x_

    def x_theta(self, theta):
        self.width = self.sigma * self.sigmaMulti
        self.x_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp * theta / 180, self.sigma, drag=self.dragFactor)
        return self.x_

    def x_theta_phi(self, theta, phi):
        self.width = self.sigma * self.sigmaMulti
        self.x_ = Gaussian(self.width, self.ssbFreq, self.iqScale, phi, self.skewPhase, self.amp * theta / 180, self.sigma, drag=self.dragFactor)
        return self.x_

    def x2(self):
        self.width = self.sigma * self.sigmaMulti
        self.x2_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.x2_

    def x2N(self):
        self.width = self.sigma * self.sigmaMulti
        self.x2N_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 180, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.x2N_

    def y(self):
        self.width = self.sigma * self.sigmaMulti
        self.y_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 90, self.skewPhase, self.amp, self.sigma, drag=self.dragFactor)
        return self.y_

    def y2(self):
        self.width = self.sigma * self.sigmaMulti
        self.y2_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 90, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.y2_

    def y2N(self):
        self.width = self.sigma * self.sigmaMulti
        self.y2N_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase - 90, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.y2N_

    def off(self):
        self.width = self.sigma * self.sigmaMulti
        self.off_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, 0.000001 * self.amp, self.sigma, drag=self.dragFactor)
        return self.off_

    def marker(self):
        self.width = self.sigma * self.sigmaMulti
        self.marker_ = Marker(self.width + 20)
        return self.marker_

    def markerOff(self):
        self.width = self.sigma * self.sigmaMulti
        self.markerOff_ = MarkerOff(self.width + 20)
        return self.markerOff_


class box():
    def __init__(self, boxCondition: dict):
        self.amp = boxCondition.get('amp', 0.1)
        self.width = boxCondition.get('width', 200)
        self.ssbFreq = boxCondition.get('ssbFreq', 0.0)
        self.iqScale = boxCondition.get('iqScale', 1)
        self.phase = boxCondition.get('phase', 0)
        self.skewPhase = boxCondition.get('skewPhase', 0)
        self.rampSlope = boxCondition.get('rampSlope', 0.5)
        self.cutFactor = boxCondition.get('cutFactor', 3)
        self.dragFactor = boxCondition.get('dragFactor', 0)

    def smooth(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def hanning(self):
        self.smooth_ = hanning(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp, drag=self.dragFactor)
        return self.smooth_

    def smoothX(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def smoothY(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase + 90, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def smoothPhase(self, phi=0.0):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase + phi, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def smoothXN(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase + 180, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def smoothYN(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase - 90, self.skewPhase, self.amp,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def off(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp * 0.00000001,
                                 self.rampSlope, cut_factor=self.cutFactor, drag=self.dragFactor)
        return self.smooth_

    def marker(self):
        self.marker_ = Marker(self.width + 20)
        return self.marker_

    def markerOff(self):
        self.markerOff_ = MarkerOff(self.width + 20)
        return self.markerOff_


class Sin():
    def __init__(self, width, amp, freq, phase, smooth=False):
        phase = phase / 180. * np.pi
        x = np.arange(width)
        y = amp * np.sin(x / (1000. / freq) * 2.0 * np.pi + phase)
        data_out = y
        if smooth:
            data_out[0:11] = y[0:11] * np.exp(np.linspace(-5, 0, 11))
            data_out[-11:] = y[-11:] * np.exp(np.linspace(0, -5, 11))
            data_out[11:-11] = y[11:-11]
        self.width = width
        self.data_list = data_out
        self.I_data = amp * np.sin(x / (1000. / freq) * 2.0 * np.pi + phase)
        self.Q_data = amp * np.sin(x / (1000. / freq) * 2.0 * np.pi + phase + 0.5 * np.pi)

if __name__ == '__main__':
    # pulse0 = Zeros(0)
    # pulse1 = gau({}).x()
    # pulse3 = gau({}).x2()
    # pulse2 = box({}).smooth()
    # combo1 = combinePulse([pulse1,pulse2,pulse3],[pulse1.width, pulse2.width+pulse1.width])
    # combo2 = combinePulse([pulse0, pulse1,pulse2,pulse3],[0, pulse1.width, pulse2.width+pulse1.width])
    # plt.figure()
    # plt.plot(combo1.I_data)
    # plt.plot(combo1.Q_data)
    # plt.plot(combo2.I_data)
    # plt.plot(combo2.Q_data)

    # sb = smoothBox(20, 0.1, 1, 0, 90, 0.2, 0.5, 3)
    # plt.figure()
    # plt.plot(sb.I_data)
    Indata = np.array([1.35928789e-05, 2.97184560e-05, 6.27046077e-05, 1.27682415e-04,
                       2.50911829e-04, 4.75849487e-04, 8.70916447e-04, 1.53830268e-03,
                       2.62219824e-03, 4.31367740e-03, 6.84838589e-03, 1.04926987e-02,
                       1.55147435e-02, 2.21391231e-02, 3.04884011e-02, 4.05198009e-02,
                       5.19706769e-02, 6.43291574e-02, 7.68450276e-02, 8.85894687e-02,
                       9.85613951e-02, 1.05825421e-01, 1.09655795e-01, 1.09655795e-01,
                       1.05825421e-01, 9.85613951e-02, 8.85894687e-02, 7.68450276e-02,
                       6.43291574e-02, 5.19706769e-02, 4.05198009e-02, 3.04884011e-02,
                       2.21391231e-02, 1.55147435e-02, 1.04926987e-02, 6.84838589e-03,
                       4.31367740e-03, 2.62219824e-03, 1.53830268e-03, 8.70916447e-04,
                       4.75849487e-04, 2.50911829e-04, 1.27682415e-04, 6.27046077e-05,
                       2.97184560e-05, 1.35928789e-05, 6.00005278e-06, 2.55597801e-06,
                       1.05079397e-06, 4.16904375e-07, 1.59629719e-07, 5.89860705e-08,
                       2.10350525e-08, 7.23929297e-09, 2.40440224e-09, 7.70684302e-10,
                       2.38398941e-10, 7.11689482e-11, 2.05038394e-11, 5.70083216e-12,
                       1.52967700e-12, 3.96113494e-13, 9.89915114e-14, 2.38745224e-14,
                       5.55686502e-15, 1.24819782e-15, 2.70579826e-16, 5.66064375e-17,
                       1.14286426e-17, 2.22680352e-18, 4.18723722e-19, 7.59856618e-20,
                       1.33074288e-20, 2.24913267e-21, 3.66854975e-22, 5.77473649e-23,
                       8.77260040e-24, 1.28612430e-24, 1.81968391e-25, 2.48466239e-26,
                       3.27413994e-27, 4.16375824e-28, 5.11013287e-29, 6.05253483e-30,
                       6.91832279e-31, 7.63172589e-32, 8.12462124e-33, 8.34722083e-34,
                       8.27635480e-35, 7.91944444e-36, 7.31322189e-37, 6.51750245e-38,
                       5.60547053e-39, 4.65266050e-40, 3.72691143e-41, 2.88107920e-42,
                       2.14941236e-43, 1.54754293e-44, 1.07528624e-45, 7.21047520e-47,
                       4.66618664e-48])[:100]
    QuData = np.zeros(100)
    tempPulseObj = Pulse(0, 0.1, 1.05, 0, 0.2)
    testPulse = tempPulseObj.anyPulse_generator(Indata, QuData)