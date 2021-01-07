import numpy as np
import warnings
from scipy import signal
from sd1_api import keysightSD1

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
            module_ch_num = int(module.instrument.getOptions("channels")[-1])
            setattr(self, str(module_name), queueCollection(module_ch_num))
            if isinstance(module.instrument, keysightSD1.SD_AIN):
                self.dig_trig_num_dict[module_name] = {f"ch{i+1}" : 0 for i in range(module_ch_num)}
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
        getattr(module_, f'ch{channel}')[str(waveIndex)].append([pulse, timeDelay])

        if (module in self.dig_trig_num_dict.keys()) and msmt:
            self.dig_trig_num_dict[module][f"ch{channel}"] += 1

    def addTwoChan(self, AWG: str, channel: list, waveIndex: list, pulse: list, timeDelay: int):
        if len(channel) != 2:
            raise KeyError("channel number must be two!")
        for i in range(2):
            self.add(AWG, channel[i], waveIndex, pulse[i], timeDelay)


class Pulse(object):  # Pulse.data_list, Pulse.I_data, Pulse.Q_data
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase):
        self.vmax = 1.0                              # The max voltage that AWG is using
        self.width = width                           # How long the pulse is going to be. It is an integer number.
        self.ssb_freq = ssb_freq                     # The side band frequency, in order to get rid of the DC leakage from the mixer. Units: GHz.
        self.iqscale = iqscale                       # The voltage scale for different channels (i.e. the for I and Q signals). It is a floating point number.
        self.phase = phase / 180. * np.pi            # The phase difference between I and Q channels.
        self.skew_phase = skew_phase / 180. * np.pi
        self.Q_data = None                           # The I and Q data that will has the correction of IQ scale
        self.I_data = None                           # and phase. Both of them will be an array with floating number.
        xdataM = np.zeros(self.width // 2) + 1.0
        xdataM[:5] = np.linspace(0, 1, 5)
        xdataM[-5:] = np.linspace(1, 0, 5)
        self.mark_data = xdataM

    def iq_generator(self, data):
        # This method is taking "raw pulse data" and then adding the correction of IQ scale and phase to it.
        # The input is an array of floating point number.
        # For example, if you are making a Gaussain pulse, this will be an array with number given by exp(-((x-mu)/2*sigma)**2)
        # It generates self.Q_data and self.I_data which will be used to create waveform data in the .AWG file
        # For all the pulse that needs I and Q correction, the method needs to be called after doing in the data_generator after
        # you create the "raw pulse data"

        # Making I and Q correction
        tempx = np.arange(self.width)
        self.I_data = data * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase)
        self.Q_data = data * np.cos(tempx * self.ssb_freq * 2. * np.pi + self.phase + self.skew_phase) * self.iqscale

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

        self.width = np.max(pulseEndTimeList)
        self.I_data = np.zeros(self.width)
        self.Q_data = np.zeros(self.width)

        for t in range(self.width):
            for i in range(pulse_num):
                pulse_ = pulseList[i]
                if pulseStartTimeList[i] <= t and t <= pulseEndTimeList[i]:
                    self.I_data[t] += pulse_.I_data[t - pulseStartTimeList[i]]
                    self.Q_data[t] += pulse_.Q_data[t - pulseStartTimeList[i]]

        max_dac = np.max([np.max(np.abs(self.I_data)), np.max(np.abs(self.Q_data))])
        if max_dac > 1:
            raise TypeError("awg DAC>1")

        self.marker = Marker(self.width + 40)
        xdataM = np.zeros(self.width // 2) + 1.0
        xdataM[:5] = np.linspace(0, 1, 5)
        xdataM[-5:] = np.linspace(1, 0, 5)
        self.mark_data = xdataM


class smoothBox(Pulse):
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase, height, ramp_slope, cut_factor=3):
        super(smoothBox, self).__init__(width, ssb_freq, iqscale, phase, skew_phase)
        x = np.arange(width)
        self.data_list = 0.5 * height * (np.tanh(ramp_slope * x - cut_factor) -
                                         np.tanh(ramp_slope * (x - width) + cut_factor))
        if self.data_list[len(self.data_list) // 2] < 0.9 * height:
            warnings.warn('wave peak is much shorter than desired amplitude')
        self.iq_generator(self.data_list)


class Gaussian(Pulse):
    def __init__(self, width, ssb_freq, iqscale, phase, skew_phase, amp, deviation, drag=0):
        super(Gaussian, self).__init__(width, ssb_freq, iqscale, phase, skew_phase)
        self.data_list = signal.gaussian(width, deviation)
        self.DRAG_generator(self.data_list, amp, drag)


class Marker(Pulse):
    def __init__(self, width):
        super(Marker, self).__init__(width // 2, 0, 1, 0, 0)
        x = np.zeros(width // 2) + 1.0
        x[:5] = np.linspace(0, 1, 5)
        x[-5:] = np.linspace(1, 0, 5)
        self.data_list = x
        self.I_data = self.data_list
        self.Q_data = self.data_list


class MarkerOff(Pulse):
    def __init__(self, width):
        super(MarkerOff, self).__init__(width / 2, 0, 1, 0, 0)
        x = np.zeros(width // 2) + 0.01
        x[0] = 0.01
        x[1] = 0.003
        x[2] = 0.005
        x[-3] = 0.005
        x[-2] = 0.003
        x[-1] = 0.01
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
        self.x_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp, self.sigma, drag=self.dragFactor)
        return self.x_

    def x2(self):
        self.x2_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.x2_

    def x2N(self):
        self.x2N_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 180, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.x2N_

    def y(self):
        self.y_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 90, self.skewPhase, self.amp, self.sigma, drag=self.dragFactor)
        return self.y_

    def y2(self):
        self.y2_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase + 90, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.y2_

    def y2N(self):
        self.y2N_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase - 90, self.skewPhase, self.amp * 0.5, self.sigma, drag=self.dragFactor)
        return self.y2N_

    def off(self):
        self.off_ = Gaussian(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, 0.000001 * self.amp, self.sigma, drag=self.dragFactor)
        return self.off_

    def marker(self):
        self.marker_ = Marker(self.width + 20)
        return self.marker_

    def markerOff(self):
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

    def smooth(self):
        self.smooth_ = smoothBox(self.width, self.ssbFreq, self.iqScale, self.phase, self.skewPhase, self.amp, self.rampSlope, cut_factor=self.cutFactor)
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
