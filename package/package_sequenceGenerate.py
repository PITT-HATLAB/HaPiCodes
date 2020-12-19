# -*- coding: utf-8 -*-
"""
Created on 2019/05/15

@author: Pinlei Lu
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings


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
        super(Marker, self).__init__(width / 2, 0, 1, 0, 0)
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
        self.marker_ = Marker(self.width + 40)
        return self.marker_

    def markerOff(self):
        self.markerOff_ = MarkerOff(self.width + 40)
        return self.markerOff_

    def markerHalf(self):
        self.marker_ = Marker(self.width + 20)
        return self.marker_

    def markerOffHalf(self):
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
        self.marker_ = Marker(self.width + 40)
        return self.marker_

    def markerOff(self):
        self.markerOff_ = MarkerOff(self.width + 40)
        return self.markerOff_

    def markerHalf(self):
        self.marker_ = Marker(self.width + 20)
        return self.marker_

    def markerOffHalf(self):
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


class waveformModulesCollection(object):

    """queue collections for each module (AWG)

    chan{int}: dict. The queue(dictionary) for each channel in the module. Eg: chan1, chan2, chan3, chan4
    """

    def __init__(self, module_dict, chanNum=4):
        for module in module_dict.keys():
            setattr(self, str(module), {})
        self.module_dict = module_dict
        return

    def __dir__(self):
        return self.module_dict.keys()

class queueCollection(object):

    """queue collections for each module (AWG)

    chan{int}: dict. The queue(dictionary) for each channel in the module. Eg: chan1, chan2, chan3, chan4
    """

    def __init__(self, chanNum=4):
        for i in range(1, chanNum + 1):
            setattr(self, f'chan{i}', {})
        return


class queueModulesCollection(object):
    def __init__(self, module_dict):
        for module in module_dict.keys():
            setattr(self, str(module), queueCollection())
        return

    def add(self, AWG: str, channel: int, waveIndex: int, pulse, timeDelay: int):
        awg_ = getattr(self, AWG)
        try:
            q = getattr(awg_, f'chan{channel}')
        except AttributeError:
            raise AttributeError("Check the channel number of AWG module")
        if str(waveIndex) not in q.keys():
            getattr(awg_, f'chan{channel}')[str(waveIndex)] = []

        getattr(awg_, f'chan{channel}')[str(waveIndex)].append([pulse, timeDelay])

    def addTwoChan(self, AWG: str, channel: list, waveIndex: list, pulse: list, timeDelay: int):
        if len(channel) != 2:
            raise KeyError("channel number must be two!")
        for i in range(2):
            self.add(AWG, channel[i], waveIndex, pulse[i], timeDelay)


# class Seq(object):
#     def __init__(self):
#         return

#     def addPulse(self, index, pulse, time_, trigger):

#         return


# class AWGPulseSequence():
#     def __init__(self):
#         self.waveDict = {}
#         self.waveformList = []
#         self.sequeceList = []
#         return

#     def updateWaveDic(self, wave: str):
#         self.waveDict[]
#         return

#     def addIQ(self, index, channel, pulse, time_, trigger):
#         return


class SequenceEasy():
    """
    This class is easy mode of pulse generation, but make sure, no pulse
    superposition.
    """
    def __init__(self, num_shot, sequence_length):
        self.num_shot = num_shot  # this probably will be the number of trigger we need (TODO: update!)
        self.sequence_length = sequence_length
        self.temp_sequence = {}
        for i in range(8):
            self.temp_sequence[str(i + 1)] = []
        self.sequence_list = []  # the list put into the queue
        self.waveform_list = []  # the list of all different waveform(go to the RAM)
        self.length_of_list = int((num_shot) * sequence_length)
        self.shotNum = np.zeros(8, dtype=int)
        self.check = 0
        self.plot = 0

    def addPulse(self, pulse, time, channel, trigger=0):
        self.shotNum[channel - 1] += 1
        if channel in (1, 3):
            I_value = pulse.I_data
            Q_value = pulse.Q_data
            waveform_num = self.updateWaveform(I_value)
            self.temp_sequence[str(channel)].append((waveform_num, time, trigger))
            waveform_num = self.updateWaveform(Q_value)
            self.temp_sequence[str(channel + 1)].append((waveform_num, time, trigger))
            self.shotNum[channel] += 1
        elif channel in (2, 4):
            raise NameError("For the channel 2 or 4, it should be added automatically")
        elif channel in (5, 6, 7, 8):
            pulse_data = pulse.I_data
            waveform_num = self.updateWaveform(pulse_data)
            self.temp_sequence[str(channel)].append((waveform_num, time, trigger))
        else:
            raise NameError("Wrong channel")
        return

    def updateWaveform(self, waveform):
        count = 0
        for i in range(len(self.waveform_list)):
            if np.sum(waveform != self.waveform_list[i]):
                pass
            else:
                count += 1
                waveform_num = i
        if count == 0:
            self.waveform_list.append(waveform)
            waveform_num = len(self.waveform_list) - 1
        return waveform_num

    def checkFormat(self):
        if self.check == 0:
            for i in range(8):
                self.sequence_list.append(np.array(self.temp_sequence[str(i + 1)]))
            self.shotNum = self.shotNum / self.num_shot
            # Here define the trigger time in the HVI and return it to the PXIe
            qubitTrig = np.where(np.array(self.sequence_list[0])[:, 2] == 1)
            self.qubitTrigTime = np.array(self.sequence_list[0])[qubitTrig, 1]
            cavityTrig = np.where(np.array(self.sequence_list[2])[:, 2] == 1)
            self.cavityTrigTime = np.array(self.sequence_list[2])[cavityTrig, 1]
            qcDelay = (self.cavityTrigTime - self.qubitTrigTime).flatten()
            self.QC_interval_ini = qcDelay[0]
            self.QC_interval_max = qcDelay[-1]
            try:
                self.QC_interval_step = qcDelay[1] - qcDelay[0]
                if self.QC_interval_step == 0:
                    self.delayTimeList = qcDelay[0]
                else:
                    self.delayTimeList = np.arange(self.QC_interval_ini, self.QC_interval_max + self.QC_interval_step, self.QC_interval_step)
            except IndexError:
                self.QC_interval_step = 0
                self.delayTimeList = qcDelay[0]
            if self.QC_interval_max - self.QC_interval_ini != self.QC_interval_step * (len(qcDelay) - 1):
                raise NameError("The inteval is varing, we cannot process this kind of pulse")
            for i in range(4):
                self.sequence_list[i] = self.sequence_list[i] + [0, 65, 0]
            for j in range(2):
                self.sequence_list[j + 2][:, 1] = self.sequence_list[j + 2][:, 1] - self.delayTimeList
            self.sequence_list[5][:, 1] = self.sequence_list[5][:, 1] - self.delayTimeList
            if self.sequence_list[5][:, 1].any<0:
                raise TypeError('Wrong marker sequence definition')
        else:
            pass
        self.check += 1
        # print(self.QC_interval_ini, self.QC_interval_step, self.QC_interval_max)
        return self.QC_interval_ini, self.QC_interval_step, self.QC_interval_max


    def checkFormat_withPrepare(self, prepare_num=1):
        if self.check == 0:
            for i in range(8):
                self.sequence_list.append(np.array(self.temp_sequence[str(i + 1)]))
            self.shotNum = self.shotNum / self.num_shot
            # Here define the trigger time in the HVI and return it to the PXIe
            qubitTrig = np.where(np.array(self.sequence_list[0])[:, 2] == 1)[0][prepare_num :: prepare_num+1]
            self.qubitTrigTime = np.array(self.sequence_list[0])[qubitTrig, 1]

            cavityTrig = np.where(np.array(self.sequence_list[2])[:, 2] == 1)[0][prepare_num :: prepare_num+1]
            self.cavityTrigTime = np.array(self.sequence_list[2])[cavityTrig, 1]
            qcDelay = (self.cavityTrigTime - self.qubitTrigTime).flatten()
            self.QC_interval_ini = qcDelay[0]
            self.QC_interval_max = qcDelay[-1]
            try:
                self.QC_interval_step = qcDelay[1] - qcDelay[0]
                if self.QC_interval_step == 0:
                    self.delayTimeList = qcDelay[0]
                else:
                    self.delayTimeList = np.arange(self.QC_interval_ini, self.QC_interval_max + self.QC_interval_step, self.QC_interval_step)
            except IndexError:
                self.QC_interval_step = 0
                self.delayTimeList = qcDelay[0]
            if self.QC_interval_max - self.QC_interval_ini != self.QC_interval_step * (len(qcDelay) - 1):
                raise NameError("The inteval is varing, we cannot process this kind of pulse")
            for i in range(4):
                self.sequence_list[i] = self.sequence_list[i] + [0, 65, 0]
            for j in range(2):
                self.sequence_list[j + 2][prepare_num :: prepare_num+1, 1] = self.sequence_list[j + 2][prepare_num :: prepare_num+1, 1] - self.delayTimeList
            self.sequence_list[5][prepare_num :: prepare_num+1, 1] = self.sequence_list[5][prepare_num :: prepare_num+1, 1] - self.delayTimeList
            if self.sequence_list[5][prepare_num :: prepare_num+1, 1].any<0:
                raise TypeError('Wrong marker sequence definition')
            # for j in range(2):
            #     self.sequence_list[j + 2][:, 1] = self.sequence_list[j + 2][:, 1] - self.delayTimeList
            # self.sequence_list[5][:, 1] = self.sequence_list[5][:, 1] - self.delayTimeList
            # if self.sequence_list[5][:, 1].any<0:
            #     raise TypeError('Wrong marker sequence definition')
        else:
            pass
        self.check += 1
        # print(self.QC_interval_ini, self.QC_interval_step, self.QC_interval_max)
        return self.QC_interval_ini, self.QC_interval_step, self.QC_interval_max

    def showPulse(self, real=0):
        if self.plot == 0:
            self.sequencePlot = self.sequence_list
            if not real:
                for i in range(4):
                    self.sequencePlot[i] = self.sequence_list[i] - [0, 65, 0]
                for j in range(2):
                    self.sequencePlot[j + 2][:, 1] = self.sequence_list[j + 2][:, 1] + self.delayTimeList
                self.sequence_list[5][:, 1] = self.sequence_list[5][:, 1] + self.delayTimeList
        else:
            pass
        self.plot += 1
        xdata = np.arange(self.sequence_length)
        plt.figure(figsize=(15, 8))
        signalList = np.empty(4, dtype=object)
        markList = np.empty(4, dtype=object)

        for i in range(4):
            sdata = np.zeros(self.sequence_length)
            mdata = np.zeros(self.sequence_length)
            plt.subplot(4, 1, i + 1)
            plt.subplots_adjust(hspace=0)
            try:
                for j in range(self.shotNum[i]):
                    sigStart = self.sequencePlot[i][j][1]
                    sdata[sigStart:sigStart + len(self.waveform_list[self.sequencePlot[i][j][0]])] = self.waveform_list[self.sequencePlot[i][j][0]]
                signalList[i], = plt.plot(xdata, sdata, label='signal')

                for j in range(self.shotNum[i / 2 + 4]):
                    markStart = self.sequencePlot[i / 2 + 4][j][1]
                    mdata[markStart:markStart + 2 * len(self.waveform_list[self.sequencePlot[i / 2 + 4][j][0]])] = 1
                    mdata[0] = 0
                markList[i], = plt.plot(xdata, mdata, label='marker')
                plt.legend()
            except IndexError:
                raise
            plt.tick_params(axis='y', labelsize=10)
            plt.ylim(-1.2, 1.2)
        plt.xlabel('time / ns')

        def update(val):
            int_val = int(val)
            if (slider.val != int_val):
                slider.set_val(int_val)
            for i in range(4):
                sdata = np.zeros(self.sequence_length)
                for j in range(self.shotNum[i]):
                    sigStart = self.sequencePlot[i][self.shotNum[i] * int_val + j][1] - int_val * self.sequence_length
                    sdata[sigStart:sigStart + len(self.waveform_list[self.sequencePlot[i][self.shotNum[i] * int_val + j][0]])] = self.waveform_list[self.sequencePlot[i][self.shotNum[i] * int_val + j][0]]
                signalList[i].set_ydata(sdata)

                mdata = np.zeros(self.sequence_length)
                for j in range(self.shotNum[i / 2 + 4]):
                    markStart = self.sequencePlot[i / 2 + 4][self.shotNum[i / 2 + 4] * int_val + j][1] - int_val * self.sequence_length
                    mdata[markStart:markStart + 2 * len(self.waveform_list[self.sequencePlot[i / 2 + 4][self.shotNum[i] * int_val + j][0]])] = 1
                    mdata[0] = 0
                markList[i].set_ydata(mdata)

        slider_axes = plt.axes([0.2, 0.9, 0.65, 0.03])
        slider = Slider(slider_axes, "shot", 0, self.num_shot - 1, valinit=1)
        slider.on_changed(update)
        plt.title("Remerber, we always trigger at START, and this is for channel 1,2,3,4")
        plt.show()



if __name__ == '__main__':
    # pulse_ = doubleBox(40,40, 0, 1.0, 0, 0, 0.5,0.5, 0.5, cut_factor=3)
    # plt.plot(pulse_.I_data)
    # plt.plot(pulse_.Q_data)
    # plt.show()
    # pulse_ = smoothBox(50, 0, 1.4, 0, 0, 0.5, 0.5, cut_factor=3)
    # plt.plot(pulse_.I_data)
    # plt.plot(pulse_.Q_data)
    # plt.show()

    # condition_ = {'amp': 0.5,
    #               'ssbFreq': 0.1,
    #               'iqScale': 1,
    #               'phase': 0,
    #               'skewPhase': 0,
    #               'sigma': 100,
    #               'sigmaMulti': 6,
    #               'dragFactor': 0}

    # pulse_ = gau(condition_)
    # # pulse_.x2()
    # plt.plot(pulse_.x2().I_data)
    # plt.plot(pulse_.x2().I_data)


    condition_ = {'amp': 0.1,
                  'width': 200,
                  'ssbFreq': 0.0,
                  'iqScale': 1,
                  'phase': 0.0,
                  'skewPhase': 0.0,
                  'rampSlope': 0.5,
                  'cutFactor': 3}
    pulse_ = box(condition_)

    plt.plot(pulse_.smooth().I_data)
    pulse_.rampSlope = 2
    plt.plot(pulse_.smooth().I_data)

    plt.show()


    # print(seq1.sequence_list)

    # seq = SequenceEasy(21, 2000)
    # box = Square(500, 0, 1, 0, 0, 0.5)
    # mark_qubit = Square(110, 0, 1, 0, 0, 1)
    # mark_cav = Square(510, 0, 1, 0, 0, 1)
    # gau_x = Gaussian(90, 0.1, 1., 0, 0, 0.9, 15)
    # gau_y = Gaussian(90, 0.1, 1., 90, 0, 0.9, 15)
    # gau_x2 = Gaussian(90, 0.1, 1., 0, 0, 0.45, 15)
    # gau_y2 = Gaussian(90, 0.1, 1., 90, 0, 0.45, 15)
    # gau_id = Gaussian(90, 0.1, 1., 0, 0, 0.0, 15)
    # pulse_list = [gau_id, gau_id,
    #               gau_x, gau_x,
    #               gau_y, gau_y,
    #               gau_x, gau_y,
    #               gau_y, gau_x,  # First Group
    #               gau_x2, gau_id,
    #               gau_y2, gau_id,
    #               gau_x2, gau_y2,
    #               gau_y2, gau_x2,
    #               gau_x2, gau_y,
    #               gau_y2, gau_x,
    #               gau_x, gau_y2,
    #               gau_y, gau_x2,
    #               gau_x2, gau_x,
    #               gau_x, gau_x2,
    #               gau_y2, gau_y,
    #               gau_y, gau_y2,  # Second Group
    #               gau_x, gau_id,
    #               gau_y, gau_id,
    #               gau_x2, gau_x2,
    #               gau_y2, gau_y2]  # Thrid Group
    # time_point = 0
    #
    # for i in range(21):
    #     seq.addPulse(pulse_list[2 * i], time_point + 5, 1, trigger=1)
    #     seq.addPulse(pulse_list[2 * i + 1], time_point + 205, 1, trigger=0)
    #     seq.addPulse(mark_qubit, time_point, 5, trigger=1)
    #     seq.addPulse(mark_qubit, time_point + 200, 5, trigger=0)
    #     seq.addPulse(box, time_point + 5, 3, trigger=1)
    #     seq.addPulse(mark_cav, time_point + 0, 6, trigger=1)
    #     time_point += 2000
    # seq.checkFormat()
    # seq.showPulse()
