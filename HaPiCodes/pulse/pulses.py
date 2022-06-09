import warnings
import logging
from typing import Dict, Callable, List, Union
from inspect import getfullargspec
from functools import wraps
from copy import deepcopy

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def init_recorder(func: Callable):
    """A decorator that can automatically records the parameters used for a pulse class
    initializaiton. For inheritance case, only the highest layer parameters will be recorded.

    :param func: __init__ function of a class.
    :example:

    >>> class MyPulse:
    ...     @init_recorder
    ...     def __init__(self, width, ssbFreq=0.1, name='pulse1'):
    ...         pass
    >>> p = MyPulse(100, 0.05)
    >>> p.init_args
    {'width': 100, 'ssbFreq': 0.05, 'name': 'pulse1'}
    """

    names, varargs, keywords, defaults = getfullargspec(func)[:4]

    @wraps(func)
    def wrapper(self, *args, **kargs):
        if not hasattr(self, "init_args"):
            setattr(self, "init_args", {})

            for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
                getattr(self, "init_args")[name] = arg

            for name, default in zip(reversed(names), reversed(defaults)):
                if name not in getattr(self, "init_args"):
                    getattr(self, "init_args")[name] = default

        func(self, *args, **kargs)

    return wrapper


class SingleChannelPulse():
    def __init__(self, data: Union[List, np.ndarray], name: str = None,
                 channel: Dict[str, List] = None):
        """ Base class for pulses with only one channel.

        :param data:
        :param name: Name of the pulse.
        :param channel: Channel assigned for output. e.g. {"pulse": ["A1", 1]}
        """
        self.pulse_data = data,
        self.width = len(data)
        self.name = name
        self.channel = channel
        self.mark_data = None
    
    def clone(self, OMIT_NON_EXIST_PARAM=False, **newParams):
        """Clone the current pulse with updated parameters.

        :param newParams: kwargs for updated parameters, must match the parameter names
            in self.__init__
        :return: a copy of the current pulse with updated parameters
        """
        try:
            param_dict = deepcopy(self.init_args)
        except AttributeError:
            raise AttributeError(f"'init_args' not found for current pulse {self.__class__}. "
                                 f"To enable pulse cloning, the __init__ function must be decorated"
                                 f" by init_recoder. See built-in pulses for example")
        newParams_ = {}
        if ("width" in newParams) and ("markerWidth" not in newParams):
            warnings.warn("You forget to change markerWidth when changing pulseWidth")
        for param, val in newParams.items():
            if (param not in param_dict):
                if not OMIT_NON_EXIST_PARAM:
                    raise AttributeError(f"'{param}' not in initial parameters of {self.__class__}, "
                                         f"available params are {list(param_dict.keys())}")
            else:
                newParams_[param] = val

        param_dict.update(newParams_)
        pulse_ = self.__class__(**param_dict)
        return pulse_


class Marker(SingleChannelPulse):
    def __init__(self, width, risingWidth=10, fallingWidth=10, **kwargs):
        pulse_data = np.zeros(width) + 1.0
        pulse_data[:risingWidth] = np.linspace(0, 1, risingWidth)
        pulse_data[-fallingWidth:] = np.linspace(1, 0, fallingWidth)
        super(Marker, self).__init__(pulse_data, **kwargs)

class MarkerOff(SingleChannelPulse):
    def __init__(self, width, **kwargs):
        pulse_data = np.zeros(width) + 0.00001
        super(MarkerOff, self).__init__(pulse_data, **kwargs)


# -------------------------- pulses w/ I,Q and marker-----------------------------------------------
class Pulse():  # Pulse.data_list, Pulse.I_data, Pulse.Q_data, Pulse.mark_data
    def __init__(self, width: int, ssbFreq: float = 0, phase: float = 0, iqScale: float = 1,
                 skewPhase: float = 0, name: str = None, autoMarker=True,
                 channel: Dict[str, List] = None, **kwargs):
        """ Base class for pulses with I,Q and marker channel. The most commonly used case of Pulse.
        The pulse data can be accessed via Pulse.I_data, Pulse.Q_data and Pulse.mark_data.(unit ns)
        For AWGs with less than 1GSample/s, these data will be down sampled at uploading step.

        :param width: How long the pulse is going to be. Must be an integer number.
        :param ssbFreq: The side band frequency. Units: GHz.
        :param phase: Phase of the pulse for pulses with ssbFreq, in deg.
        :param iqScale: The voltage scale for different channels (i.e. the for I and Q signals).
        :param skewPhase: The phase_rad difference between I and Q channels. in deg.
        :param name: Name of the pulse.
        :param autoMarker: When True, the marker channel data will be automatically defined as a box
            pulse that is 20ns longer than the IQ data, with 10 ns rising and 10 ns falling.
        :param channel: channels assigned for the I,Q and marker output.
            e.g. {"I": ["A1", 1], "Q": ["A1", 2], "M": ["M1", 1]}
        """
        self.name = name
        self.width = int(width)
        self.ssbFreq = ssbFreq
        self.phase = phase
        self.iqScale = iqScale
        self.skewPhase = skewPhase
        self.phase_rad = phase / 180. * np.pi
        self.skewPhase_rad = skewPhase / 180. * np.pi
        if channel is None:
            self.channel: Dict[str, List] = {}
        else:
            self.channel = channel

        self.Q_data = None  # The I and Q data that will has the correction of IQ scale
        self.I_data = None  # and skewPhase. Both of them will be an array with floating number.
        self.mark_data = None

        if autoMarker:
            self.marker_generator(
                self.width)  # automatically generates a marker that is 20 ns longer than IQ data

    def plot(self, plotName=None):
        plt.figure(plotName)
        plt.title(self.name)
        plt.plot(self.I_data, label="I")
        plt.plot(self.Q_data, label="Q")
        plt.plot(self.mark_data, label="Marker")
        plt.legend()

    def fft(self, plotName=None):
        fourierTransform = np.fft.fft(self.I_data - 1j * self.Q_data)
        freq = np.fft.fftfreq(self.I_data.shape[-1])
        plt.figure(plotName)
        plt.title(self.name)
        plt.plot(freq, abs(fourierTransform))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')


    def marker_generator(self, width: int = None):
        width = self.width if width is None else width
        xdataM = np.zeros(int(width + 20)) + 1.0
        xdataM[:10] = np.linspace(0, 1, 10)
        xdataM[-10:] = np.linspace(1, 0, 10)
        self.mark_data = xdataM
        return xdataM

    def IQ_generator(self, data, amp):
        # This method is taking "raw pulse data" and then adding the correction of IQ scale and skewPhase to it.
        # The input is an array of floating point number.
        # For example, if you are making a Gaussain pulse, this will be an array with number given by amp*exp(-((x-mu)/2*sigma)**2)
        # It generates self.Q_data and self.I_data which will be used to create waveform data in the .AWG file
        # For all the pulse that needs I and Q correction, the method needs to be called after doing in the data_generator after
        # you create the "raw pulse data"

        # Making I and Q correction
        tempx = np.arange(self.width)
        self.I_data = amp * data * np.cos(tempx * self.ssbFreq * 2. * np.pi + self.phase_rad)
        self.Q_data = amp * data * np.cos(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad + self.skewPhase_rad) * self.iqScale
        self.amp = amp

    def DRAG_generator(self, data, amp, factor):
        tempx = np.arange(self.width)

        InPhase_shape = data
        Quadrature_shape = -np.gradient(data) * factor
        self.I_data = InPhase_shape * np.cos(tempx * self.ssbFreq * 2. * np.pi + self.phase_rad) + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad)  # noqa: E127
        self.Q_data = InPhase_shape * np.cos(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad + self.skewPhase_rad) * self.iqScale + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad + self.skewPhase_rad) * self.iqScale  # noqa: E127

        max_dac = np.max([np.max(np.abs(self.I_data)), np.max(np.abs(self.Q_data))])
        if max_dac * amp > 1:
            raise TypeError("awg DAC>1")

        self.I_data = amp * self.I_data
        self.Q_data = amp * self.Q_data

        self.amp = amp
        self.dragFactor = factor

    def anyPulse_generator(self, InData, QuData):
        if self.ssbFreq == 0:
            self.skewPhase = 90
        self.width = len(InData)
        tempx = np.arange(len(InData))
        InPhase_shape = InData
        Quadrature_shape = QuData

        self.I_data = InPhase_shape * np.cos(tempx * self.ssbFreq * 2. * np.pi + self.phase_rad) + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad)  # noqa: E127
        self.Q_data = InPhase_shape * np.cos(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad + self.skewPhase_rad) * self.iqScale + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + self.phase_rad + self.skewPhase_rad) * self.iqScale  # noqa: E127

    def generatorFromMagAndPhase(self, magData, phaseData, dragFactor):
        if self.ssbFreq == 0:
            self.skewPhase = 90
        self.width = np.max([len(magData), len(phaseData)])
        tempx = np.arange(self.width)
        InPhase_shape = magData
        try:
            Quadrature_shape = -np.gradient(magData) * dragFactor
        except ValueError:
            Quadrature_shape = magData * 0

        self.I_data = magData * np.cos(tempx * self.ssbFreq * 2. * np.pi + phaseData) + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + phaseData)  # noqa: E127
        self.Q_data = magData * np.cos(
            tempx * self.ssbFreq * 2. * np.pi + phaseData + self.skewPhase_rad) * self.iqScale + \
                      Quadrature_shape * np.sin(
            tempx * self.ssbFreq * 2. * np.pi + phaseData + self.skewPhase_rad) * self.iqScale  # noqa: E127

        try:
            max_dac = np.max([np.max(np.abs(self.I_data)), np.max(np.abs(self.Q_data))])
        except ValueError:
            max_dac = 0
        if max_dac > 1:
            raise TypeError("awg DAC>1")
        self.dragFactor = dragFactor


    def clone(self, OMIT_NON_EXIST_PARAM=False, **newParams):
        """Clone the current pulse with updated parameters.

        :param newParams: kwargs for updated parameters, must match the parameter names
            in self.__init__
        :return: a copy of the current pulse with updated parameters
        """
        try:
            param_dict = deepcopy(self.init_args)
        except AttributeError:
            raise AttributeError(f"'init_args' not found for current pulse {self.__class__}. "
                                 f"To enable pulse cloning, the __init__ function must be decorated"
                                 f" by init_recoder. See built-in pulses for example")
        newParams_ = {}
        if ("width" in newParams) and ("markerWidth" not in newParams):
            warnings.warn("You forget to change markerWidth when changing pulseWidth")
        for param, val in newParams.items():
            if (param not in param_dict):
                if not OMIT_NON_EXIST_PARAM:
                    raise AttributeError(f"'{param}' not in initial parameters of {self.__class__}, "
                                         f"available params are {list(param_dict.keys())}")
            else:
                newParams_[param] = val

        param_dict.update(newParams_)
        pulse_ = self.__class__(**param_dict)
        return pulse_


class Zeros(Pulse):
    @init_recorder
    def __init__(self, width: int, name: str = None, markerWidth=None):
        super(Zeros, self).__init__(width, name=name)
        self.Q_data = np.zeros(int(self.width))
        self.I_data = np.zeros(int(self.width))
        if markerWidth is not None:
            self.marker_generator(markerWidth - 20)

class SmoothBox(Pulse):
    @init_recorder
    def __init__(self, amp: float, width: int, rampSlope: float = 0.1, cutFactor: float = 3,
                 ssbFreq: float = 0, phase: float = 0, iqScale: float = 1, skewPhase: float = 0,
                 dragFactor: float = 0, markerWidth=None, **kwargs):
        super(SmoothBox, self).__init__(width, ssbFreq, phase, iqScale, skewPhase, **kwargs)
        x = np.arange(int(width))
        self.data_list = 0.5 * (np.tanh(rampSlope * x - cutFactor) -
                                np.tanh(rampSlope * (x - width) + cutFactor))
        if self.data_list[len(self.data_list) // 2] < 0.9 * amp:
            warnings.warn('wave peak is much shorter than desired amplitude')
        self.DRAG_generator(self.data_list, amp, dragFactor)
        if markerWidth is not None:
            self.marker_generator(markerWidth - 20)

class SmoothBox1Ch(SingleChannelPulse):
    @init_recorder
    def __init__(self, amp: float, width: int, rampSlope: float = 0.1, cutFactor: float = 3,
                 ssbFreq: float = 0, phase: float = 0, iqScale: float = 1, skewPhase: float = 0,
                 dragFactor: float = 0, markerWidth=None, **kwargs):
        pulse_ = SmoothBox(amp, width, rampSlope, cutFactor, ssbFreq, phase, 1, 90, dragFactor, markerWidth, **kwargs)

        self.pulse_data = pulse_.I_data
        self.width = len(self.pulse_data)
        if markerWidth is not None:
            self.marker_data = pulse_.marker_generator(markerWidth - 20)


class Hanning(Pulse):
    @init_recorder
    def __init__(self, amp, width, ssbFreq, phase, iqScale, skewPhase, drag=0, markerWidth=None, **kwargs):
        super(Hanning, self).__init__(width, ssbFreq, phase, iqScale, skewPhase, **kwargs)
        x = np.arange(int(width))
        self.data_list = 1 / 2 * (1 - np.cos(np.pi / (width // 2) * x))
        if self.data_list[len(self.data_list) // 2] < 0.9 * amp:
            warnings.warn('wave peak is much shorter than desired amplitude')
        self.DRAG_generator(self.data_list, amp, drag)
        if markerWidth is not None:
            self.marker_generator(markerWidth - 20)

class Gaussian(Pulse):
    @init_recorder
    def __init__(self, amp: float, sigma: int = 10, sigmaMulti: int = 6, ssbFreq: float = 0,
                 phase: float = 0, iqScale: float = 1, skewPhase: float = 0,
                 dragFactor: float = 0, markerWidth=None, **kwargs):
        """ Gaussian Pulse
        """
        width = int(sigma * sigmaMulti)
        super(Gaussian, self).__init__(width, ssbFreq, phase, iqScale, skewPhase, **kwargs)
        self.data_list = signal.gaussian(width, sigma)
        self.DRAG_generator(self.data_list, amp, dragFactor)
        if markerWidth is not None:
            self.marker_generator(markerWidth - 20)

class Gaussian1Ch(SingleChannelPulse):
    @init_recorder
    def __init__(self, amp: float, sigma: int = 10, sigmaMulti: int = 6, freq: float = 0,
                 phase: float = 0, iqScale: float = 1, skewPhase: float = 0,
                 dragFactor: float = 0, markerWidth=None, **kwargs):
        pulse_ = Gaussian(amp, sigma, sigmaMulti, freq, phase, 1, 90, dragFactor, markerWidth, **kwargs)
        
        self.pulse_data = pulse_.I_data
        self.width = len(self.pulse_data)
        if markerWidth is not None:
            self.marker_data = pulse_.marker_generator(markerWidth - 20)

class AWG(Pulse):
    def __init__(self, I_data: Union[List, np.ndarray], Q_data: Union[List, np.ndarray],
                 mark_data: Union[List, np.ndarray] = None, ssbFreq: float = 0, phase: float = 0,
                 iqScale: float = 1, skewPhase: float = 0, name: str = None,
                 channel: Dict[str, List] = None):
        self.width = np.max(len(I_data), len(Q_data))
        if mark_data is None:
            autoMarker = True
        else:
            autoMarker = False
        super(AWG, self).__init__(self.width, ssbFreq, phase, iqScale, skewPhase, name, autoMarker,
                                  channel)
        if autoMarker is False:
            self.mark_data = mark_data

        self.anyPulse_generator(I_data, Q_data)

class AWGfromMagAndPhase(Pulse):
    def __init__(self, mag_data: Union[List, np.ndarray], phase_data: Union[List, np.ndarray],
                 mark_length: int = None, ssbFreq: float = 0, iqScale: float = 1,
                 skewPhase: float = 0, dragFactor: float = 0, name: str = None,
                 channel: Dict[str, List] = None):
        
        self.width = np.max([len(mag_data), len(phase_data)])
        
        if mark_length is None:
            autoMarker = True
        else:
            autoMarker = False
        super(AWGfromMagAndPhase, self).__init__(self.width, ssbFreq, 0, iqScale, skewPhase, name, autoMarker,
                                  channel)
        if autoMarker is False:
            self.mark_data = self.marker_generator(mark_length - 20)

        self.generatorFromMagAndPhase(mag_data, phase_data, dragFactor)


def combinePulse(pulseList: List[Pulse], pulseTimeList, name: str = None, markerWidth=None) -> Pulse:
    """ A handy function that can combine multiple pulses to a single pulse.
    The pulse start point is always defined as the start of the first pulse in the list, all the
    time in the timelist should be defined relative to that time. Pulses happened at the samtime
    will be added on top of each other.
    """
    if len(pulseTimeList) != len(pulseList) - 1:
        raise TypeError("in valid time list definition, see class description")
    pulseTimeList = np.array(pulseTimeList, dtype=int)
    pulse_num = len(pulseList)
    pulseStartTimeList = np.append([0], pulseTimeList)
    pulseEndTimeList = [pulseStartTimeList[i] + pulseList[i].width - 1 for i in range(pulse_num)]

    width = np.max(pulseEndTimeList) + 1
    pulse_ = Pulse(width)
    pulse_.I_data = np.zeros(pulse_.width)
    pulse_.Q_data = np.zeros(pulse_.width)
    pulse_.name = name

    for i in range(pulse_num):
        p_ = pulseList[i]
        if p_.width >0 :
            pad_front = pulseStartTimeList[i]
            pad_rear = pulse_.width - pulseEndTimeList[i] - 1
            padded_pulse_I = np.pad(p_.I_data, (pad_front, pad_rear), 'constant',
                                    constant_values=(0, 0))
            padded_pulse_Q = np.pad(p_.Q_data, (pad_front, pad_rear), 'constant',
                                    constant_values=(0, 0))
            pulse_.I_data += padded_pulse_I
            pulse_.Q_data += padded_pulse_Q

    try:
        max_dac = np.max([np.max(np.abs(pulse_.I_data)), np.max(np.abs(pulse_.Q_data))])
        if max_dac > 1:
            raise TypeError("awg DAC>1")
    except ValueError:
        pass

    xdataM = np.zeros(pulse_.width + 20) + 1.0
    xdataM[:10] = np.linspace(0, 1, 10)
    xdataM[-10:] = np.linspace(1, 0, 10)
    pulse_.mark_data = xdataM

    if markerWidth is not None:
        pulse_.marker_generator(markerWidth - 20)

    return pulse_


# -------------------------- group pulses ----------------------------------------------------
class GroupPulse():
    def __init__(self, pulseClass: type(Pulse), width: int, ssbFreq: float = 0, iqScale: float = 1,
                 phase: float = 0, skewPhase: float = 0, name: str = None, markerWidth=None):
        """ Base group pulse class. Pulses in this group should be added using the add_pulse method

        :param pulseClass: class of the pulses in this pulse group.
        :param width: How long the pulse is going to be. Must be an integer number.
        :param ssbFreq: The side band frequency. Units: GHz.
        :param iqScale: The voltage scale for different channels (i.e. the for I and Q signals).
        :param phase: Phase of the pulse for pulses with ssbFreq, in deg
        :param skewPhase: The phase difference between I and Q channels. in deg
        :param name: Name of the pulse
        """
        self.pulseClass = pulseClass
        self.name = name
        self.width = int(width)
        self.ssbFreq = ssbFreq
        self.iqScale = iqScale
        self.phase = phase
        self.skewPhase = skewPhase


        self.channel: Dict[str, Dict[str, int]] = {}

        self.pulse_dict = {}

    def newPulse(self, **newParams):
        new_params = deepcopy(self.init_args)
        new_params.update(newParams)
        new_pulse = self.pulseClass(**new_params)
        return new_pulse

    def clone(self, OMIT_NON_EXIST_PARAM=False, **newParams):
        """Clone the current pulse group with updated parameters.

        :param newParams: kwargs for updated parameters, must match the parameter names
            in self.__init__
        :return: a copy of the current pulse with updated parameters
        """
        try:
            param_dict = deepcopy(self.init_args)
        except AttributeError:
            raise AttributeError(f"'init_args' not found for current pulse {self.__class__}. "
                                 f"To enable pulse cloning, the __init__ function must be decorated"
                                 f" by init_recoder. See built-in pulses for example")

        newParams_ = {}
        for param, val in newParams.items():
            if (param not in param_dict):
                if not OMIT_NON_EXIST_PARAM:
                    raise AttributeError(f"'{param}' not in initial parameters of {self.__class__}, "
                                         f"available params are {list(param_dict.keys())}")
            else:
                newParams_[param] = val

        param_dict.update(newParams_)
        pulse_ = self.__class__(**param_dict)

        return pulse_

    def add_pulse(self, name, pulse):
        if hasattr(self, name):
            raise AttributeError(f"{self.__class__} already has attribute {name}. rename the pulse")
        setattr(self, name, pulse)
        self.pulse_dict[name] = pulse


class GaussianGroup(GroupPulse):
    @init_recorder
    def __init__(self, amp: float, sigma: int = 10, sigmaMulti: int = 6, ssbFreq: float = 0,
                 iqScale: float = 1, phase: float = 0, skewPhase: float = 0,
                 dragFactor: float = 0, name: str = None, markerWidth=None, **kwargs):
        self.amp = amp
        self.ssbFreq = ssbFreq
        self.iqScale = iqScale
        self.phase = phase
        self.skewPhase = skewPhase
        self.sigma = sigma
        self.sigmaMulti = sigmaMulti
        self.dragFactor = dragFactor
        self.width = int(self.sigma * self.sigmaMulti)
        self.name = name
        self.kwargs = dict(kwargs)
        if ('DS', True) in self.kwargs.items():
            super().__init__(Gaussian1Ch, self.width, ssbFreq, iqScale, phase, skewPhase, name, markerWidth=markerWidth)
        else:
            super().__init__(Gaussian, self.width, ssbFreq, iqScale, phase, skewPhase, name, markerWidth=markerWidth)

        self.add_pulse("x", self.newPulse())
        self.add_pulse("x2", self.newPulse(amp=self.amp / 2))
        self.add_pulse("x3", self.newPulse(amp=self.amp / 3))
        self.add_pulse("x2N", self.newPulse(amp=self.amp / 2, phase=self.phase + 180))
        self.add_pulse("x3N", self.newPulse(amp=self.amp / 3, phase=self.phase + 180))
        self.add_pulse("y", self.newPulse(amp=self.amp, phase=self.phase + 90))
        self.add_pulse("y2", self.newPulse(amp=self.amp / 2, phase=self.phase + 90))
        self.add_pulse("y2N", self.newPulse(amp=self.amp / 2, phase=self.phase - 90))
        self.add_pulse("off", self.newPulse(amp=self.amp * 0.000001))


class BoxGroup(GroupPulse):
    @init_recorder
    def __init__(self, amp: float, width: int, rampSlope: float = 0.1, cutFactor: float = 3,
                 ssbFreq: float = 0, iqScale: float = 1, phase: float = 0, skewPhase: float = 0,
                 dragFactor: float = 0, name: str = None, **kwargs):
        self.amp = amp
        self.width = int(width)
        self.ssbFreq = ssbFreq
        self.iqScale = iqScale
        self.phase = phase
        self.skewPhase = skewPhase
        self.rampSlope = rampSlope
        self.cutFactor = cutFactor
        self.dragFactor = dragFactor
        self.name = name
        self.kwargs = dict(kwargs)
        if ('DS', True) in self.kwargs.items():
            super().__init__(SmoothBox1Ch, width, ssbFreq, iqScale, phase, skewPhase, name)
        else:
            super().__init__(SmoothBox, width, ssbFreq, iqScale, phase, skewPhase, name)

        self.add_pulse("smooth", self.newPulse())
        self.add_pulse("smoothX", self.newPulse())
        self.add_pulse("smoothY", self.newPulse(amp=self.amp, phase=self.phase + 90))
        self.add_pulse("smoothXN", self.newPulse(amp=self.amp, phase=self.phase + 180))
        self.add_pulse("smoothYN", self.newPulse(amp=self.amp, phase=self.phase + 270))
        self.add_pulse("off", self.newPulse(amp=self.amp * 0.000001))

class BoxGroupSubH(GroupPulse):
    @init_recorder
    def __init__(self, amp: float, width: int, rampSlope: float = 0.1, cutFactor: float = 3,
                 ssbFreq: float = 0, iqScale: float = 1, phase: float = 0, skewPhase: float = 0,
                 dragFactor: float = 0, name: str = None, **kwargs):
        self.amp = amp
        self.width = int(width)
        self.ssbFreq = ssbFreq
        self.iqScale = iqScale
        self.phase = phase
        self.skewPhase = skewPhase
        self.rampSlope = rampSlope
        self.cutFactor = cutFactor
        self.dragFactor = dragFactor
        self.name = name
        self.kwargs = dict(kwargs)
        super().__init__(SmoothBox, self.width, ssbFreq, iqScale, phase, skewPhase, name)

        self.add_pulse("smooth", self.newPulse())
        self.add_pulse("smoothX", self.newPulse())
        self.add_pulse("smoothY", self.newPulse(amp=self.amp, phase=self.phase - 90))
        self.add_pulse("smoothXN", self.newPulse(amp=self.amp, phase=self.phase + 180))
        self.add_pulse("smoothYN", self.newPulse(amp=self.amp, phase=self.phase + 90))
        self.add_pulse("off", self.newPulse(amp=self.amp * 0.000001))

class BoxGroupSubH(GroupPulse):
    @init_recorder
    def __init__(self, amp: float, width: int, rampSlope: float = 0.1, cutFactor: float = 3,
                 ssbFreq: float = 0, iqScale: float = 1, phase: float = 0, skewPhase: float = 0,
                 dragFactor: float = 0, name: str = None, **kwargs):
        self.amp = amp
        self.width = int(width)
        self.ssbFreq = ssbFreq
        self.iqScale = iqScale
        self.phase = phase
        self.skewPhase = skewPhase
        self.rampSlope = rampSlope
        self.cutFactor = cutFactor
        self.dragFactor = dragFactor
        self.name = name
        self.kwargs = dict(kwargs)
        super().__init__(SmoothBox, self.width, ssbFreq, iqScale, phase, skewPhase, name)

        self.add_pulse("smooth", self.newPulse())
        self.add_pulse("smoothX", self.newPulse())
        self.add_pulse("smoothY", self.newPulse(amp=self.amp, phase=self.phase - 90))
        self.add_pulse("smoothXN", self.newPulse(amp=self.amp, phase=self.phase + 180))
        self.add_pulse("smoothYN", self.newPulse(amp=self.amp, phase=self.phase + 90))
        self.add_pulse("off", self.newPulse(amp=self.amp * 0.000001))
        
        
if __name__ == '__main__':
    box_group = BoxGroup(1, 200, rampSlope=0.5)
    # gau_group.x.fft("test")
    # gau_group2 = gau_group.clone(amp=0.5, sigma=100)
    # gau_group2.x.fft("test")


    plt.figure()
    plt.plot(box_group.smoothX.I_data)
    box_group = BoxGroup(1, 12)
    plt.plot(box_group.smoothX.I_data)
    # plt.plot(gau_group2.x.I_data)
