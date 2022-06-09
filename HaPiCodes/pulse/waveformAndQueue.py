import warnings
import logging
from typing import Dict, List, Union

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from HaPiCodes.pulse import pulses as ps
from HaPiCodes.pulse.pulses import Pulse, GroupPulse, SingleChannelPulse

PULSE_TYPE = Union[Pulse, GroupPulse, SingleChannelPulse]


def constructPulseDictFromYAML(pulseParams: Dict[str, Dict],
                               userPulsePackage=None) -> Dict[str, Pulse]:
    """Generate pulses defined in the pulseParams in YAML file, and put the pulses
    in a dictionary. This function will first try to find the pulse shape in the
    built-in pulse shape package (HaPiCodes.pulse.pulses), if not found, the
    function will try to find pulse shape in userPulsePackage.
    For group pulses, each pulse in GroupPulse.pulse_dict will be added individually.

    :param pulseParams: Dict[pulse_name: pule_parameters]
    :param userPulsePackage: package that contains the user defined pulses
    :return: Dict[pulse_name, Pulse]
    """
    pulseDict = {}
    for name, pulse_def in pulseParams.items():
        pulse_shape = pulse_def.get("shape")
        try:
            pulse_class = getattr(ps, pulse_shape)
        except AttributeError:
            try:
                pulse_class = getattr(userPulsePackage, pulse_shape)
            except AttributeError:
                raise NameError(f"Can't find pulse '{pulse_shape}' in either built-in pulse library"
                                f" or user pulse library {userPulsePackage}")
        pulse_ = pulse_class(**{k: v for k, v in pulse_def.items() if k != "shape"})
        if isinstance(pulse_, GroupPulse):
            for n_, p_ in pulse_.pulse_dict.items():
                pulseDict[f"{name}.{n_}"] = p_
        elif isinstance(pulse_, Pulse) or isinstance(pulse_, SingleChannelPulse):
            pulseDict[name] = pulse_

    return pulseDict


class modulesWaveformCollection(object):
    """ waveForm collection for all modules, for uploading to HVI
    """

    def __init__(self, module_dict):
        for module in module_dict.keys():
            setattr(self, str(module), {})
            # The following code is actually designed for digitizer. (Because we generally won't upload
            # following parts into waveformCollection which will result in failed unloading queue)
            getattr(self, module)['trigger.dig'] = []
            getattr(self, module)['trigger.fpga4'] = []
            getattr(self, module)['trigger.fpga5'] = []
            getattr(self, module)['trigger.fpga6'] = []
            getattr(self, module)['trigger.fpga7'] = []
        self.module_dict = module_dict
        self.waveInfo = {}
        return

    def __dir__(self):
        return self.module_dict.keys()


class queueCollection(object):
    """queue collections for a single module (AWG)
    ch{int}: dict. The queue(dictionary) for each channel in the module. Eg: ch1, ch2.
    """

    def __init__(self, chanNum=4):
        for i in range(1, chanNum + 1):
            setattr(self, f'ch{i}', {})
        return


class modulesQueueCollection(object):
    """ queue collection for all modules, for uploading to HVI
    """

    def __init__(self, module_dict):
        self.dig_trig_num_dict = {}
        for module_name, module in module_dict.items():
            try:
                from HaPiCodes.sd1_api import keysightSD1
                module_ch_num = int(module.instrument.getOptions("channels")[-1])
                if isinstance(module.instrument, keysightSD1.SD_AIN):
                    self.dig_trig_num_dict[module_name] = {f"ch{i + 1}": 0 for i in
                                                           range(module_ch_num)}
            except (AttributeError, OSError) as e:
                module_ch_num = 4  # dummy module has 4 channels by default

            setattr(self, str(module_name), queueCollection(module_ch_num))
        self.module_dict = module_dict
        self.maxIndexNum = 0

        return

    def add(self, module: str, channel: int, waveIndex: int, pulse, timeDelay: int,
            msmt: bool = False):
        module_ = getattr(self, module)
        try:
            q = getattr(module_, f'ch{channel}')
        except AttributeError:
            raise AttributeError("Check the channel number of AWG module")
        if str(waveIndex) not in q.keys():
            getattr(module_, f'ch{channel}')[str(waveIndex)] = []
            self.maxIndexNum = np.max([int(waveIndex), self.maxIndexNum])
        timeDelay = np.ceil(timeDelay / 10) * 10
        getattr(module_, f'ch{channel}')[str(waveIndex)].append([pulse, timeDelay])

        if (module in self.dig_trig_num_dict.keys()) and msmt:
            self.dig_trig_num_dict[module][f"ch{channel}"] += 1

    def addTwoChan(self, AWG: str, channel: list, waveIndex: int, pulse: list, timeDelay: int):
        if len(channel) != 2:
            raise KeyError("channel number must be two!")
        for i in range(2):
            self.add(AWG, channel[i], waveIndex, pulse[i], timeDelay)


class Waveforms:
    def __init__(self, pulseDict: Dict[str, Pulse] = None):
        """ collects all the waveforms defined in the experiment sequence. Not all of the pulses are
        necessarily uploaded to the AWG. Calling an instance of this class will return a dictionary
        that contains all the pulses.

        :param pulseDict: a dictionary that contains some initial pulses.
        """
        self.pulseDict = {} if pulseDict is None else pulseDict

    def addPulse(self, name: str, pulse: PULSE_TYPE):
        """ add a pulse to the waveform collection

        :param name: name of the pulse
        :param pulse: pulse object
        :return: name of the added pulse
        """
        # check for duplicate name
        if name in self.pulseDict:
            raise NameError(f"duplicate pulse name '{name}'")
        self.pulseDict[name] = pulse
        return name

    def cloneAddPulse(self, oldPulseName: str, newPulseName: str, OMIT_NON_EXIST_PARAM=False, **newParams):
        """ Clone an existing pulse in the collection with updated parameters, and add the new pulse
        to the collection.

        :param oldPulseName: name of the pulse to clone from
        :param newPulseName: name of the new pulse
        :param newParams: parameters to update, must be a parameter that the old pulse has.
        :return: name of the new pulse.
        """
        newParams["name"] = newPulseName
        # clone, update amd add the new pulse
        pulse_ = self.pulseDict[oldPulseName].clone(OMIT_NON_EXIST_PARAM, **newParams)
        self.addPulse(newPulseName, pulse_)
        return newPulseName

    def __call__(self):
        return self.pulseDict


class Queue:
    def __init__(self, module_dict: Dict = None):
        """ This is a class that contains all the queued waveforms. This class works as a mediator
        that makes it easy for user to check the queued waveform, as well as automatically generates
        the 'modulesWaveformCollection' and 'modulesQueueCollection' classes, which will be
        uploaded to pathwave HVI for lower layer waveform and queue configuration.

        :param module_dict: dictionary that contains all the modules
        """
        module_dict = {} if module_dict is None else module_dict
        # parameters for communicating with pathwave HVI
        self._MW = modulesWaveformCollection(module_dict)
        self._MQ = modulesQueueCollection(module_dict)

    def updateW(self, module: str, pulseName: str, pulse: Union[List, np.ndarray]):
        """ update the modulesWaveformCollection used for communicating with HVI. Pulses already in
        in the collection will not be uploaded.

        :param module: name of the module that the pulse is going to upload to
        :param pulseName: name of the pulse
        :param pulse: pulse data to upload
        :return:
        """
        pulseName = "pulse." + pulseName
        if pulseName in getattr(self._MW, module).keys():
            pass
        else:
            getattr(self._MW, module)[pulseName] = pulse

    def updateQ(self, module: str, channel: int, index: int, pulseName: str, time_: int):
        """update the modulesQueueCollection used for communicating with HVI.

        :param module: name of the module that the pulse is going to upload to
        :param channel: channel of the module that the pulse is going to upload to
        :param index: index of the pulse in whole experiment sequence
        :param pulseName: name of the pulse to queue
        :param time_: time of the pulse in one pulse sequence, in ns. It is strongly recommended
            that this time is multiples of 10 ns.
        :return:
        """
        pulseName = "pulse." + pulseName
        if pulseName not in getattr(self._MW, module):
            raise NameError(pulseName + " is not initialize in W yet", module)
        self._MQ.add(module, channel, index, pulseName, time_)

    def updateWforIQM(self, pulseName: str, pulse: PULSE_TYPE, IQMChannel: Dict[str, List],
                      Mupdate=True):
        """ a short hand function to update the modulesWaveformCollection for pulses with I, Q and
        marker channel.

        :param pulseName: name of the pulse
        :param pulse: pulse to upload
        :param IQMChannel: channels assigned for the I,Q and Marker output.
            e.g. {"I": ["A1", 1], "Q": ["A1", 2], "M": ["M1", 1]}
        :param Mupdate: When False, the marker channel will not be updated
        :return:
        """
        self.updateW(IQMChannel['I'][0], pulseName + '.I', pulse.I_data)
        self.updateW(IQMChannel['Q'][0], pulseName + '.Q', pulse.Q_data)
        if Mupdate:
            self.updateW(IQMChannel['M'][0], pulseName + '.M', pulse.mark_data)

    def updateQforIQM(self, pulseName: str, index: int, time_: int, IQMChannel: Dict[str, List],
                      Mupdate=True, pulseMarkerDelay: int = 10):
        """ a short hand function to update the modulesQueueCollection for pulses with I, Q and
        marker channel.

        :param pulseName: name of the pulse
        :param index: index of the pulse in whole experiment sequence
        :param time_: time of the pulse in one pulse sequence, in ns. It is strongly recommended
            that this time is multiples of 10 ns.
        :param IQMChannel: channels assigned for the I,Q and Marker output.
            e.g. {"I": ["A1", 1], "Q": ["A1", 2], "M": ["M1", 1]}
        :param Mupdate: When False, the marker channel will not be updated
        :param pulseMarkerDelay: delay time between the pulse and marker. marker will be triggered
            pulseMarkerDelay ns before the pulse
        :return:
        """
        self.updateQ(IQMChannel['I'][0], IQMChannel['I'][1], index, pulseName + '.I', time_)
        self.updateQ(IQMChannel['Q'][0], IQMChannel['Q'][1], index, pulseName + '.Q', time_)
        if Mupdate:
            self.updateQ(IQMChannel['M'][0], IQMChannel['M'][1], index, pulseName + '.M',
                         time_ - pulseMarkerDelay)


class ExperimentSequence():
    def __init__(self, module_dict: Dict, msmtInfoDict: Dict, subbuffer_used=False):
        """ Base class for experiment sequences. This class automatically acquires the basic
        experiment information from the msmtInfoDict, and provide functions for writing experiment
        pulse/trigger sequences. Users can define their own collection of experiment sequences by
        creating a child class that inherits this class, and use params/functions provided in this
        class to write their own sequences.

        :param module_dict: dictionary that contains all the modules. e.g. {"A1": inst1}. At this
            step (as long as the pulse is not uploaded to a real instrument), the values(inst1, etc)
             in this this dictionary can be any object. So it is easy to write a dummy module_dict
            and check the sequence without involving any real instrument.
        :param msmtInfoDict: dictionary that contains the measurement information. Usually loaded
            from YAML file.
        :param subbuffer_used: Whether subbuffer is used in the digitizer FPGA. This is used for
            determining which kind of digital trigger will be sent to the digitizer.
        """
        self.module_dict = module_dict
        self.subbuffer_used = subbuffer_used
        self.info = msmtInfoDict
        self.pulseMarkerDelay = self.info['regularMsmtPulseInfo']['pulseMarkerDelay']
        self.digMsmtDelay = self.info['regularMsmtPulseInfo']['digMsmtDelay']
        self.msmtLeakOutTime = self.info['regularMsmtPulseInfo']['msmtLeakOutTime']

        # get all the pulses defined in msmtInfoDict.yaml
        self.pulse_dict = constructPulseDictFromYAML(msmtInfoDict["pulseParams"])
        self.W = Waveforms(self.pulse_dict)
        self.Q = Queue(module_dict)

        # get all the channels defined in msmtInfoDict.yaml
        self.channel_dict = {}
        for ch_name, chs in self.info['combinedChannelUsage'].items():
            self.channel_dict[ch_name] = chs

        # only for preview pulse dict
        self.queue_dict = {}
        self.numOfIndex = 1
        self.maxTime = 0

        try:
            self.phaseCorr_dict={}
            if isinstance(self.info['sampleNames'], list):
                for i, name in enumerate(self.info['sampleNames']):
                    self.phaseCorr_dict[name] = {}
            elif isinstance(self.info['sampleNames'], str):
                name = self.info['sampleNames']
                self.phaseCorr_dict[name] = {}
        except KeyError:
            pass

    def _updataQueueDict(self, pulseName: str, index: int, pulseTime: int, channelName: str):

        if channelName not in self.queue_dict.keys():
            self.queue_dict[channelName] = {}

        if index not in self.queue_dict[channelName].keys():
            self.queue_dict[channelName][index] = []
        
        self.queue_dict[channelName][index].append([pulseTime, pulseName])
        self.numOfIndex = max([self.numOfIndex, index+1])
        pulseWidth_ = self.W()[pulseName].width if pulseName in self.W() else 0
        pulseEndTime = pulseTime + pulseWidth_
        self.maxTime = max([self.maxTime, pulseEndTime])

    def queuePulse(self, pulseName: str, index: int, pulseTime: int, channel: Union[str, Dict],
                   omitMarker=False, fillTriggerGap=False):
        """ function to queue a pulse in the experiment sequence.

        :param pulseName: name of the pulse
        :param index: index of the pulse in whole experiment sequence
        :param pulseTime: time of the pulse in one pulse sequence, in ns. It is strongly recommended
            that this time is multiples of 10 ns.
        :param channel: str: channel names you have defined in the yaml file.
                        dict: channels assigned for the pulse output.
                                e.g. {"I": ["A1", 1], "Q": ["A1", 2], "M": ["M1", 1]}
        :param channel: channel names you have defined in the yaml file
        :param omitMarker: When True, the marker channel will not be updated. This is designed for
            the case when multiple pulses share the same marker channel, where only one of these
            pulses should have omitMarker=False.
        :return: time point after the pulse is finished
        """

        try:
            pulse_ = self.W()[pulseName]
        except KeyError:
            raise KeyError(f"pulse {pulseName} is not defined. Use self.W.addPulse/cloneAddPulse")
 
        if isinstance(channel, str):
            channelName = channel
            channel = self.channel_dict[channel]
        else:
            channelName = ""
            for ch in channel.values():
                channelName += ch[0] + "_" + str(ch[1]) + "_*_"

        self._updataQueueDict(pulseName, index, pulseTime, channelName)

        if isinstance(pulse_, Pulse):
            self.Q.updateWforIQM(pulseName, pulse_, channel, not omitMarker)
            self.Q.updateQforIQM(pulseName, index, pulseTime, channel, not omitMarker,
                                 self.pulseMarkerDelay)
            self.W()[pulseName].channel = channel
        elif isinstance(pulse_, SingleChannelPulse):
            pulse_module_ = list(channel.values())[0][0]
            pulse_channel_ = list(channel.values())[0][1]
            self.Q.updateW(pulse_module_, pulseName, pulse_.pulse_data)
            self.Q.updateQ(pulse_module_, pulse_channel_, index, pulseName, pulseTime)
            self.W()[pulseName].channel = channel
        if fillTriggerGap:
            return np.ceil((pulse_.width + pulseTime)/10)*10
        else:
            return pulse_.width + pulseTime

    def addDigTrigger(self, index: int, time_: int, DigChannel: Union[str, Dict]):
        """ add a digitizer trigger in the queue for taking data

        :param index: index of the digitizer trigger in whole experiment sequence
        :param time_: time of the trigger in one pulse sequence, in ns. It is strongly recommended
            that this time is multiples of 20 ns.
        :param DigChannel: str: digitizer channel names you have defined in the yaml file.
                           dict:  digitizer channel names you have defined in the yaml file.
                                    e.g., {"Sig": ["D1", 1], "Ref": ["D1", 2]}, {"Sig": ["D1", 1]}
        :return:
        """
        
        if isinstance(DigChannel, str):
            DigChannelName = DigChannel
            DigChannel = self.channel_dict[DigChannel]
        else:
            DigChannelName = ""
            for ch in DigChannel.values():
                DigChannelName += ch[0] + "_" + str(ch[1]) + "_*_"

        self._updataQueueDict("DigTrigger", index, time_, DigChannelName)

        for (mod_, ch_) in DigChannel.values():
            fgpaTriggerCh = 'trigger.fpga' + str(3 + ch_)
            if not self.subbuffer_used:
                self.Q._MQ.add(mod_, ch_, index, 'trigger.dig', time_, msmt=True)
                self.Q._MQ.add(mod_, ch_, index, fgpaTriggerCh, time_, msmt=False)
            else:
                self.Q._MQ.add(mod_, ch_, index, fgpaTriggerCh, time_, msmt=True)

    def addMsmt(self, CavDrivePulseName: str, index: int, time_: int,
                CavDriveChannel: Union[str, Dict], DigChannel: Union[str, Dict]):
        """ add a drive pulse to the readout resonator (cavity ) and trigger the digitizer to take
        the measurement result.

        :param CavDrivePulseName: pulse name for cavity drive
        :param index: index of the msmt in whole experiment sequence
        :param time_: time of the msmt in one pulse sequence, in ns. It is strongly recommended
            that this time is multiples of 10 ns.
        :param CavDriveChannel: str: cavity drive channel name you have defined in the yaml file.
                                dict: CavDriveChannel: AWG channel to output cavity drive
                                        e.g., {"I": ["A1", 1], "Q": ["A1", 2], "M": ["M1", 1]}
        :param DigChannel: str: digitizer channel name you have defined in the yaml file.
                           dict: digitizer channel to take data.
                                    e.g., {"Sig": ["D1", 1], "Ref": ["D1", 2]}, {"Sig": ["D1", 1]}
        :return:
        """

        self.queuePulse(CavDrivePulseName, index, time_, CavDriveChannel)

        if time_ - self.digMsmtDelay < 0:
            raise ValueError(
                f"Cavity drive time for MSMT must be later than digMsmtDelay ({self.digMsmtDelay})")
        self.addDigTrigger(index, time_ - self.digMsmtDelay, DigChannel)
        return self.msmtLeakOutTime + time_

    def plot(self, plotType='word') -> object:
        self.maxTime = int(self.maxTime)
        self.numOfChannel = len(self.queue_dict.keys())
        for channel, index_dict in self.queue_dict.items():
            for i in range(self.numOfIndex):
                if i not in index_dict.keys():
                    self.queue_dict[channel][i] = []


        indexSlider = 0
        if plotType == 'word' or 0:
            import matplotlib.cm as cm
            from matplotlib.widgets import Slider, Button
            colors_ = cm.rainbow(np.linspace(0, 1, self.numOfChannel))
            plt.figure(figsize=(10, self.numOfChannel + 1))
            for i in range(self.numOfChannel):
                plt.axhline(i, 0, self.maxTime, color=colors_[i])
            plt.yticks(np.arange(self.numOfChannel), list(self.queue_dict.keys()))
            plt.xlim(0 - self.maxTime/10, self.maxTime + self.maxTime/10)
            plt.ylim(-0.5, self.numOfChannel+0.1)
            channelNumYaxis = 0

            text_dict = {}
            for channel, index_dict in self.queue_dict.items():
                maxNumPulse = 0
                for i in range(self.numOfIndex):
                    maxNumPulse = max([maxNumPulse, len(index_dict[i])])
                height = channelNumYaxis - 0.3
                # height += 1 / (len(index_dict[0])+1)
                text_dict[channel] = []
                for time, pulse in index_dict[0]:
                    text_dict[channel].append(plt.text(time, height, str(time) + ": " + pulse, ha='center', rotation=30))
                    # height += 1 / (len(index_dict[0])+1)
                for k in range(maxNumPulse - len(index_dict[0]) + 1):
                    text_dict[channel].append(plt.text(time, height, "", ha='center', rotation=30))
                channelNumYaxis += 1
            plt.subplots_adjust(bottom=0.25)
            axPos= plt.axes([0.15, 0.1, 0.7, 0.04], facecolor='lightgoldenrodyellow')
            indexSlider = Slider(ax=axPos, label='Index', valmin=0.0, valmax=self.numOfIndex, valstep=1)

            def indexUpdate(index_):
                channelNumYaxis = 0
                for channel, index_dict in self.queue_dict.items():
                    iPulse = 0
                    height = channelNumYaxis - 0.3
                    # height += 1 / (len(index_dict[index_])+1)
                    for time, pulse in index_dict[index_]:
                        text_dict[channel][iPulse].set_position((time, height))
                        text_dict[channel][iPulse].set_text(str(time) + ": " + pulse)
                        iPulse += 1
                        # height += 1 / (len(index_dict[index_])+1)                    
                    for restText in range(len(text_dict[channel]) - iPulse):
                        text_dict[channel][iPulse + restText].set_text("")

                    channelNumYaxis += 1

            indexSlider.on_changed(indexUpdate)

        elif plotType=='realPulse' or 1:
            import matplotlib.cm as cm
            from matplotlib.widgets import Slider, Button
            colors_ = cm.rainbow(np.linspace(0, 1, self.numOfChannel * 3))
            fig = plt.figure(figsize=(10, self.numOfChannel + 1))
            ax = fig.add_subplot(111)
            plt.yticks(np.arange(self.numOfChannel) * 2.5, list(self.queue_dict.keys()))
            
            finalTime = 0
            channelNumYaxis = 0
            line_dict = {}
            for channel, index_dict in self.queue_dict.items():
                line_dict[channel] = {}
                height = channelNumYaxis
                timeList = np.arange(0, self.maxTime + self.msmtLeakOutTime, 1)
                IdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                QdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                MdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                for time, pulse in index_dict[0]:
                    time = int(time)
                    if pulse == 'DigTrigger':
                        arrow = plt.arrow(time, channelNumYaxis + 2, 0, -0.5, head_length=1.5, width=5, ec='k', fc='k')
                        line_dict[channel][f'arrow_{time}'] = arrow
                    else:
                        pulseClass = self.W()[pulse]
                        pulseLength = pulseClass.width
                        IdataList[time : time + pulseLength] = pulseClass.I_data
                        QdataList[time : time + pulseLength] = pulseClass.Q_data
                        MdataList[time - self.pulseMarkerDelay : time - self.pulseMarkerDelay + len(pulseClass.mark_data)] = pulseClass.mark_data
                        finalTime = max(finalTime, time + pulseLength)
                lineI, = plt.plot(timeList, IdataList + channelNumYaxis, color=colors_[int(channelNumYaxis//2.5) * 3])
                lineQ, = plt.plot(timeList, QdataList + channelNumYaxis + 1, color=colors_[int(channelNumYaxis//2.5) * 3 + 1])
                lineM, = plt.plot(timeList, channelNumYaxis + MdataList * 2, color=colors_[int(channelNumYaxis//2.5) * 3 + 2])
                line_dict[channel]['lineI'] = lineI
                line_dict[channel]['lineQ'] = lineQ
                line_dict[channel]['lineM'] = lineM
                line_dict[channel]['channelNumYaxis'] = channelNumYaxis
                channelNumYaxis += 2.5

            ax.axis([0, finalTime*1.1, -0.5, self.numOfChannel * 2.5 + 0.1])
            plt.subplots_adjust(bottom=0.25)
            axPos= plt.axes([0.15, 0.1, 0.7, 0.04], facecolor='lightgoldenrodyellow')
            indexSlider = Slider(ax=axPos, label='Index', valmin=0.0, valmax=self.numOfIndex-1, valstep=1)

            def indexUpdate(index_):      
                finalTime = 0
                for channel, index_dict in self.queue_dict.items():
                    height = line_dict[channel]['channelNumYaxis']
                    timeList = np.arange(0, self.maxTime + self.msmtLeakOutTime, 1)
                    IdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                    QdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                    MdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                    for time, pulse in index_dict[index_]:
                        time = int(time)
                        if pulse == 'DigTrigger':
                            for name, arrow in line_dict[channel].items():
                                if 'arrow' in name:
                                    arrow.set_xy(([[time, line_dict[channel]['channelNumYaxis']+2], [time, line_dict[channel]['channelNumYaxis']]]))
                        else:
                            pulseClass = self.W()[pulse]
                            pulseLength = pulseClass.width
                            IdataList[time : time + pulseLength] = pulseClass.I_data
                            QdataList[time : time + pulseLength] = pulseClass.Q_data
                            MdataList[time - self.pulseMarkerDelay : time - self.pulseMarkerDelay + len(pulseClass.mark_data)] = pulseClass.mark_data
                            finalTime = max(finalTime, time + pulseLength)
                    line_dict[channel]['lineI'].set_ydata(IdataList + line_dict[channel]['channelNumYaxis'])
                    line_dict[channel]['lineQ'].set_ydata(QdataList + line_dict[channel]['channelNumYaxis'] + 1)
                    line_dict[channel]['lineM'].set_ydata(line_dict[channel]['channelNumYaxis'] + MdataList * 2)
                     
                ax.axis([0, finalTime*1.1, -0.5, self.numOfChannel * 2.5 + 0.1])
            indexSlider.on_changed(indexUpdate)

        else:
            print("plotType only accept 'word' or 'realPulse'")
            pass

        return indexSlider

    def __call__(self, plot=0, sortOrder='time'):

        if sortOrder == 'channel':
            returnDict = self.queue_dict
        elif sortOrder == 'time':
            from collections import OrderedDict
            self.time_queue_dict = {}
            for sweepIndex in range(self.numOfIndex):
                self.time_queue_dict[sweepIndex] = {}
                for channel, index_dict in self.queue_dict.items():
                    for index, timeAndPulse in index_dict.items():
                        if sweepIndex == index:
                            for time_, pulse_ in timeAndPulse:
                                if time_ not in self.time_queue_dict[sweepIndex].keys():
                                    self.time_queue_dict[sweepIndex][time_] = []
                                self.time_queue_dict[sweepIndex][time_].append([channel, pulse_])

            for index, timeDict in self.time_queue_dict.items():
                self.time_queue_dict[index] = dict(OrderedDict(sorted(timeDict.items())))
            returnDict = self.time_queue_dict

        return returnDict

class Experiments(ExperimentSequence):
    def __init__(self, module_dict, msmtInfoDict, subbuffer_used=0):
        super().__init__(module_dict, msmtInfoDict, subbuffer_used)

    ###################-----------------Pulse Definition---------------------#######################
    def driveAndMsmt(self):
        time_ = self.queuePulse('piPulse_gau.x', 0, 500, "Qdrive")
        self.addMsmt("msmt_box", 0, time_ + 40, "Cdrive", "Dig")
        return self.W, self.Q

    def piPulseTuneUp(self, ampArray):
        for i, amp in enumerate(ampArray):
            pi_pulse_ = self.W.cloneAddPulse('piPulse_gau.x', f'piPulse_gau.x.{i}', amp=amp)
            time_ = self.queuePulse(pi_pulse_, i, 500, "Qdrive")
            self.addMsmt("msmt_box", i, time_ + 40, "Cdrive", "Dig")
        return self.W, self.Q


if __name__ == '__main__':
    import yaml
    from HaPiCodes.test_examples import msmtInfoSel

    yamlFile = msmtInfoSel.cwYaml
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    dummy_modules = {"A1": 0, "A2": 0, "A3": 0, "A5": 0, "D1": 0, "M2": 0}
    WQ = Experiments(dummy_modules, msmtInfoDict, subbuffer_used=1)
    W, Q = WQ.piPulseTuneUp(ampArray=np.linspace(-0.5, 0.5, 101))
    #
    # pd = constructPulseDictFromYAML(msmtInfoDict["pulseParams"])
