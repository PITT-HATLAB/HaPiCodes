import warnings
import logging
from typing import Dict, List, Union

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import yaml

from HaPiCodes.pulse import pulses as pc
from HaPiCodes.pulse.pulses import Pulse


import testInfoSel
yamlFile=testInfoSel.cwYaml
msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))


class modulesWaveformCollection(object):
    """ waveForm collection for all modules, for uploading to HVI
    """
    def __init__(self, module_dict):
        for module in module_dict.keys():
            setattr(self, str(module), {})
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



class Waveforms:
    def __init__(self, pulseDict:Dict[str, Pulse]):
        self.pulseDict = pulseDict

    def addPulse(self, name, pulse):
        # updateW
        pass

    def cloneAddPulse(self, pulseName, newName, paramName, paramVal):
        # pulse = deepcopy(self.pulseDict[pulseName])
        self.checkNewName(self, newName)
        # setattr(pulse, paramName, paramVal)
        # self.addPulse(newName, pulse)


    def __call__(self):
        return self.pulseDict

class Queue:
    pass





def constructPulseDictFromYAML(pulseParams: Dict[str, Dict],
                               userPulsePackage=None) -> Dict[str, Pulse]:
    """Generate pulses defined in the pulseParams in YAML file, and put the pulses
    in a dictionary. This function will first try to find the pulse shape in the
    built in pulse shape package (HaPiCodes.pulse.pulses), if not found, the
    function will try to find pulse shape in userPulsePackage.

    :param pulseParams: Dict[pulse_name: pule_parameters]
    :param userPulsePackage: package that contains the user defined pulses
    :return: Dict[pulse_name, Pulse]
    """
    pulseDict = {}
    for name, param in pulseParams.items():
        pulse_shape = param.pop("shape")
        try:
            pulse_class = getattr(pc, pulse_shape)
        except AttributeError:
            try:
                pulse_class = getattr(userPulsePackage, pulse_shape)
            except AssertionError:
                raise NameError(f"Can't find pulse {pulse_shape} in either built-in pulse library "
                                f"or user pulse library {userPulsePackage}")
        pulseDict[name] = pulse_class(param)

    return pulseDict





"""
class ExperiemntSqeuence():
    def __init__(self, module_dict, yamlDict, subbuffer_used=0):
        self._W = modulesWaveformCollection(module_dict)
        self._Q = modulesQueueCollection(module_dict)
        self.W = Waveforms()
        self.Q = Queue()


        self.subbuffer_used = subbuffer_used

        self.info = yamlDict

        self.piPulse_gau_condition = self.info['pulseParams']['piPulse_gau']
        self.piPulse_gau = pc.gau(self.piPulse_gau_condition)

        self.msmt_box_condition = self.info['pulseParams']['msmt_box']
        self.msmt_box = pc.box(self.msmt_box_condition)

        self.pulse_defined_dict = {}
        for pulseName in self.info['pulseParams'].keys():
            if pulseName[-3:] == 'box':
                self.pulse_defined_dict[pulseName] = pc.box(self.info['pulseParams'][pulseName])
            if pulseName[-3:] == 'gau':
                self.pulse_defined_dict[pulseName] = pc.gau(self.info['pulseParams'][pulseName])

        self.QdriveChannel = self.info['combinedChannelUsage']['Qdrive']
        self.CdriveChannel = self.info['combinedChannelUsage']['Cdrive']
        self.DigChannel = self.info['combinedChannelUsage']['Dig']

        self.qDriveMsmtDelay = self.info['regularMsmtPulseInfo']['qDriveMsmtDelay']
        self.digMsmtDelay = self.info['regularMsmtPulseInfo']['digMsmtDelay']

        for module in module_dict:
            getattr(self.W, module)['trigger.dig'] = []
            getattr(self.W, module)['trigger.fpga4'] = []
            getattr(self.W, module)['trigger.fpga5'] = []
            getattr(self.W, module)['trigger.fpga6'] = []
            getattr(self.W, module)['trigger.fpga7'] = []

    def updateW(self, module, pulseName, pulse):
        if pulseName in getattr(self.W, module).keys():
            pass
        else:
            getattr(self.W, module)[pulseName] = pulse

    def updateWforIQM(self, name, pulse, driveChan, Mupdate=1):
        self.updateW(driveChan['I'][0], name + '.I', pulse.I_data)
        self.updateW(driveChan['Q'][0], name + '.Q', pulse.Q_data)
        if Mupdate:
            self.updateW(driveChan['M'][0], name + '.M', pulse.mark_data)

    def updateQforIQM(self, pulseName, index, time, driveChan, Mupdate=1):
        if pulseName + '.I' not in getattr(self.W, driveChan['I'][0]).keys():
            raise NameError(pulseName + " is not initialize in W yet", driveChan['I'][0])
        self.Q.add(driveChan['I'][0], driveChan['I'][1], index, pulseName + '.I', time)
        self.Q.add(driveChan['Q'][0], driveChan['Q'][1], index, pulseName + '.Q', time)
        if Mupdate:
            self.Q.add(driveChan['M'][0], driveChan['M'][1], index, pulseName + '.M', time)

    def addQdrive(self, pulseName, index, time):
        self.updateQforIQM(pulseName, index, time, driveChan=self.QdriveChannel)

    def addCdrive(self, pulseName, index, time):
        self.updateQforIQM(pulseName, index, time, driveChan=self.CdriveChannel)

    def addCdriveAndMSMT(self, pulseName, index, time):
        self.addCdrive(pulseName, index, time)
        if time - self.digMsmtDelay < 0:
            raise ValueError(f"C drive time for MSMT must be longer than digMsmtDelay ({self.digMsmtDelay})")
        self.addMsmt(index, time - self.digMsmtDelay)


    def addMsmt(self, index, time):
        fgpaTriggerSig = 'trigger.fpga' + str(3 + self.DigChannel['Sig'][1])
        fgpaTriggerRef = 'trigger.fpga' + str(3 + self.DigChannel['Ref'][1])

        if not self.subbuffer_used:
            self.Q.add(self.DigChannel['Sig'][0], self.DigChannel['Sig'][1], index, 'trigger.dig', time, msmt=True)
            self.Q.add(self.DigChannel['Ref'][0], self.DigChannel['Ref'][1], index, 'trigger.dig', time, msmt=True)
            self.Q.add(self.DigChannel['Sig'][0], self.DigChannel['Sig'][1], index, fgpaTriggerSig, time, msmt=False)
            self.Q.add(self.DigChannel['Ref'][0], self.DigChannel['Ref'][1], index, fgpaTriggerRef, time, msmt=False)
        else:
            self.Q.add(self.DigChannel['Sig'][0], self.DigChannel['Sig'][1], index, fgpaTriggerSig, time, msmt=True)
            self.Q.add(self.DigChannel['Ref'][0], self.DigChannel['Ref'][1], index, fgpaTriggerRef, time, msmt=True)

"""


constructPulseDictFromYAML(msmtInfoDict["pulseParams"])