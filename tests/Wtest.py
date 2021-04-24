from copy import deepcopy

def getPulseDictFromYAML():



class Waveforms():
    def __init__(self, pulseDict, var):
        self.pulseDict = pulseDict
        self.var = var

    def addPulse(self, name, pulse):
        # updateW
        pass

    def cloneAddPulse(self, pulseName, newName, paramName, paramVal):
        pulse = deepcopy(self.pulseDict[pulseName])
        self.checkNewName(self, newName)
        setattr(pulse, paramName, paramVal)
        self.addPulse(newName, pulse)


    def __call__(self):
        return self.pulseDict


class Pulse(object):  # Pulse.data_list, Pulse.I_data, Pulse.Q_data
    def __init__(self, width, ssb_freq=0, iqscale=1, phase=0, skew_phase=0):
        self.init_Dict = {}

    def copy(self):
        return Pulse(**self.init_Dict)