import pulse.pulseConfig as pc
import numpy as np
import yaml


class waveformAndQueue():
    def __init__(self, module_dict, yamlDict, subbuffer_used=0):
        self.W = pc.modulesWaveformCollection(module_dict)
        self.Q = pc.modulesQueueCollection(module_dict)
        self.subbuffer_used = subbuffer_used
        self.info = yamlDict

        self.piPulse_gau_condtion = self.info['pulseParams']['piPulse_gau']
        self.piPulse_gau = pc.gau(self.piPulse_gau_condtion)

        self.msmt_box_condition = self.info['pulseParams']['msmt_box']
        self.msmt_box = pc.box(self.msmt_box_condition)

        self.pulse_defined_dict = {}
        for pulseName in self.info['pulseParams'].keys():
            if pulseName[-3:] == 'box':
                self.pulse_defined_dict[pulseName] = pc.box(self.info['pulseParams'][pulseName])
            if pulseName[-3:] == 'gau':
                self.pulse_defined_dict[pulseName] = pc.gau(self.info['pulseParams'][pulseName])

        self.QdriveInfo = self.info['combinedChannelUsage']['Qdrive']
        self.CdriveInfo = self.info['combinedChannelUsage']['Cdrive']
        self.DigInfo = self.info['combinedChannelUsage']['Dig']

        self.qDriveMsmtDelay = self.info['regularMsmtPulseInfo']['qDriveMsmtDelay']

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

    def updateWforIQM(self, name, pulse, driveInfo, Mupdate=1):
        self.updateW(driveInfo['I'][0], name + '.I', pulse.I_data)
        self.updateW(driveInfo['Q'][0], name + '.Q', pulse.Q_data)
        if Mupdate:
            self.updateW(driveInfo['M'][0], name + '.M', pulse.mark_data)

    def updateQforIQM(self, pulseName, index, time, driveInfo, Mupdate=1):
        if pulseName + '.I' not in getattr(self.W, driveInfo['I'][0]).keys():
            raise NameError(pulseName + " is not initialize in W yet", driveInfo['I'][0])
        self.Q.add(driveInfo['I'][0], driveInfo['I'][1], index, pulseName + '.I', time)
        self.Q.add(driveInfo['Q'][0], driveInfo['Q'][1], index, pulseName + '.Q', time)
        if Mupdate:
            self.Q.add(driveInfo['M'][0], driveInfo['M'][1], index, pulseName + '.M', time - 100)

    def addQdrive(self, pulseName, index, time):
        self.updateQforIQM(pulseName, index, time, driveInfo=self.QdriveInfo)

    def addCdrive(self, pulseName, index, time):
        self.updateQforIQM(pulseName, index, time, driveInfo=self.CdriveInfo)

    def addMsmt(self, index, time):
        if not self.subbuffer_used:
            self.Q.add(self.DigInfo['Sig'][0], self.DigInfo['Sig'][1], index, 'trigger.dig', time, msmt=True)
            self.Q.add(self.DigInfo['Ref'][0], self.DigInfo['Ref'][1], index, 'trigger.dig', time, msmt=True)
        else:
            fgpaTriggerIndex = 3 + self.DigInfo['Sig'][1]
            self.Q.add(self.DigInfo['Sig'][0], self.DigInfo['Sig'][1], index, 'trigger.fpga' + str(fgpaTriggerIndex), time, msmt=True)
            fgpaTriggerIndex = 3 + self.DigInfo['Ref'][1]
            self.Q.add(self.DigInfo['Ref'][0], self.DigInfo['Ref'][1], index, 'trigger.fpga' + str(fgpaTriggerIndex), time, msmt=True)

###################-----------------Pulse Defination---------------------#########################

    def driveAndMsmt(self):
        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveInfo)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)
        self.addQdrive('pulse.piPulse_gau', 0, 200)
        self.addCdrive('pulse.msmt_box', 0, 200 + self.qDriveMsmtDelay)
        self.addMsmt(0, 40)

        return self.W, self.Q

    def piPulseTuneUp(self):
        maxAmp = self.info['regularMsmtPulseInfo']['piPulseTuneUpAmp'][0]
        step = self.info['regularMsmtPulseInfo']['piPulseTuneUpAmp'][1]
        ampArray = np.linspace(-maxAmp, maxAmp, step + 1)[:step]

        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)
        for i in range(step):
            self.piPulse_gau.amp = ampArray[i]
            self.updateWforIQM(f'pulse.piPulse_gau{i}', self.piPulse_gau.x(), self.QdriveInfo)

        for i in range(step):
            self.addQdrive(f'pulse.piPulse_gau{i}', i, 200)
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
            self.addMsmt(i, 0)
        return self.W, self.Q

    def t1Msmt(self):
        maxTime = self.info['regularMsmtPulseInfo']['T1MsmtTime'][0]
        step = self.info['regularMsmtPulseInfo']['T1MsmtTime'][1]
        timeArray = np.linspace(0, maxTime * 1e3, step + 1, dtype=int)[:step]

        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveInfo)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        for i in range(step):
            self.addQdrive('pulse.piPulse_gau', i, 200)
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay + timeArray[i])
            self.addMsmt(i, timeArray[i])
        return self.W, self.Q

    def t2RMsmt(self):
        maxTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][0]
        step = self.info['regularMsmtPulseInfo']['T2MsmtTime'][1]
        timeArray = np.linspace(0, maxTime * 1e3, step + 1, dtype=int)[:step]

        self.updateWforIQM('pulse.piOver2Pulse_gau', self.piPulse_gau.x2(), self.QdriveInfo)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        for i in range(step):
            self.addQdrive('pulse.piOver2Pulse_gau', i, 200)
            self.addQdrive('pulse.piOver2Pulse_gau', i, 400 + timeArray[i])
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay + timeArray[i])
            self.addMsmt(i, timeArray[i])
        return self.W, self.Q

    def t2EMsmt(self):
        maxTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][0]
        step = self.info['regularMsmtPulseInfo']['T2MsmtTime'][1]
        timeArray = np.linspace(0, maxTime * 1e3, step + 1, dtype=int)[:step]

        self.updateWforIQM('pulse.piOver2Pulse_gau', self.piPulse_gau.x2(), self.QdriveInfo)
        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveInfo)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        for i in range(step):
            self.addQdrive('pulse.piOver2Pulse_gau', i, 200)
            self.addQdrive('pulse.piPulse_gau', i, 300 + timeArray[i] // 2)
            self.addQdrive('pulse.piOver2Pulse_gau', i, 400 + timeArray[i])
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay + timeArray[i])
            self.addMsmt(i, timeArray[i])
        return self.W, self.Q

    def allXY(self):
        self.updateWforIQM('pulse.pix2_gau', self.piPulse_gau.x2(), self.QdriveInfo)
        self.updateWforIQM('pulse.pix_gau', self.piPulse_gau.x(), self.QdriveInfo)
        self.updateWforIQM('pulse.piy2_gau', self.piPulse_gau.y2(), self.QdriveInfo)
        self.updateWforIQM('pulse.piy_gau', self.piPulse_gau.y(), self.QdriveInfo)
        self.updateWforIQM('pulse.pix2N_gau', self.piPulse_gau.x2N(), self.QdriveInfo)
        self.updateWforIQM('pulse.piy2N_gau', self.piPulse_gau.y2N(), self.QdriveInfo)
        self.updateWforIQM('pulse.pioff_gau', self.piPulse_gau.off(), self.QdriveInfo)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        pulseList = [['pulse.pioff_gau', 'pulse.pioff_gau'],
                     ['pulse.pix_gau', 'pulse.pix_gau'],
                     ['pulse.piy_gau', 'pulse.piy_gau'],
                     ['pulse.pix_gau', 'pulse.piy_gau'],
                     ['pulse.piy_gau', 'pulse.pix_gau'],  # first: give u g
                     ['pulse.pix2_gau', 'pulse.pioff_gau'],
                     ['pulse.piy2_gau', 'pulse.pioff_gau'],
                     ['pulse.pix2_gau', 'pulse.piy2_gau'],
                     ['pulse.piy2_gau', 'pulse.pix2_gau'],
                     ['pulse.pix2_gau', 'pulse.piy_gau'],
                     ['pulse.piy2_gau', 'pulse.pix_gau'],
                     ['pulse.pix_gau', 'pulse.piy2_gau'],
                     ['pulse.piy_gau', 'pulse.pix2_gau'],
                     ['pulse.pix2_gau', 'pulse.pix_gau'],
                     ['pulse.pix_gau', 'pulse.pix2_gau'],
                     ['pulse.piy2_gau', 'pulse.piy_gau'],
                     ['pulse.piy_gau', 'pulse.piy2_gau'],  # second: give u g+e
                     ['pulse.pix_gau', 'pulse.pioff_gau'],
                     ['pulse.piy_gau', 'pulse.pioff_gau'],
                     ['pulse.pix2_gau', 'pulse.pix2_gau'],
                     ['pulse.piy2_gau', 'pulse.piy2_gau']]  # third: give u e

        for i in range(21):
            self.addQdrive(pulseList[i][0], i, 200)
            self.addQdrive(pulseList[i][1], i, 300)
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
            self.addMsmt(i, 0)

    def pulseSpec(self):
        self.piPulse_gau.amp /= 10
        self.piPulse_gau.sigma *= 10

        minFreq = 0.09
        maxFreq = 0.11
        step = 100
        freqArray = np.linspace(minFreq, maxFreq, step + 1)[:100]

        for i in range(step):
            self.piPulse_gau.ssbFreq = freqArray[i]
            self.updateWforIQM(f'pulse.piPulse_gau{i}', self.piPulse_gau.x(), self.QdriveInfo)
            self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        for i in range(step):
            self.addQdrive(f'pulse.piPulse_gau{i}', i, 200)
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
            self.addMsmt(i, 0)
        return self.W, self.Q

    def pulseSpecWithQSB(self):
        self.piPulse_gau.amp /= 10
        self.piPulse_gau.sigma *= 10

        minFreq = 0.09
        maxFreq = 0.11
        step = 100
        freqArray = np.linspace(minFreq, maxFreq, step + 1)[:100]

        for i in range(step):
            self.piPulse_gau.ssbFreq = freqArray[i]

            QSBAndDrive = pc.combinePulse([self.pulse_defined_dict['QSB_box'].smooth(), self.piPulse_gau.x()], [2000])
            self.updateWforIQM(f'pulse.QSBAndDrive{i}', QSBAndDrive, self.QdriveInfo)
            self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveInfo)

        for i in range(step):
            self.addQdrive(f'pulse.QSBAndDrive{i}', i, 200)
            self.addCdrive('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
            self.addMsmt(i, 0)
        return self.W, self.Q


if __name__ == '__main__':
    print('hello')
    moduleDict = {'A1': [],
                  'A2': [],
                  'D1': [],
                  'D2': [],
                  'M1': [],
                  'M2': []}
    # WQ = waveformAndQueue(moduleDict, "1224Q1_info.yaml", subbuffer_used=0)
    # W, Q = WQ.pulseSpecWithQSB()
    # print(W.M1)

    with open('1224Q1_info.yaml') as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    print(info)

    with open('1224Q1_infoV2.yaml', 'w') as file:
        yaml.safe_dump(info, file, sort_keys=False, default_flow_style=None)