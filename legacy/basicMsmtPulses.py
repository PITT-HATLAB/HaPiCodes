import HaPiCodes.pulse.pulses as pc
import numpy as np
import yaml


class waveformAndQueue():
    def __init__(self, module_dict, yamlDict, subbuffer_used=0):
        self.W = pc.modulesWaveformCollection(module_dict)
        self.Q = pc.modulesQueueCollection(module_dict)
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

###################-----------------Pulse Defination---------------------#########################

    def driveAndMsmt(self, qdrive=1):
        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)
        if qdrive:
            self.addQdrive('pulse.piPulse_gau', 0, 200)
        self.addCdriveAndMSMT('pulse.msmt_box', 0, 200 + self.qDriveMsmtDelay)
        return self.W, self.Q

    def piPulseTuneUp(self):
        minAmp = self.info['regularMsmtPulseInfo']['piPulseTuneUpAmp'][0]
        maxAmp = self.info['regularMsmtPulseInfo']['piPulseTuneUpAmp'][1]
        nStep = self.info['regularMsmtPulseInfo']['piPulseTuneUpAmp'][2]
        ampArray = np.linspace(minAmp, maxAmp, nStep + 1)[:nStep]

        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)
        for i in range(nStep):
            self.piPulse_gau.amp = ampArray[i]
            self.updateWforIQM(f'pulse.piPulse_gau{i}', self.piPulse_gau.x(), self.QdriveChannel)

        for i in range(nStep):
            self.addQdrive(f'pulse.piPulse_gau{i}', i, 200)
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
        return self.W, self.Q

    def t1Msmt(self):
        minTime = self.info['regularMsmtPulseInfo']['T1MsmtTime'][0] * 1e3
        maxTime = self.info['regularMsmtPulseInfo']['T1MsmtTime'][1] * 1e3
        nStep = self.info['regularMsmtPulseInfo']['T1MsmtTime'][2]
        timeArray = np.linspace(minTime, maxTime, nStep + 1, dtype=int)[:nStep]

        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        for i in range(nStep):
            self.addQdrive('pulse.piPulse_gau', i, 200)
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay + timeArray[i])
        return self.W, self.Q

    def t2RMsmt(self):
        minTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][0] * 1e3
        maxTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][1] * 1e3
        nStep = self.info['regularMsmtPulseInfo']['T2MsmtTime'][2]
        timeArray = np.linspace(minTime, maxTime, nStep + 1, dtype=int)[:nStep]

        self.updateWforIQM('pulse.piOver2Pulse_gau', self.piPulse_gau.x2(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        for i in range(nStep):
            self.addQdrive('pulse.piOver2Pulse_gau', i, 200)
            self.addQdrive('pulse.piOver2Pulse_gau', i, 500 + timeArray[i])
            self.addCdriveAndMSMT('pulse.msmt_box', i, 500 + self.qDriveMsmtDelay + timeArray[i])
        return self.W, self.Q

    def t2EMsmt(self):
        minTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][0] * 1e3
        maxTime = self.info['regularMsmtPulseInfo']['T2MsmtTime'][1] * 1e3
        nStep = self.info['regularMsmtPulseInfo']['T2MsmtTime'][2]
        timeArray = np.linspace(minTime, maxTime, nStep + 1, dtype=int)[:nStep]

        self.updateWforIQM('pulse.piOver2Pulse_gau', self.piPulse_gau.x2(), self.QdriveChannel)
        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        for i in range(nStep):
            self.addQdrive('pulse.piOver2Pulse_gau', i, 200)
            self.addQdrive('pulse.piPulse_gau', i, 400 + timeArray[i] // 2)
            self.addQdrive('pulse.piOver2Pulse_gau', i, 600 + timeArray[i])
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + 400 + self.qDriveMsmtDelay + timeArray[i])
        return self.W, self.Q

    def allXY(self):
        self.updateWforIQM('pulse.pix2_gau', self.piPulse_gau.x2(), self.QdriveChannel)
        self.updateWforIQM('pulse.pix_gau', self.piPulse_gau.x(), self.QdriveChannel)
        self.updateWforIQM('pulse.piy2_gau', self.piPulse_gau.y2(), self.QdriveChannel)
        self.updateWforIQM('pulse.piy_gau', self.piPulse_gau.y(), self.QdriveChannel)
        self.updateWforIQM('pulse.pix2N_gau', self.piPulse_gau.x2N(), self.QdriveChannel)
        self.updateWforIQM('pulse.piy2N_gau', self.piPulse_gau.y2N(), self.QdriveChannel)
        self.updateWforIQM('pulse.pioff_gau', self.piPulse_gau.off(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

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
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
        return self.W, self.Q


    def pulseSpec(self):
        self.piPulse_gau.amp /= 10
        self.piPulse_gau.sigma *= 10

        minFreq = 0.09
        maxFreq = 0.11
        step = 100
        freqArray = np.linspace(minFreq, maxFreq, step + 1)[:100]

        for i in range(step):
            self.piPulse_gau.ssbFreq = freqArray[i]
            self.updateWforIQM(f'pulse.piPulse_gau{i}', self.piPulse_gau.x(), self.QdriveChannel)
            self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        for i in range(step):
            self.addQdrive(f'pulse.piPulse_gau{i}', i, 200)
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
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
            self.updateWforIQM(f'pulse.QSBAndDrive{i}', QSBAndDrive, self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        for i in range(step):
            self.addQdrive(f'pulse.QSBAndDrive{i}', i, 200)
            self.addCdriveAndMSMT('pulse.msmt_box', i, 200 + self.qDriveMsmtDelay)
        return self.W, self.Q


    def selectionDriveAndMsmt(self):
        self.updateWforIQM('pulse.piPulse_gau', self.piPulse_gau.x(), self.QdriveChannel)
        self.updateWforIQM('pulse.piOver2Pulse_gau', self.piPulse_gau.x2(), self.QdriveChannel)
        self.updateWforIQM('pulse.msmt_box', self.msmt_box.smooth(), self.CdriveChannel)

        self.addQdrive('pulse.piOver2Pulse_gau', 0, 200)
        self.addCdriveAndMSMT('pulse.msmt_box', 0, 200 + self.qDriveMsmtDelay)
        # self.addQdrive('pulse.piPulse_gau', 0, 5200)
        self.addCdriveAndMSMT('pulse.msmt_box', 0, 5200 + self.qDriveMsmtDelay)
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

