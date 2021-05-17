from HaPiCodes.pulse.waveformAndQueue import ExperimentSequence

import numpy as np


class BasicExperiments(ExperimentSequence):
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

    def t2R(self, timeArray):
        for i, iTime in enumerate(timeArray):
            time_ = self.queuePulse('piPulse_gau.x2', i, 500, "Qdrive")
            time_ += iTime
            time_ = self.queuePulse('piPulse_gau.x2', i, time_, "Qdrive")
            self.addMsmt("msmt_box", i, time_ + 40, "Cdrive", "Dig")
        return self.W, self.Q

    def t2E(self, timeArray):
        for i, iTime in enumerate(timeArray):
            time_ = self.queuePulse('piPulse_gau.x2', i, 500, "Qdrive")
            time_ += iTime/2
            time_ = self.queuePulse('piPulse_gau.x', i, time_, "Qdrive")
            time_ += iTime/2
            time_ = self.queuePulse('piPulse_gau.x2', i, time_, "Qdrive")
            self.addMsmt("msmt_box", i, time_ + 40, "Cdrive", "Dig")
        return self.W, self.Q

    def multiPiPulse(self, numOfPiPulse=10):
        for i in range(numOfPiPulse):
            time_ = 500
            for j in range(i):
                time_ = self.queuePulse('piPulse_gau.x2', i, time_, "Qdrive")
                time_ += 40
            self.addMsmt("msmt_box", i, time_, "Cdrive", "Dig")

        return self.W, self.Q
'''
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

            QSBAndDrive = ps.combinePulse([self.pulse_defined_dict['QSB_box'].smooth(), self.piPulse_gau.x()], [2000])
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
'''

if __name__ == '__main__':
    print('hello')
    module_dict = {"A1": None,
                   "A2": None,
                   "A3": None,
                   "A4": None,
                   "A5": None,
                   "M1": None,
                   "M2": None,
                   "M3": None,
                   "D1": None,
                   "D2": None
                   }
    import yaml
    from HaPiCodes.test_examples import msmtInfoSel
    yamlFile = msmtInfoSel.cwYaml
    msmtInfoDict = yaml.safe_load(open(yamlFile, 'r'))
    WQ = BasicExperiments(module_dict, msmtInfoDict, subbuffer_used=0)
    W, Q = WQ.t2E(np.linspace(0, 10000, 101))
    redict, slider = WQ(plot=2)
