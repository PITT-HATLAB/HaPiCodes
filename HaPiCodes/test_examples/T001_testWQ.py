import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.pulse.waveformAndQueue import ExperimentSequence
from HaPiCodes.test_examples import msmtInfoSel
from HaPiCodes.simulation.constants import *
from HaPiCodes.simulation.qutipPac import *

module_dict = { "A1": None,
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

msmtInfoDict = yaml.safe_load(open(msmtInfoSel.cwYaml, 'r'))

class SimulationExperiments(ExperimentSequence):
    def __init__(self, module_dict, msmtInfoDict, subbuffer_used=0):
        super().__init__(module_dict, msmtInfoDict, subbuffer_used)
        if "deviceInfo" not in msmtInfoDict.keys():
            raise AttributeError("You need to have 'deviceInfo' in yaml file to simulate.")
        else:
            device = msmtInfoDict['deviceInfo']

        self.opts = qt.Options(nsteps=1e6, atol=1e-9, rtol=1e-9)
        self.p_bar = qt.ui.progressbar.TextProgressBar()

        self.dim = device['dim']
        if len(self.dim) != 2:
            raise TypeError('Only support two mode, still developing more functionality')

        self.f_qubit = device['qubit']['freq'] * GHz
        self.t1_qubit = device['qubit']['T1'] * ns
        self.t2_qubit = device['qubit']['T2R'] * ns
        self.alpha_qubit = device['qubit']['alpha'] * GHz
        self.f_cav = device['cav']['freq'] * GHz
        self.I, self.q, self.c = operator_creator(self.dim, annihilation=1)
        self.qt1, self.qt2 = generateDecayAndDecoherence(self.t1_qubit, self.t2_qubit, self.q)

        self.qdrive_re = 0.5 * (self.q.dag() + self.q)
        self.qdrive_im = 0.5j * (self.q.dag() - self.q)
        self.qNum = self.q.dag() * self.q

        self.cdrive_re = 0.5 * (self.c.dag() + self.c)
        self.cdrive_im = 0.5j * (self.c.dag() - self.c)
        self.cNum = self.c.dag() * self.c

        # self.hamil0 = self.f_qubit * self.q.dag() * self.q + self.alpha_qubit/2 * self.q.dag() * self.q.dag() * self.q * self.q + self.f_cav * self.c.dag() * self.c
        self.hamil0 = self.alpha_qubit/2 * self.q.dag() * self.q.dag() * self.q * self.q + self.f_cav * self.c.dag() * self.c


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

    def generateDrivePulse(self):
        finalTime = 0
        channelNumYaxis = 0
        self.realPulseDict = {}
        for channel, index_dict in self.queue_dict.items():
            self.realPulseDict[channel] = {}

            timeList = np.arange(0, self.maxTime + self.msmtLeakOutTime, 1)
            IdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
            # IdataList[0] = 1e-6
            QdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
            # QdataList[0] = 1e-6
            MdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
            # MdataList[0] = 1e-6

            for idx, timePulse in index_dict.items():
                self.realPulseDict[channel][idx] = {}
                for time, pulse in timePulse:
                    time = int(time)
                    if pulse == 'DigTrigger':
                        self.realPulseDict[channel][idx]['Dig'] = time
                    else:
                        pulseClass = self.W()[pulse]
                        pulseLength = pulseClass.width
                        IdataList[time: time + pulseLength] = pulseClass.I_data
                        QdataList[time: time + pulseLength] = pulseClass.Q_data
                        MdataList[time - self.pulseMarkerDelay: time + pulseLength + 10] = pulseClass.mark_data
                        finalTime = max(finalTime, time + pulseLength)

                        self.realPulseDict[channel][idx]['I'] = IdataList
                        self.realPulseDict[channel][idx]['Q'] = QdataList
                        self.realPulseDict[channel][idx]['M'] = MdataList

    def plot_pulse(self, pulse, tlist):
        fig, ax = plt.subplots()
        if callable(pulse):
            pulse = np.array([pulse(t, args=None) for t in tlist])
        ax.plot(tlist, pulse)
        ax.set_xlabel('time')
        ax.set_ylabel('pulse amplitude')
        plt.show()

    def simulate(self):
        self.generateDrivePulse()
        qdrivePulse = self.realPulseDict['Qdrive'][0]['I']
        def qdrive_re_time(t, args):
            return qdrivePulse[int(np.floor(t))] * 1 * MHz

        rho0 = qt.tensor(qt.basis(self.dim[0], 0), qt.basis(self.dim[1], 0))
        lastTime = max(list(test()[0].keys()))
        tlist = np.linspace(0, lastTime, lastTime + 1)
        self.plot_pulse(qdrive_re_time, tlist)
        hamil = [self.hamil0, [self.qdrive_re, qdrive_re_time]]
        res = qt.mesolve(hamil, rho0, tlist, [], [self.qNum, self.cNum], options=self.opts, progress_bar=self.p_bar)

        return res


if __name__ == '__main__':
    test = SimulationExperiments(module_dict, msmtInfoDict, 1)
    W, Q = test.driveAndMsmt()
    # test.generateDrivePulse()
    res = test.simulate()
    plt.figure()
    plt.plot(res.expect[0])
    plt.plot(res.expect[1])