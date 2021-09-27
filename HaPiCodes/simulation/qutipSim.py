import os
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.pulse.waveformAndQueue import ExperimentSequence
from HaPiCodes.test_examples import msmtInfoSel
from HaPiCodes.simulation.constants import *
from HaPiCodes.simulation.qutipPac import *
from qutip.parallel import parallel_map
import types
from functools import partial


def copy_func(f, name=None):
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
        f.__defaults__, f.__closure__)

module_dict_ = {"A0": None,
                "A1": None,
                "A2": None,
                "A3": None,
                "A4": None,
                "A5": None,
                "M0": None,
                "M1": None,
                "M2": None,
                "M3": None,
                "D1": None,
                "D2": None
               }

class SimulationExperiments(ExperimentSequence):
    def __init__(self, msmtInfoDict, module_dict=None, subbuffer_used=0):
        if module_dict is None:
            module_dict = module_dict_
        super().__init__(module_dict, msmtInfoDict, subbuffer_used)

        if "deviceInfo" not in msmtInfoDict.keys() or 'expSimCor' not in msmtInfoDict.keys():
            raise AttributeError("You need to have both 'deviceInfo' and 'expSimCor' in yaml file to simulate.")
        else:
            self.device = msmtInfoDict['deviceInfo']
            self.drive = msmtInfoDict['expSimCor']

        self.opts = qt.Options(nsteps=1e6, atol=1e-9, rtol=1e-9)
        self.p_bar = qt.ui.progressbar.TextProgressBar()

        self.dim = msmtInfoDict['dim']
        if len(self.device.keys()) != len(self.dim):
            raise ValueError("dim length should be the same as device number")

        self.modes = operator_creator(self.dim, annihilation=1)
        self.numOp = []
        for modeName, modeInfo in self.device.items():
            if modeInfo['idx'] >= len(self.dim):
                raise ValueError("idx should be smaller than the length of dim")

            # Now, self.device['q1']['op'] will be the corresponding operator.
            self.device[modeName]['op'] = self.modes[modeInfo['idx'] + 1]
            self.numOp.append(self.device[modeName]['op'].dag() * self.device[modeName]['op'])

        # TODO: need to add CNOT
        self.driveOp = {}
        self.driveFunc = {}
        for driveName, driveInfo in self.drive.items():
            if driveInfo['cato'] == 'drive':

                self.driveOp[driveName] = [self.device[driveInfo['mode']]['op'].dag() + self.device[driveInfo['mode']]['op'], 1j * (self.device[driveInfo['mode']]['op'].dag() - self.device[driveInfo['mode']]['op'])]

            elif driveInfo['cato'] == 'swap':
                self.driveOp[driveName] = self.device[driveInfo['mode'][0]]['op'].dag() * self.device[driveInfo['mode'][1]]['op'] + \
                                          self.device[driveInfo['mode'][0]]['op'] * self.device[driveInfo['mode'][1]]['op'].dag()

            else:
                raise TypeError('need to finish different types of drive')

        # TODO: need to finish all Hamiltonian term
        self.hamil0 = 0
        for modeName, modeInfo in self.device.items():
            if 'alpha' in modeInfo.keys():
                self.hamil0 += modeInfo['alpha'] * GHz * self.device[modeName]['op'].dag() * self.device[modeName]['op'].dag() * self.device[modeName]['op'] * self.device[modeName]['op']

    def generateDrivePulse(self):
        finalTime = 0
        self.realPulseDict = {}
        for channel, index_dict in self.queue_dict.items():
            self.realPulseDict[channel] = {}

            timeList = np.arange(0, self.maxTime + self.msmtLeakOutTime, 1)

            for idx, timePulse in index_dict.items():
                self.realPulseDict[channel][idx] = {}
                IdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                IdataList[0] = 1e-6
                QdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                QdataList[0] = 1e-6
                MdataList = np.zeros(self.maxTime + self.msmtLeakOutTime)
                MdataList[0] = 1e-6
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

    def simulateSingleIndex(self, index, expect=1):
        # initial state
        rho0 = qt.basis(self.dim[0], 0)
        for i in range(len(self.dim) - 1):
            rho0 = qt.tensor(rho0, qt.basis(self.dim[i + 1], 0))

        # sequence time
        # TODO: when stop sequence?
        lastTime = max(list(self()[index].keys()))
        tlist = np.linspace(0, lastTime, lastTime + 1)

        # Hamiltonian
        hamil = [self.hamil0]

        def makeDriveFunc(drivePulse, driveInfo):
            def driveFunc(t, args):
                return drivePulse[int(np.floor(t))] * driveInfo['ampCor']
            return driveFunc

        for driveName, driveInfo in self.drive.items():
            if driveName in self.realPulseDict.keys():
                drivePulse = self.realPulseDict[driveName][index]['I']
                self.driveFunc[driveName] = makeDriveFunc(drivePulse, driveInfo)
                hamil.append([self.driveOp[driveName][0], self.driveFunc[driveName]])
                if np.abs(np.sum(self.realPulseDict[driveName][index]['Q'])) > 1e-12:
                    drivePulseQ = self.realPulseDict[driveName][index]['Q']
                    self.driveFunc[driveName + '_Q'] = makeDriveFunc(drivePulseQ, driveInfo)
                    hamil.append([self.driveOp[driveName][1], self.driveFunc[driveName + '_Q']])

        exOp = []
        if expect:
            exOp = self.numOp

        # TODO: add decay Op
        res = qt.mesolve(hamil, rho0, tlist, [], exOp, options=self.opts)
        return [res.expect[i][-1] for i in range(len(res.expect))]

    def simulate(self):
        res = parallel_map(self.simulateSingleIndex, list(self().keys()), progress_bar=self.p_bar)
        return np.array(res)

    ###################-----------------Obsolete---------------------#######################

    # def driveAndMsmt(self):
    #     time_ = self.queuePulse('piPulse_gau.x', 0, 500, "q1Drive")
    #     self.addDigTrigger(0, time_ + 500, "Dig")
    #     return self.W, self.Q
    #
    # def piPulseTuneUp(self, ampArray):
    #     for i, amp in enumerate(ampArray):
    #         pi_pulse_ = self.W.cloneAddPulse('piPulse_gau.x', f'piPulse_gau.x.{i}', amp=amp)
    #         time_ = self.queuePulse(pi_pulse_, i, 500, "q1Drive")
    #         self.addDigTrigger(i, time_ + 500, "Dig")
    #     return self.W, self.Q
    #

