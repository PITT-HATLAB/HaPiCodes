import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HaPiCodes.test_examples import msmtInfoSel
from HaPiCodes.simulation.qutipSim import SimulationExperiments


class BasicExperiments(SimulationExperiments):
    def __init__(self, msmtInfoDict):
        super().__init__(msmtInfoDict)

    def driveAndMsmt(self):
        time_ = self.queuePulse('piPulse_gau.x', 0, 500, "q1Drive")
        self.addDigTrigger(0, time_ + 500, "Dig")
        return self.W, self.Q

    def testPulse(self):
        time_ = 500
        time_ = self.queuePulse('piPulse_gau.x', 0, time_, "q1Drive")
        time_ = self.queuePulse('piPulse_box', 0, time_, "q1Drive")
        time_ = self.queuePulse('piPulse_hanning', 0, time_, "q1Drive")
        self.queuePulse('piPulse_boxOnOff', 0, time_, "q1Drive")
        return self.W, self.Q

    def piPulseTuneUp(self, ampArray):
        for i, amp in enumerate(ampArray):
            pi_pulse_ = self.W.cloneAddPulse('piPulse_gau.x', f'piPulse_gau.x.{i}', amp=amp)
            time_ = self.queuePulse(pi_pulse_, i, 500, "q1Drive")
            self.addDigTrigger(i, time_ + 500, "Dig")
        return self.W, self.Q

    def dragTuneUp(self, dragArray):
        for i, iDrag in enumerate(dragArray):

            yp = self.W.cloneAddPulse('piPulse_gau.y', f'piPulse_gau.y.{i}', dragFactor=iDrag)
            x9 = self.W.cloneAddPulse('piPulse_gau.x2', f'piPulse_gau.x2.{i}', dragFactor=iDrag)
            xp = self.W.cloneAddPulse('piPulse_gau.x', f'piPulse_gau.x.{i}', dragFactor=iDrag)
            y9 = self.W.cloneAddPulse('piPulse_gau.y2', f'piPulse_gau.y2.{i}', dragFactor=iDrag)
            time_ = 500

            if i % 2 == 0:
                time_ = self.queuePulse(yp, i, time_, "q1Drive")
                time_ = self.queuePulse(x9, i, time_ + 40, "q1Drive")
            else:
                time_ = self.queuePulse(xp, i, time_, "q1Drive")
                time_ = self.queuePulse(y9, i, time_ + 40, "q1Drive")

            self.addDigTrigger(i, time_ + 500, "Dig")

        return self.W, self.Q

    def exchangeBetweenQ1AndC1(self, timeArray):
        for i, iTime in enumerate(timeArray):
            time_ = self.queuePulse('piPulse_gau.x', i, 500, "q1Drive")
            exchange = self.W.cloneAddPulse('exchange_box', f'exchange_box.{i}', width=iTime)
            time_ = self.queuePulse(exchange, i, time_ + 100, "Sq1c1Drive")
            self.addDigTrigger(i, time_ + 500, "Dig")
        return self.W, self.Q

    #TODO: WIP only works for 1 qubit gates for now
    #assumes no data depenedencies for now since only 1 qubit gates...
    #for later assume uses barriers between dependencies? - not sure better way rn
    def generateFromCircuit(self, qcircuit):
        time_ = [50, 50] #starting offset for 2 qubits
        for i, gate in enumerate(qcircuit.gates):
            ####dangerous hardcoding
            target = gate.targets[0] #0, 1
            if gate.name != 'M':
                pulse_name = gate.name[1].lower() #x, y
                amp = gate.arg_value * .15/np.pi
            ####
                pi_pulse_ = self.W.cloneAddPulse(f'piPulse_gau.{pulse_name}', f'piPulse_gau.{pulse_name}.{i}', amp=amp)
                time_[target] = self.queuePulse(pi_pulse_, 0, 100+time_[target], f"q{1+target}Drive")
            else:
                #Not working?
                self.addDigTrigger(0, time_[target]+ 500, "Dig")
                
        return self.W, self.Q