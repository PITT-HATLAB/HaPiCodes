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

    def piPulseTuneUp(self, ampArray):
        for i, amp in enumerate(ampArray):
            pi_pulse_ = self.W.cloneAddPulse('piPulse_gau.x', f'piPulse_gau.x.{i}', amp=amp)
            time_ = self.queuePulse(pi_pulse_, i, 500, "q1Drive")
            self.addDigTrigger(i, time_ + 500, "Dig")
        return self.W, self.Q

    def exchangeBetweenQ1AndC1(self, timeArray):
        for i, iTime in enumerate(timeArray):
            time_ = self.queuePulse('piPulse_gau.x', i, 500, "q1Drive")
            exchange = self.W.cloneAddPulse('exchange_box', f'exchange_box.{i}', width=iTime)
            time_ = self.queuePulse(exchange, i, time_ + 100, "Sq1c1Drive")
            self.addDigTrigger(i, time_ + 500, "Dig")
        return self.W, self.Q