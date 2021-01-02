import package_pulse as pp
import numpy as np

gauCondition = {'amp': 0.1913,#0.1129,#0.19274,
                'ssbFreq': 0.1,
                'iqScale': 1.0025,
                'phase': 0,
                'skewPhase': 94.68,
                'sigma': 20,
                'sigmaMulti': 6,
                'dragFactor': 0}

boxCondition = {'amp': 0.3,
                'width': 2000,
                'ssbFreq': 0.00,
                'iqScale': 1,
                'phase': 0,
                'skewPhase': 0,
                'rampSlope': 0.5,
                'cutFactor': 3}

def driveAndMsmt(module_dict):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    W.A1[f'pulse.gau.I'] = gPulse.x().I_data
    W.A1[f'pulse.gau.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []
    W.D1['trigger.fpga6'] = []
    W.D1['trigger.fpga7'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)
    Q.addTwoChan('A1', [1, 2], 0, ['pulse.gau.I', 'pulse.gau.Q'], 200)
    Q.addTwoChan('A1', [3, 4], 0, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 500)
    Q.add('M1', 1, 0, 'pulse.gau.M', 100)
    Q.add('M1', 2, 0, 'pulse.msmtBox.M', 400)
    Q.add('D1', 1, 0, 'trigger.dig', 0)
    Q.add('D1', 2, 0, 'trigger.dig', 0)
    Q.add('D1', 3, 0, 'trigger.dig', 0)
    Q.add('D1', 4, 0, 'trigger.dig', 0)
    Q.add('D1', 1, 0, 'trigger.fpga4', 20)
    return W, Q

def driveAndMsmtReal(module_dict):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    W.A1[f'pulse.gau.I'] = gPulse.x().I_data
    W.A1[f'pulse.gau.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []
    W.D1['trigger.fpga6'] = []
    W.D1['trigger.fpga7'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)
    Q.addTwoChan('A1', [1, 2], 0, ['pulse.gau.I', 'pulse.gau.Q'], 200)
    Q.addTwoChan('A1', [3, 4], 0, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 500)
    Q.add('M1', 1, 0, 'pulse.gau.M', 100)
    Q.add('M1', 2, 0, 'pulse.msmtBox.M', 400)
    Q.add('D1', 1, 0, 'trigger.fpga4', 20)
    Q.add('D1', 1, 0, 'trigger.fpga5', 20)
    return W, Q


def pulseSpecReal(module_dict):

    qubitBoxCondition = {'amp': 0.02,
                    'width': 5000,
                    'ssbFreq': 0.00,
                    'iqScale': 1,
                    'phase': 0,
                    'skewPhase': 0,
                    'rampSlope': 0.5,
                    'cutFactor': 3}

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    qPulse = pp.box(qubitBoxCondition)
    bPulse = pp.box(boxCondition)
    W.A1[f'pulse.qPulse.I'] = qPulse.smooth().I_data
    W.A1[f'pulse.qPulse.Q'] = qPulse.smooth().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.qPulse.M'] = qPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []
    W.D1['trigger.fpga6'] = []
    W.D1['trigger.fpga7'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)
    Q.addTwoChan('A1', [1, 2], 0, ['pulse.qPulse.I', 'pulse.qPulse.Q'], 200)
    Q.addTwoChan('A1', [3, 4], 0, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 6500)
    Q.add('M1', 1, 0, 'pulse.qPulse.M', 100)
    Q.add('M1', 2, 0, 'pulse.msmtBox.M', 6400)
    Q.add('D1', 1, 0, 'trigger.fpga4', 6020)
    Q.add('D1', 1, 0, 'trigger.fpga5', 6020)
    return W, Q


def piPulseTuneUpReal(module_dict, ampArray=np.linspace(-0.5, 0.5, 100)):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    for i in range(len(ampArray)):
        gPulse.amp = ampArray[i]
        W.A1[f'pulse.gau{i}.I'] = gPulse.x().I_data
        W.A1[f'pulse.gau{i}.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []
    W.D1['trigger.fpga6'] = []
    W.D1['trigger.fpga7'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)

    for i in range(len(ampArray)):
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 200)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 500)
        Q.add('M1', 1, i, 'pulse.gau.M', 100)
        Q.add('M1', 2, i, 'pulse.msmtBox.M', 400)
        Q.add('D1', 1, i, 'trigger.fpga4', 20)
        Q.add('D1', 1, i, 'trigger.fpga5', 20)

    return W, Q


def t1MsmtReal(module_dict, timeArray=np.linspace(0, 300000, 101)[:100]):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    W.A1['pulse.gau.I'] = gPulse.x().I_data
    W.A1['pulse.gau.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)

    for i in range(len(timeArray)):
        timeIndex = timeArray[i]
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau.I', f'pulse.gau.Q'], 200)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], timeIndex + 500)
        Q.add('M1', 1, i, 'pulse.gau.M', 100)
        Q.add('M1', 2, i, 'pulse.msmtBox.M', timeIndex + 400)
        Q.add('D1', 1, i, 'trigger.fpga4', timeIndex + 20)
        Q.add('D1', 1, i, 'trigger.fpga5', timeIndex + 20)

    return W, Q

def t2RMsmtReal(module_dict, timeArray=np.linspace(0, 40000, 101)[:100] + 800):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    W.A1['pulse.gau2.I'] = gPulse.x2().I_data
    W.A1['pulse.gau2.Q'] = gPulse.x2().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)

    for i in range(len(timeArray)):
        timeIndex = timeArray[i]
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau2.I', f'pulse.gau2.Q'], 200)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau2.I', f'pulse.gau2.Q'], timeIndex)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], timeIndex + 500)
        Q.add('M1', 1, i, 'pulse.gau.M', 100)
        Q.add('M1', 1, i, 'pulse.gau.M', timeIndex - 100)
        Q.add('M1', 2, i, 'pulse.msmtBox.M', timeIndex + 400)
        Q.add('D1', 1, i, 'trigger.fpga4', timeIndex + 20)
        Q.add('D1', 1, i, 'trigger.fpga5', timeIndex + 20)

    return W, Q


def t2EMsmtReal(module_dict, timeArray=np.linspace(0, 40000, 101)[:100] + 800):

    # define all waveforms we're going to use into W
    W = pp.modulesWaveformCollection(module_dict)
    gPulse = pp.gau(gauCondition)
    bPulse = pp.box(boxCondition)
    W.A1['pulse.gau.I'] = gPulse.x().I_data
    W.A1['pulse.gau.Q'] = gPulse.x().Q_data
    W.A1['pulse.gau2.I'] = gPulse.x2().I_data
    W.A1['pulse.gau2.Q'] = gPulse.x2().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = []
    W.D1['trigger.fpga4'] = []
    W.D1['trigger.fpga5'] = []

    # define Queue here
    Q = pp.modulesQueueCollection(module_dict)

    for i in range(len(timeArray)):
        timeIndex = timeArray[i]
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau2.I', f'pulse.gau2.Q'], 200)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau.I', f'pulse.gau.Q'], timeIndex/2)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau2.I', f'pulse.gau2.Q'], timeIndex)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], timeIndex + 500)
        Q.add('M1', 1, i, 'pulse.gau.M', 100)
        Q.add('M1', 1, i, 'pulse.gau.M', timeIndex/2 - 100)
        Q.add('M1', 1, i, 'pulse.gau.M', timeIndex - 100)
        Q.add('M1', 2, i, 'pulse.msmtBox.M', timeIndex + 400)
        Q.add('D1', 1, i, 'trigger.fpga4', timeIndex + 20)
        Q.add('D1', 1, i, 'trigger.fpga5', timeIndex + 20)

    return W, Q