import numpy as np
import matplotlib.pyplot as plt
import package_sequenceGenerate as sG
import time

gauCondition = {'amp': 0.1,
                'ssbFreq': 0.1,
                'iqScale': 1,
                'phase': 0,
                'skewPhase': 0,
                'sigma': 20,
                'sigmaMulti': 6,
                'dragFactor': 0}

boxCondition = {'amp': 0.5,
                'width': 200,
                'ssbFreq': 0.0,
                'iqScale': 1,
                'phase': 0,
                'skewPhase': 0,
                'rampSlope': 0.5,
                'cutFactor': 3}

module_dict = {"A1": 0,
               "M1": 0,
               "D1": 0,
               "A2": 0,
               "A3": 0,
               "M2": 0,
               "M3": 0}

ampArrayEg = np.linspace(0, 1, 11)[:10]


def piPulseTuneUp(ampArray=ampArrayEg):

    # define all waveforms we're going to use into W
    W = sG.waveformModulesCollection(module_dict)
    gPulse = sG.gau(gauCondition)
    bPulse = sG.box(boxCondition)
    for i in range(len(ampArray)):
        gPulse.amp = ampArray[i]
        W.A1[f'gau{i}.I'] = gPulse.x().I_data
        W.A1[f'gau{i}.Q'] = gPulse.x().Q_data
    W.A1['msmtBox.I'] = bPulse.smooth().I_data
    W.A1['msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['gau.M'] = gPulse.marker().I_data
    W.M1['msmtBox.M'] = bPulse.marker().I_data

    # define Queue here
    Q = sG.queueModulesCollection(module_dict)
    for i in range(len(ampArray)):
        Q.add('A1', 1, i, f'gau{i}.I', 10)
        Q.add('A1', 1, i, f'gau{i}.I', 50)
        Q.add('A1', 2, i, f'gau{i}.Q', 10)
        Q.add('A1', 2, i, f'gau{i}.Q', 50)
        Q.add('A1', 3, i, 'msmtBox.I', 110)
        Q.add('A1', 4, i, 'msmtBox.Q', 110)
        Q.add('M1', 1, i, 'gau.M', 0)
        Q.add('M1', 2, i, 'msmtBox.M', 100)
        Q.add('D1', 1, i, 'data!', 100)
    return W, Q


general_dict = {'relaxingTime': 300}  # float; 0.01us


if __name__ == '__main__':
    start = time.time()
    W, Q = piPulseTuneUp()
    print(Q.A1.chan1)
    print(Q.D1.chan1)
    print(Q.M1.chan1)

    index = 0
    wf_index = {}
    for k in W.A1.keys():
        wf_index[k] = index
        index += 1
    for i in range(1, 5):
        for k, v in getattr(Q.A1, f'chan{i}').items():
            trigger = 1  # first pulse always trigger
            for p in v:
                # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
                print([i, wf_index[p[0]], trigger, 0, 1, 0])
                trigger = 0

    sequencer = kthvi.Sequencer('seqName', sys_def)
    xdata = ampArrayEg
    for i in range(len(xdata)):
        # first is 10 ns, then is relaxing time
        delay = 10 if i == 0 else int(general_dict['relaxingTime'] * 1e3)
        syncBlock = sequencer.sync_sequence.add_sync_multi_sequence_block(f"syncBlock{i}", delay)
        seq = syncBlock.sequence[module]
        actionList = [seq.engine.actions[config.awgTriggerActionName + str(c)] for c in range(1, 3)]
        instruction = seq.add_instruction(f"trigger12_block{i}", p[1])
        instruction.set_parameter(seq.instuction_set.action_excute.action, actionList)
        
        

    print(time.time() - start)
