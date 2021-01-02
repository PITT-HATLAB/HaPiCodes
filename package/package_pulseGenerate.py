import numpy as np
import matplotlib.pyplot as plt
import package_sequenceGenerate as sG
import time
from package_PathWave import ApplicationConfig as config
import package_PathWave as PW
from collections import OrderedDict 
import json
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
sys.path.append('C:\HatCode\PXIe')
import keysightSD1
import warnings
import keysight_hvi as kthvi

with open(r"sysInfo.json") as file_:
    info = json.load(file_)
triggerAwgDigDelay = info['sysConstants']['triggerAwgDigDelay']

gauCondition = {'amp': 0.1,
                'ssbFreq': 0.0,
                'iqScale': 1,
                'phase': 0,
                'skewPhase': 0,
                'sigma': 10,
                'sigmaMulti': 8,
                'dragFactor': 0}

boxCondition = {'amp': 0.5,
                'width': 50,
                'ssbFreq': 0.0,
                'iqScale': 1,
                'phase': 0,
                'skewPhase': 0,
                'rampSlope': 0.5,
                'cutFactor': 3}

pulse_general_dict = {'relaxingTime': 10}  # float; precision: 0.01us; relax after the start of the firt pulse

ampArrayEg = np.linspace(0.5, 1, 1)


def piPulseTuneUp(ampArray=ampArrayEg):

    # define all waveforms we're going to use into W
    W = sG.waveformModulesCollection(module_dict)
    gPulse = sG.gau(gauCondition)
    bPulse = sG.box(boxCondition)
    for i in range(len(ampArray)):
        gPulse.amp = ampArray[i]
        W.A1[f'pulse.gau{i}.I'] = gPulse.x().I_data
        W.A1[f'pulse.gau{i}.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.markerHalf().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.markerHalf().I_data
    W.D1['trigger.dig'] = [1, 2]
    W.D1['trigger.fpga7'] = [1]

    # define Queue here
    Q = sG.queueModulesCollection(module_dict)
    for i in range(len(ampArray)):
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 10)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 200)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 600)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 100)
        Q.add('M1', 1, i, 'pulse.msmtBox.M', 10)
        Q.add('M1', 1, i, 'pulse.msmtBox.M', 200)
        Q.add('D1', 1, i, 'trigger.dig', triggerAwgDigDelay)
        Q.add('D1', 2, i, 'trigger.dig', triggerAwgDigDelay)
        Q.add('D1', 3, i, 'trigger.dig', triggerAwgDigDelay)
        Q.add('D1', 4, i, 'trigger.dig', triggerAwgDigDelay)
        # Q.add('D1', 1, i, 'trigger.fpga7', 150)
    return W, Q


if __name__ == '__main__':
    pointPerCycle = 200
    cycles = 1
    start = time.time()
    module_dict = PW.open_modules()
    for module in module_dict:
        if module_dict[module].instrument.getProductName() != "M3102A":
            PW.configAWG(module_dict[module])
        # else: 
        #     PW.configDig(module_dict[module], fpgaName="C:\\PXI_FPGA\\Projects\\Origional\\Origional.data\\bin\\Origional_2020-12-01T19_03_56\\Origional.k7z", manualReloadFPGA=0)

    chanNum = 4
    W, Q = piPulseTuneUp()
    print()
    for i in range(1 , 5):
        module_dict['D1'].instrument.DAQconfig(i, pointPerCycle, cycles, 0, 1)

    xdata = ampArrayEg
    PW.uploadAndQueueWaveform(module_dict, W, Q)
    hvi = PW.defineAndCompileHVI(module_dict, Q, xdata, pulse_general_dict)

    dataReceive = PW.digAcceptData(module_dict['D1'], hvi, pointPerCycle, cycles, chan='1111', timeout=1000)
    for engine_name in module_dict:
        module_dict[engine_name].instrument.close()
    print("Modules closed\n")
    # for j in range(1, 5):
    #     plt.figure()
        # dataRescale = dataReceive[str(j)].reshape(cycles, pointPerCycle)
        # for i in range(cycles):
        #     plt.plot(np.arange(pointPerCycle)*2, dataRescale[i], label=f'cycle{i}')
        # plt.legend()
    plt.figure()
    for j in range(1, 5):
        dataRescale = dataReceive[str(j)]/2**15 + j
        plt.plot(np.arange(len(dataRescale)) * 2, dataRescale, label=f'chan{j}')
    plt.legend()
    print(time.time() - start)
    plt.show()

    '''
    May need in the future

    # syncBlock1 = sequencer.sync_sequence.add_sync_multi_sequence_block("syncBlock1", 30)
    # seqA1 = syncBlock1.sequences['A1']
    # aList = [seqA1.engine.actions["triggerAWG1"]]
    # instruA1_1 = seqA1.add_instruction("block1TriggerAWG", 0, seqA1.instruction_set.action_execute.id)
    # instruA1_1.set_parameter(seqA1.instruction_set.action_execute.action, aList)

    # seqD1 = syncBlock1.sequences['D1']
    # dList = [seqD1.engine.actions['triggerDig1']]
    # instruD1_1 = seqD1.add_instruction('block1TriggerDig', 0, seqD1.instruction_set.action_execute.id)
    # instruD1_1.set_parameter(seqD1.instruction_set.action_execute.action, dList)


    # syncBlock2 = sequencer.sync_sequence.add_sync_multi_sequence_block("syncBlock2", 10030)
    # seq2A1 = syncBlock2.sequences['A1']
    # a2List = [seq2A1.engine.actions["triggerAWG1"]]
    # instruA1_2 = seq2A1.add_instruction("block2TriggerAWG", 0, seq2A1.instruction_set.action_execute.id)
    # instruA1_2.set_parameter(seq2A1.instruction_set.action_execute.action, a2List)

    # seq2D1 = syncBlock2.sequences['D1']
    # d2List = [seq2D1.engine.actions['triggerDig1']]
    # instruD1_2 = seq2D1.add_instruction('block2TriggerDig', 0, seq2D1.instruction_set.action_execute.id)
    # instruD1_2.set_parameter(seq2D1.instruction_set.action_execute.action, d2List)
    '''