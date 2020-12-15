import numpy as np
import matplotlib.pyplot as plt
import package_sequenceGenerate as sG
import time
# from package_PathWave import ApplicationConfig as config


# wait to be deleted
class config:
    " Defines module descriptors, configuration options and names of HVI engines, actions, triggers"

    def __init__(self):
        # Configuration options
        self.hardwareSimulated = False
        # Define options to open the instruments. Complete option list can be found in the SD1 user guide
        # Here all channels number start from 1
        self.options = 'channelNumbering=keysight'
        # Define names of HVI engines, actions, triggers
        self.boards = [['M3202A', 1, 2, self.options, "A1"],  # model, chassis, slot, options, name
                       ['M3202A', 1, 3, self.options, "A2"],
                       ['M3202A', 1, 4, self.options, "A3"],
                       ['M3102A', 1, 5, self.options, "D1"],
                       ['M3102A', 1, 6, self.options, "D2"],
                       ['M3201A', 1, 7, self.options, "M1"],
                       ['M3201A', 1, 8, self.options, "M2"],
                       ['M3201A', 1, 9, self.options, "M3"]]
        self.fpTriggerName = "triggerFP"  # for future develop
        # here define action name
        self.awgTriggerActionName = "triggerAWG"  # 1, 2, 3, 4
        self.digTriggerActionName = "triggerDig"  # 1, 2, 3, 4
        self.fpgaTriggerActionName = "triggerFPGA"  # 0, 1, 2, 3, 4, 5, 6, 7
        self.awgResetPhaseActionName = "resetAWG"  # 1, 2, 3, 4

    class ModuleDescriptor:
        "Descriptor for module objects, the class is only for information store."

        def __init__(self, modelNumber, chassisNumber, slotNumber, options, engineName):
            self.modelNumber = modelNumber
            self.chassisNumber = chassisNumber
            self.slotNumber = slotNumber
            self.options = options
            self.engineName = engineName

    class Module:
        "Class defining a modular instrument object and its properties"
        # We will add more descriptions if needed.
        # hviSupport: one digitizer (slot 6) card didn't support hvi

        def __init__(self, moduleName, insturmentObject, hviSupport):
            self.moduleName = moduleName
            self.instrument = insturmentObject
            self.hviSupport = hviSupport


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
        W.A1[f'pulse.gau{i}.I'] = gPulse.x().I_data
        W.A1[f'pulse.gau{i}.Q'] = gPulse.x().Q_data
    W.A1['pulse.msmtBox.I'] = bPulse.smooth().I_data
    W.A1['pulse.msmtBox.Q'] = bPulse.smooth().Q_data
    W.M1['pulse.gau.M'] = gPulse.marker().I_data
    W.M1['pulse.msmtBox.M'] = bPulse.marker().I_data
    W.D1['trigger.dig'] = [1, 2]
    W.D1['trigger.fpga7'] = [1]

    # define Queue here
    Q = sG.queueModulesCollection(module_dict)
    for i in range(len(ampArray)):
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 10)
        Q.addTwoChan('A1', [1, 2], i, [f'pulse.gau{i}.I', f'pulse.gau{i}.Q'], 50)
        Q.addTwoChan('A1', [3, 4], i, ['pulse.msmtBox.I', 'pulse.msmtBox.Q'], 110)
        Q.add('M1', 1, i, 'pulse.gau.M', 0)
        Q.add('M1', 2, i, 'pulse.msmtBox.M', 100)
        Q.add('D1', 1, i, 'trigger.dig', 100)
        Q.add('D1', 1, i, 'trigger.fpga7', 150)
    return W, Q


pulse_general_dict = {'relaxingTime': 300}  # float; 0.01us


if __name__ == '__main__':
    chanNum = 4
    start = time.time()
    W, Q = piPulseTuneUp()

    xdata = ampArrayEg
    wUpload = sG.waveformModulesCollection(module_dict)
    for module in module_dict.keys():
        index = 0
        w_dict = {}
        for pulseName, waveformArray in getattr(W, module).items():
            w_dict[pulseName] = [index, waveformArray]
            index += 1
        setattr(wUpload, module, w_dict)

        for chan in range(1, chanNum + 1):
            for seqOrder, seqInfo in getattr(getattr(Q, module), f'chan{chan}').items():
                trigger = 1
                for singlePulse in seqInfo:
                    startDelay = singlePulse[1] if trigger == 1 else 0
                    # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
                    print([chan, w_dict[singlePulse[0]][0], 1, 0, 1, 0])  # singlePulse = ['pulseName', timeDelay] # Attention: need to double check here
                    trigger = 0

    sequencer = kthvi.Sequencer('seqName', sys_def)
    for seqOrder in range(len(xdata)):
        # first is 10 ns, then is relaxing time
        delay = 10 if seqOrder == 0 else int(pulse_general_dict['relaxingTime'] * 1e3)
        syncBlock = sequencer.sync_sequence.add_sync_multi_sequence_block(f"syncBlock{seqOrder}", delay)

        for module in module_dict.keys():
            seq = syncBlock.sequence[module]

            time_sort = {}  # First clean up the time slot for different instructions
            for chan in range(1, chanNum + 1):
                try:
                    pulseInEachSeq = getattr(getattr(Q, module), f'chan{chan}')[str(seqOrder)]
                    for singlePulse in pulseInEachSeq:
                        if str(singlePulse[1]) not in time_sort.keys():
                            time_sort[str(singlePulse[1])] = []
                        if 'pulse' in singlePulse[0]:
                            time_sort[str(singlePulse[1])] += [config().awgTriggerActionName + str(chan)]
                        elif 'trigger.dig' in singlePulse[0]:
                            time_sort[str(singlePulse[1])] += [config().digTriggerActionName + str(chan)]
                        elif 'trigger.fpga' in singlePulse[0]:
                            time_sort[str(singlePulse[1])] += [config().fpgaTriggerActionName + singlePulse[0][-1]]
                except KeyError:
                    pass
            print(time_sort)
            time_ = 0
            for timeIndex, actionList in time_sort.items():
                time_ = int(timeIndex) - time_
                aList = [seq.engine.actions[a_] for a_ in actionList]
                instru = seq.add_instruction(f"block{seqOrder}time{timeIndex}", int(time_), seq.instruction_set.action_execute.id)
                instru.set_parameter(seq.instuction_set.action_excute.action, aList)




    # # print(wUpload.A1)
    # index = 0
    # wf_index = {}
    # for k in W.A1.keys():
    #     wf_index[k] = index
    #     index += 1
    # for i in range(1, 5):
    #     for k, v in getattr(Q.A1, f'chan{i}').items():
    #         trigger = 1  # first pulse always trigger
    #         for p in v:
    #             # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
    #             # print([i, wf_index[p[0]], trigger, 0, 1, 0])
    #             trigger = 0

    # sequencer = kthvi.Sequencer('seqName', sys_def)
    # xdata = ampArrayEg
    # for i in range(len(xdata)):
    #     # first is 10 ns, then is relaxing time
    #     delay = 10 if i == 0 else int(general_dict['relaxingTime'] * 1e3)
    #     syncBlock = sequencer.sync_sequence.add_sync_multi_sequence_block(f"syncBlock{i}", delay)
    #     seq = syncBlock.sequence[module]
    #     actionList = [seq.engine.actions[config.awgTriggerActionName + str(c)] for c in range(1, 3)]
    #     instruction = seq.add_instruction(f"trigger12_block{i}", p[1])
    #     instruction.set_parameter(seq.instuction_set.action_excute.action, actionList)
    print(time.time() - start)
