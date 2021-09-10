import warnings
from typing import List, Callable, Union, Optional, Dict
import pathlib
import yaml
import numpy as np
from collections import OrderedDict
from HaPiCodes.sd1_api import keysightSD1
import keysight_hvi as kthvi
from HaPiCodes.sd1_api.SD1AddOns import AIN, AOU
import HaPiCodes.pathwave
pathwave_path = pathlib.Path(HaPiCodes.pathwave.__path__[0])
with open(str(pathwave_path)+'/sysInfo.yaml', 'r') as file_:
    sysInfoDict = yaml.safe_load(file_)



class ApplicationConfig:
    """ Defines module descriptors, configuration options and names of HVI engines, actions, triggers
    change self.boards after changing boards on the chassis
    """
    def __init__(self, boards: Union[str, List]="user_defined"):
        # Configuration options
        self.hardwareSimulated = False
        # Define options to open the instruments. Complete option list can be found in the SD1 user guide
        # Here all channels number start from 1
        # Define names of HVI engines, actions, triggers
        if boards == "default":
            self.boards = sysInfoDict["defaultBoards"] # use default boards
        elif boards == "user_defined":
            try:
                with open(str(pathwave_path) + '\\user_defined_boards.yaml', 'r') as file_:
                    user_defined_boards = yaml.safe_load(file_)
                    self.boards = user_defined_boards["boards"]
            except FileNotFoundError:
                raise FileNotFoundError("Cannot find user_defined_boards.yaml. Please define the boards using "
                                        "HaPiCodes.pathwave.HVIConfig.defineBoards(). See defaultBoards in "
                                        "pathwave/sysInfo.yaml for example.")
        else:
            self.boards =boards # user defined boards
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
            self.chanNum = 4

def open_modules(boards: Union[str, List]="user_defined"):
    """
    Opens and creates all the necessary instrument objects.
    Returns a dictionary of module objects whose keys are the HVI engine names.
    Please check SD1 3.x User Manual for options to open the instrument objects
    """
    config = ApplicationConfig(boards)
    # Initialize output variables
    numModules = 0
    module_dict = {}  # dictionary of modules

    # Append all module from config file into one list.
    allModulesList = []
    for board in config.boards:
        allModulesList.append(config.ModuleDescriptor(*board))

    # Open SD1 instrument objects
    for descriptor in allModulesList:
        if descriptor.modelNumber == "M3102A":
            instrObj = AIN()
        else:
            instrObj = AOU()
            instrObj.waveformFlush()
        instObjOptions = descriptor.options + \
            ',simulate=true' if config.hardwareSimulated else descriptor.options
        id = instrObj.openWithOptions(
            descriptor.modelNumber, descriptor.chassisNumber, descriptor.slotNumber, instObjOptions, descriptor.engineName)
        if id < 0:
            raise Exception("Error opening instrument in chassis: {}, slot: {}! Error code: {} - {}. Exiting...".format(
                descriptor.chassisNumber, descriptor.slotNumber, id, keysightSD1.SD_Error.getErrorMessage(id)))

        hviInfo = instrObj.getOptions("hvi")
        if hviInfo == "HVI":
            hviSupport = 1
        elif hviInfo == "none":
            hviSupport = 0
        else:
            raise Exception("PXI module in chassis {}, slot {} returned hvi information: {}, which is incorrect. Exiting... ".format(
                instrObj.getChassis(), instrObj.getSlot(), hviInfo))

        module_dict[descriptor.engineName] = config.Module(descriptor.engineName, instrObj, hviSupport)
        numModules += 1

    return module_dict



def define_hw_platform(sysDef, chassisList, M9031Descriptors, pxiSyncTriggerResources):
    """
    Define HW platform: chassis, interconnections, PXI trigger resources, synchronization, HVI clocks
    """
    config = ApplicationConfig()

    # Add chassis resources
    for chassisNumber in chassisList:
        if config.hardwareSimulated:
            sysDef.chassis.add_with_options(
                chassisNumber, 'Simulate=True,DriverSetup=model=M9018B,NoDriver=True')
        else:
            sysDef.chassis.add(chassisNumber)

    # Add M9031 modules for multi-chassis setups
    if M9031Descriptors:
        pass  # TODO: for future multi-chassis communication

    # Assign the defined PXI trigger resources
    sysDef.sync_resources = pxiSyncTriggerResources

    # Assign clock frequencies that are outside the set of the clock frequencies of each HVI engine
    # Use the code line below if you want the application to be in sync with the 10 MHz clock
    sysDef.non_hvi_core_clocks = [10e6]


def define_hvi_engines(sysDef, module_dict):
    """
    Define all the HVI engines to be included in the HVI
    """
    # For each instrument to be used in the HVI application add its HVI Engine to the HVI Engine Collection
    for engineName in module_dict.keys():
        if module_dict[engineName].hviSupport:
            sysDef.engines.add(
                module_dict[engineName].instrument.hvi.engines.main_engine, engineName)
        else:
            warnings.warn("Engine " + engineName +
                          " didn't add into HVI, hardware is not supported (HV1 option)")


def define_hvi_actions(sysDef, module_dict):
    """ Defines AWG trigger actions for each module, to be executed by the "action execute" instruction in the HVI sequence
    Create a list of AWG trigger actions for each AWG module. The list depends on the number of channels """

    # Previously defined resource names
    config = ApplicationConfig()

    # For each AWG, define the list of HVI Actions to be executed and add such list to its own HVI Action Collection
    for engineName in module_dict.keys():
        if module_dict[engineName].hviSupport:
            for ch_index in range(1, 5):
                # Actions need to be added to the engine's action list so that they can be executed
                if module_dict[engineName].instrument.getProductName() == "M3102A":
                    action_name = config.digTriggerActionName + \
                        str(ch_index)  # arbitrary user-defined name
                    instrument_action = "daq{}_trigger".format(
                        ch_index)  # name decided by instrument API
                    action_id = getattr(
                        module_dict[engineName].instrument.hvi.actions, instrument_action)
                    sysDef.engines[engineName].actions.add(
                        action_id, action_name)
                else:  # M3201A and M3202A
                    action_name = config.awgTriggerActionName + \
                        str(ch_index)  # arbitrary user-defined name
                    instrument_action = "awg{}_trigger".format(
                        ch_index)  # name decided by instrument API
                    action_id = getattr(
                        module_dict[engineName].instrument.hvi.actions, instrument_action)
                    sysDef.engines[engineName].actions.add(
                        action_id, action_name)

                    action_name = config.awgResetPhaseActionName + \
                        str(ch_index)  # arbitrary user-defined name
                    instrument_action = "ch{}_reset_phase".format(
                        ch_index)  # name decided by instrument API
                    action_id = getattr(
                        module_dict[engineName].instrument.hvi.actions, instrument_action)
                    sysDef.engines[engineName].actions.add(
                        action_id, action_name)

            for i in range(8):
                action_name = config.fpgaTriggerActionName + str(i)
                instrument_action = f"fpga_user_{i}"
                action_id = getattr(
                    module_dict[engineName].instrument.hvi.actions, instrument_action)
                sysDef.engines[engineName].actions.add(action_id, action_name)


def define_hvi_triggers(sysDef, module_dict):
    " Defines and configure the FP trigger output of each AWG "
    # Previously defined resources
    config = ApplicationConfig()

    # Add to the HVI Trigger Collection of each HVI Engine the FP Trigger object of that same instrument
    for engineName in module_dict.keys():
        if module_dict[engineName].hviSupport:
            fpTriggerId = module_dict[engineName].instrument.hvi.triggers.front_panel_1
            fpTrigger = sysDef.engines[engineName].triggers.add(
                fpTriggerId, config.fpTriggerName)
            # Configure FP trigger in each hvi.engines[index]
            fpTrigger.config.direction = kthvi.Direction.OUTPUT
            fpTrigger.config.polarity = kthvi.Polarity.ACTIVE_HIGH
            fpTrigger.config.sync_mode = kthvi.SyncMode.IMMEDIATE
            fpTrigger.config.hw_routing_delay = 0
            fpTrigger.config.trigger_mode = kthvi.TriggerMode.LEVEL
            # NOTE: FP trigger pulse length is defined by the HVI Statements that control FP Trigger ON/OFF


preSetPxiSyncTriggerResources = [kthvi.TriggerResourceId.PXI_TRIGGER0,
                                 kthvi.TriggerResourceId.PXI_TRIGGER1,
                                 kthvi.TriggerResourceId.PXI_TRIGGER2,
                                 kthvi.TriggerResourceId.PXI_TRIGGER3]


def define_hvi_resources(sysDef, module_dict, pxiSyncTriggerResources=preSetPxiSyncTriggerResources):
    """
    Configures all the necessary resources for the HVI application to execute: HW platform, engines, actions, triggers, etc.
    """
    chassisList = [1]
    M9031Descriptors = []

    # Define HW platform: chassis, interconnections, PXI trigger resources, synchronization, HVI clocks
    define_hw_platform(sysDef, chassisList, M9031Descriptors,
                       pxiSyncTriggerResources)

    # Define all the HVI engines to be included in the HVI
    define_hvi_engines(sysDef, module_dict)

    # Define list of actions to be executed
    define_hvi_actions(sysDef, module_dict)

    # Defines the trigger resources
    define_hvi_triggers(sysDef, module_dict)




def define_instruction_compile_hvi(module_dict: dict, Q, pulse_general_dict: dict, subbuffer_used=0):
    """from Q define all instructions in all modules, and return compiled hvi.
    Args:
        module_dict (dict): all modules in dictionary; generated from open_module()
        Q (modulesQueueCollection): queueCollections in all modules.
        pulse_general_dict (dict): the general setting for all pulses (will update later)
        TODO: now relaxing time is fixed for different index, may add more variable later.
    Returns:
        hvi (sequencer.compile()): the compiled hvi can be ran.
    """
    config = ApplicationConfig()
    module_dict_temp = module_dict
    sys_def = kthvi.SystemDefinition("systemInfo")
    define_hvi_resources(sys_def, module_dict_temp)
    sequencer = kthvi.Sequencer('seqName', sys_def)
    primaryEngine = list(module_dict.keys())[0]

    subBufferReg1 = sequencer.sync_sequence.scopes[primaryEngine].registers.add(primaryEngine+"subBufferReg1", kthvi.RegisterSize.SHORT)
    subBufferReg1.initial_value = pulse_general_dict['avgNum']
    subBufferReg2 = sequencer.sync_sequence.scopes[primaryEngine].registers.add(primaryEngine+"subBufferReg2", kthvi.RegisterSize.SHORT)
    subBufferReg2.initial_value = 0

    SYNC_WHILE_LOOP_ITERATIONS = pulse_general_dict['avgNum']

    syncWhileCondition = kthvi.Condition.register_comparison(subBufferReg1, kthvi.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO, subBufferReg2)
    syncWhile = sequencer.sync_sequence.add_sync_while("subBufferSync", 100, syncWhileCondition)

    #################### reset phase block ###################
    # syncBlock = syncWhile.sync_sequence.add_sync_multi_sequence_block(f"resetAWG", 200)
    # for module in module_dict:
    #     if module_dict[module].instrument.getProductName() != 'M3102A':
    #         seq = syncBlock.sequences[module]
    #         aList = [seq.engine.actions[a_] for a_ in [config.awgResetPhaseActionName + str(i) for i in range(1, 5)]]
    #         instru = seq.add_instruction(f"resetPhaseAllAWG", 100, seq.instruction_set.action_execute.id)
    #         instru.set_parameter(seq.instruction_set.action_execute.action, aList)

    timeMax = 0
    # syncBlock0 = syncWhile.sync_sequence.add_sync_multi_sequence_block(f"syncBlock0000", int(50 * 1e3))
    for seqOrder in range(Q.maxIndexNum + 1):  # the maxIndexNum should be the len(xdata)-1
        delay = int(pulse_general_dict['relaxingTime'] * 1e3) + 60
        syncBlock = syncWhile.sync_sequence.add_sync_multi_sequence_block(f"syncBlock{seqOrder}", delay)
        timeMax = 0
        for module in module_dict_temp.keys():
            chanNum = module_dict_temp[module].chanNum
            seq = syncBlock.sequences[module]

            time_sort = {}  # First clean up the time slot for different instructions
            pulseNumList_ = []
            for chan in range(1, chanNum + 1):
                try:
                    pulseInEachSeq = getattr(getattr(Q, module), f'ch{chan}')[str(seqOrder)]
                    pulseNumList_.append(len(pulseInEachSeq))
                    for singlePulse in pulseInEachSeq:
                        singlePulse[1] = np.round(singlePulse[1])
                        if int(singlePulse[1]) not in time_sort.keys():
                            time_sort[int(singlePulse[1])] = []
                        if singlePulse[0][:6] == 'pulse.':
                            time_sort[int(singlePulse[1])] += [config.awgTriggerActionName + str(chan)]
                        elif singlePulse[0][:11] == 'trigger.dig':
                            time_sort[int(singlePulse[1])] += [config.digTriggerActionName + str(chan)]
                        elif singlePulse[0][:12] == 'trigger.fpga':
                            time_sort[int(singlePulse[1])] += [config.fpgaTriggerActionName + singlePulse[0][-1]]
                        else:
                            raise KeyError('pulse name is wrong, please check')
                except KeyError:
                    pass
            time_sort_order = OrderedDict(sorted(time_sort.items()))
            time_ = 0
            for timeIndex, actionList in time_sort_order.items():
                if module_dict_temp[module].instrument.getProductName() == 'M3201A':
                    timeIndex = timeIndex - sysInfoDict['sysConstants']['M3201A_M3202A_Delay']
                time_ = int(timeIndex) - time_
                if time_ < 0:
                    raise ValueError(f'{module}:, block{seqOrder}time{timeIndex}: time is smaller than 0, please fix the time problem')
                if subbuffer_used:
                    for chan_ in range(1, 5):
                        try:
                            actionList.remove(config.digTriggerActionName + str(chan_))
                        except ValueError:
                            pass
                if actionList == []:
                    pass
                else:
                    # print(module, f"block{seqOrder}time{timeIndex}", time_, actionList)
                    aList = [seq.engine.actions[a_] for a_ in actionList]
                    instru = seq.add_instruction(f"block{seqOrder}time{timeIndex}", time_, seq.instruction_set.action_execute.id)
                    instru.set_parameter(seq.instruction_set.action_execute.action.id, aList)
                    time_ = timeIndex
                    timeMax = np.max([timeMax, timeIndex])

            avg_pulse_density = np.array(pulseNumList_)/(timeMax + delay)*1e3 #average pulse number per us
            if len(avg_pulse_density) > 0  and max(avg_pulse_density) > 1.45:
                # I found a magic threshold around 1.47 by testing, actually I'm not quite sure what the real limit is.
                warnings.warn(f"It seems that there are a lot of triggers during a short period of time in sequence "
                              f"{seqOrder}, combine pulses into one long pulse is recommended.")


    syncBlock = syncWhile.sync_sequence.add_sync_multi_sequence_block(f"syncBlockSub", 300)
    seq = syncBlock.sequences[primaryEngine]
    reg = sequencer.sync_sequence.scopes[primaryEngine].registers[primaryEngine+"subBufferReg2"]
    regIncInstru = seq.add_instruction('regIncreaseSubBuffer', 300, seq.instruction_set.add.id)
    regIncInstru.set_parameter(seq.instruction_set.add.destination.id, reg)
    regIncInstru.set_parameter(seq.instruction_set.add.left_operand.id, reg)
    regIncInstru.set_parameter(seq.instruction_set.add.right_operand.id, 1)

    if subbuffer_used:
        lastBlock = sequencer.sync_sequence.add_sync_multi_sequence_block(f"readSub", int(pulse_general_dict['relaxingTime'] * 1e3))
        for dig_ , triggers_ in Q.dig_trig_num_dict.items():
            max_trigger_num = np.max(list(triggers_.values()))
            if max_trigger_num > 0:
                subBufferWaitReg = sequencer.sync_sequence.scopes[dig_].registers.add(dig_ + "subBufferWaitReg",
                                                                                      kthvi.RegisterSize.SHORT)
                subBufferWaitReg.initial_value = 33000
                seq = lastBlock.sequences[dig_]
                aList = [seq.engine.actions[a_] for a_ in [config.digTriggerActionName + str(i) for i in range(1, 5)]]
                instru = seq.add_instruction(f"readSubInstru", 100, seq.instruction_set.action_execute.id)
                instru.set_parameter(seq.instruction_set.action_execute.action.id, aList)
                seq.add_wait_time("waitSubBuffer", 100, subBufferWaitReg)


    hvi = sequencer.compile()
    print("HVI Compiled")
    print("This HVI application needs to reserve {} PXI trigger resources to execute".format(len(hvi.compile_status.sync_resources)))
    hvi.load_to_hw()
    print("HVI Loaded to HW")
    return hvi


def definePulseAndUpload(pulseFunc, module_dict, pulse_general_dict, subbuffer_used=0):
    W, Q = pulseFunc(module_dict)

    for module in module_dict:
        if module_dict[module].instrument.getProductName() != "M3102A":
            w_index = module_dict[module].instrument.AWGuploadWaveform(getattr(W, module))
            module_dict[module].instrument.AWGqueueAllChanWaveform(w_index, getattr(Q, module))
    hvi = define_instruction_compile_hvi(module_dict, Q, pulse_general_dict, subbuffer_used=subbuffer_used)
    return W, Q, hvi


def defineBoards(boards: List):
    with open(str(pathwave_path) + '\\user_defined_boards.yaml', "w") as f:
        boards = {"boards": boards}
        yaml.safe_dump(boards, f, sort_keys=0, default_flow_style=None)

def getUserDefinedBoards():
    with open(str(pathwave_path) + '\\user_defined_boards.yaml', "r") as f:
        boards = yaml.safe_load(f)["boards"]
    return boards

# --------------------------------------- functions for flexible usage------------------------------------
def closeModule(module_dict):
    for engine_name in module_dict:
        module_dict[engine_name].instrument.close()
    print("Modules closed\n")


def releaseHviAndCloseModule(hvi, module_dict):
    hvi.release_hw()
    print("Releasing HW...")
    closeModule(module_dict)


def uploadPulseAndQueue(W, Q, module_dict, pulse_general_dict, subbuffer_used=1):
    for module_name, module in module_dict.items():
        if module.instrument.getProductName() != "M3102A":
            w_index = module.instrument.AWGuploadWaveform(getattr(W, module_name))
            module.instrument.AWGqueueAllChanWaveform(w_index, getattr(Q, module_name))
    hvi = define_instruction_compile_hvi(module_dict, Q, pulse_general_dict, subbuffer_used)
    hvi = hvi
    return hvi

def digReceiveData(digModule, hvi, pointPerCycle, cycles, chan="1111", timeout=10, subbuffer_used = False):
    data_receive = {}
    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            data_receive[str(chanNum)] = np.zeros(pointPerCycle * cycles)
        else:
            data_receive[str(chanNum)] = []
        chanNum -= 1
    digModule.instrument.DAQstartMultiple(int(chan, 2))

    print('hvi is running')
    if subbuffer_used:
        for i in range(cycles):
            print(f"hvi running {i+1}/{cycles}")
            hvi.run(hvi.no_timeout)

    else:
        hvi.run(hvi.no_timeout)

    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            print('receive data from channel' + str(chanNum))
            data_receive[str(chanNum)] = digModule.instrument.DAQreadArray(chanNum, timeout)
        chanNum -= 1
    hvi.stop()

    return data_receive

#----------------deprecated---------------------------------------------------------------------------------------------
def autoConfigAllDAQ(module_dict, Q, avg_num, points_per_cycle_dict: Optional[Dict[str,List[int]]] = None,
                  triggerMode = keysightSD1.SD_TriggerModes.SWHVITRIG):
    if points_per_cycle_dict is None:
        points_per_cycle_dict = {}
    max_trig_num_per_exp = 0
    for trig_nums in Q.dig_trig_num_dict.values():
        max_in_module = np.max(np.fromiter(trig_nums.values(), dtype=int))
        max_trig_num_per_exp = np.max((max_in_module, max_trig_num_per_exp))
    nAvgPerHVI_list = []
    for dig_name, trig_nums in Q.dig_trig_num_dict.items():
        nAvgPerHVI_ = module_dict[dig_name].instrument.DAQAutoConfig(trig_nums, avg_num, max_trig_num_per_exp,
                                                                    points_per_cycle_dict.get(dig_name), triggerMode)
        if nAvgPerHVI_ is not None:
            nAvgPerHVI_list.append(nAvgPerHVI_)
    if len(np.unique(nAvgPerHVI_list)) == 1:
        return nAvgPerHVI_list[0]
    else:
        raise ValueError("Error automatic configuring all DAQ")



def runExperiment(module_dict, hvi, Q, timeout=10):
    # TODO: module_dict and subbuffer_used should be removed if this is a member function fo PXIModules
    data_receive = {}
    cyc_list = []
    subbuff_used_list = []
    for dig_name in Q.dig_trig_num_dict:
        dig_module = module_dict[dig_name].instrument
        data_receive[dig_name] = {}
        ch_mask = ""
        for ch, (ppc, cyc) in dig_module.DAQ_config_dict.items():
            if (ppc!=0) and (cyc != 0):
                cyc_list.append(cyc)
                data_receive[dig_name][ch] = np.zeros(ppc * cyc)
                ch_mask = '1' + ch_mask
            else:
                ch_mask = '0' + ch_mask
        module_dict[dig_name].instrument.DAQstartMultiple(int(ch_mask, 2))
        subbuff_used_list.append(module_dict[dig_name].instrument.subbuffer_used)

    if len(np.unique(cyc_list)) != 1:
        raise ValueError("All modules must have same number of DAQ cycles")
    if len(subbuff_used_list) != 1:
        raise NotImplementedError("Digitizer modules with different FPGAs is not supported at this point")

    cycles = cyc_list[0]
    subbuffer_used = subbuff_used_list[0]

    print('hvi is running')
    if subbuffer_used:
        for i in range(cycles):
            print(f"hvi running {i+1}/{cycles}")
            hvi.run(hvi.no_timeout)
    else:
        hvi.run(hvi.no_timeout)

    for module_name, channels in data_receive.items():
        for ch in channels:
            print(f"receive data from {module_name} channel {ch}")
            inst =  module_dict[module_name].instrument
            data_receive[module_name][ch] = inst.DAQreadArray(int(ch[-1]), timeout, reshapeMode = "nAvg")
    hvi.stop()

    return data_receive