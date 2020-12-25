import warnings
import numpy as np
from collections import OrderedDict
import keysightSD1
import keysight_hvi as kthvi


class ApplicationConfig:
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


def open_modules():
    """
    Opens and creates all the necessary instrument objects.
    Returns a dictionary of module objects whose keys are the HVI engine names.
    Please check SD1 3.x User Manual for options to open the instrument objects
    """
    config = ApplicationConfig()
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
            instrObj = keysightSD1.SD_AIN()
        else:
            instrObj = keysightSD1.SD_AOU()
            instrObj.waveformFlush()
        instObjOptions = descriptor.options + \
            ',simulate=true' if config.hardwareSimulated else descriptor.options
        id = instrObj.openWithOptions(
            descriptor.modelNumber, descriptor.chassisNumber, descriptor.slotNumber, instObjOptions)
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

        module_dict[descriptor.engineName] = config.Module(
            descriptor.engineName, instrObj, hviSupport)
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
                                 kthvi.TriggerResourceId.PXI_TRIGGER2]


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


def define_instruction_compile_hvi(module_dict: dict, Q, pulse_general_dict: dict):
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
    del module_dict_temp['D2']
    sys_def = kthvi.SystemDefinition("systemInfo")
    define_hvi_resources(sys_def, module_dict_temp)
    sequencer = kthvi.Sequencer('seqName', sys_def)

    timeMax = 0
    for seqOrder in range(Q.maxIndexNum):  # the maxIndexNum should be the len(xdata)
        # first is 30 ns, then is relaxing time  ## Notice the delay should at least 30 ns
        delay = 30 if seqOrder == 0 else int(pulse_general_dict['relaxingTime'] * 1e3) - timeMax
        syncBlock = sequencer.sync_sequence.add_sync_multi_sequence_block(f"syncBlock{seqOrder}", delay)
        timeMax = 0
        for module in module_dict_temp.keys():
            chanNum = module.chanNum
            seq = syncBlock.sequences[module]

            time_sort = {}  # First clean up the time slot for different instructions
            for chan in range(1, chanNum + 1):
                try:
                    pulseInEachSeq = getattr(getattr(Q, module), f'chan{chan}')[str(seqOrder)]
                    for singlePulse in pulseInEachSeq:
                        if int(singlePulse[1]) not in time_sort.keys():
                            time_sort[int(singlePulse[1])] = []
                        if 'pulse' in singlePulse[0]:
                            time_sort[int(singlePulse[1])] += [config.awgTriggerActionName + str(chan)]
                        elif 'trigger.dig' in singlePulse[0]:
                            time_sort[int(singlePulse[1])] += [config.digTriggerActionName + str(chan)]
                        elif 'trigger.fpga' in singlePulse[0]:
                            time_sort[int(singlePulse[1])] += [config.fpgaTriggerActionName + singlePulse[0][-1]]
                except KeyError:
                    pass
            time_sort_order = OrderedDict(sorted(time_sort.items()))
            time_ = 0
            for timeIndex, actionList in time_sort_order.items():
                time_ = int(timeIndex) - time_
                aList = [seq.engine.actions[a_] for a_ in actionList]
                instru = seq.add_instruction(f"block{seqOrder}time{timeIndex}", int(time_), seq.instruction_set.action_execute.id)
                instru.set_parameter(seq.instruction_set.action_execute.action, aList)
                time_ = timeIndex
                timeMax = np.max([timeMax, timeIndex])

    hvi = sequencer.compile()
    print("HVI Compiled")
    print("This HVI application needs to reserve {} PXI trigger resources to execute".format(len(hvi.compile_status.sync_resources)))
    hvi.load_to_hw()
    print("HVI Loaded to HW")
    return hvi
