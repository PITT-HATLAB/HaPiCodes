from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py

import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
sys.path.append('C:\HatCode\PXIe')
import keysightSD1
import warnings
import keysight_hvi as kthvi

class ApplicationConfig:
    " Defines module descriptors, configuration options and names of HVI engines, actions, triggers"
    def __init__(self):
        #Configuration options
        self.hardware_simulated = False 
        # Define options to open the instruments. Complete option list can be found in the SD1 user guide
        self.options = 'channelNumbering=keysight'
        # Define names of HVI engines, actions, triggers
#         self.engine_name = "AwgEngine"
        self.awg_trigger_action_name = "AwgTrigger"
        self.dig_trigger_action_name = "DigTrigger"
        self.fp_trigger_name = "FpTrigger"
        self.boards = [['M3202A', 1, 2, self.options, "A1"],
                       ['M3202A', 1, 3, self.options, "A2"], 
                       ['M3202A', 1, 4, self.options, "A3"],
                       ['M3102A', 1, 5, self.options, "D1"],
#                        ['M3102A', 1, 6, self.options, "D2"],
                       ['M3201A', 1, 7, self.options, "M1"],
                       ['M3201A', 1, 8, self.options, "M2"],
                       ['M3201A', 1, 9, self.options, "M3"]]

    class ModuleDescriptor:
        "Descriptor for module objects"
        def __init__(self, model_number, chassis_number, slot_number, options, engine_name):
            self.model_number = model_number
            self.chassis_number = chassis_number
            self.slot_number = slot_number
            self.options = options
            self.engine_name = engine_name

    class M9031Descriptor:
        "Describes the interconnection between each pair of M9031A modules"
        def __init__(self, first_M9031_chassis_number, first_M9031_slot_number, second_M9031_chassis_number, second_M9031_slot_number):
            self.chassis_1 = first_M9031_chassis_number
            self.slot_1 = first_M9031_slot_number
            self.chassis_2 = second_M9031_chassis_number
            self.slot_2 = second_M9031_slot_number

    class Module:
        "Class defining a modular instrument object and its properties"
        def __init__(self, instrument_object, num_channels):
            self.instrument = instrument_object
            self.num_channels = num_channels


def open_modules():
    """ 
    Opens and creates all the necessary instrument objects.
    Returns a dictionary of module objects whose keys are the HVI engine names.
    Please check SD1 3.x User Manual for options to open the instrument objects
    """
    config = ApplicationConfig()
    # Initialize output variables
    num_modules = 0
    module_dict = {} # dictionary of modules

    # Append all module from config file into one list.
    AllModulesList = []
    for board in config.boards:
        AllModulesList.append(config.ModuleDescriptor(*board))

    # Open SD1 instrument objects
    for descriptor in AllModulesList:
        if descriptor.model_number == "M3102A":
            instr_obj = keysightSD1.SD_AIN()
        else:
            instr_obj = keysightSD1.SD_AOU()
        instr_obj_options = descriptor.options + ',simulate=true' if config.hardware_simulated else descriptor.options
        id = instr_obj.openWithOptions(descriptor.model_number, descriptor.chassis_number, descriptor.slot_number, instr_obj_options)
        if id < 0:
            raise Exception("Error opening instrument in chassis: {}, slot: {}! Error code: {} - {}. Exiting...".format(descriptor.chassis_number, descriptor.slot_number, id, keysightSD1.SD_Error.getErrorMessage(id)))
        nCh = instr_obj.getOptions("channels")
        if nCh == "CH2":
            num_channels = 2
        elif nCh == "CH4":
            num_channels = 4
        else:
            raise Exception("PXI module in chassis {}, slot {} returned number of channels = {} which is incorrect. Exiting... ".format(instr_obj.getChassis(), instr_obj.getSlot(), nCh))
        module_dict[descriptor.engine_name] = config.Module(instr_obj, num_channels)
        num_modules += 1
    
    return module_dict

def define_hvi_resources(sys_def, module_dict, chassis_list, M9031_descriptors, pxi_sync_trigger_resources):
    """
    Configures all the necessary resources for the HVI application to execute: HW platform, engines, actions, triggers, etc. 
    """
    # Define HW platform: chassis, interconnections, PXI trigger resources, synchronization, HVI clocks
    define_hw_platform(sys_def, chassis_list, M9031_descriptors, pxi_sync_trigger_resources)

    # Define all the HVI engines to be included in the HVI
    define_hvi_engines(sys_def, module_dict)
    
    # Define list of actions to be executed
    define_hvi_actions(sys_def, module_dict)
    
    # Defines the trigger resources
    define_hvi_triggers(sys_def, module_dict)

def define_hw_platform(sys_def, chassis_list, M9031_descriptors, pxi_sync_trigger_resources):
    """
     Define HW platform: chassis, interconnections, PXI trigger resources, synchronization, HVI clocks 
    """
    config = ApplicationConfig()

    # Add chassis resources
    for chassis_number in chassis_list:
        if config.hardware_simulated:
            sys_def.chassis.add_with_options(chassis_number, 'Simulate=True,DriverSetup=model=M9018B,NoDriver=True')
        else:
            sys_def.chassis.add(chassis_number)

    # Add M9031 modules for multi-chassis setups
    if M9031_descriptors:
        interconnects = sys_def.interconnects
        for descriptor in M9031_descriptors:
            interconnects.add_M9031_modules(descriptor.chassis_1, descriptor.slot_1, descriptor.chassis_2, descriptor.slot_2)

    # Assign the defined PXI trigger resources
    sys_def.sync_resources = pxi_sync_trigger_resources
    
    # Assign clock frequencies that are outside the set of the clock frequencies of each HVI engine
    # Use the code line below if you want the application to be in sync with the 10 MHz clock
    sys_def.non_hvi_core_clocks = [10e6]


def define_hvi_engines(sys_def, module_dict):
    """
    Define all the HVI engines to be included in the HVI
    """
    # For each instrument to be used in the HVI application add its HVI Engine to the HVI Engine Collection 
    for engine_name in module_dict.keys():
        sys_def.engines.add(module_dict[engine_name].instrument.hvi.engines.main_engine, engine_name)


def define_hvi_actions(sys_def, module_dict):
    """ Defines AWG trigger actions for each module, to be executed by the "action execute" instruction in the HVI sequence
    Create a list of AWG trigger actions for each AWG module. The list depends on the number of channels """

    # Previously defined resource names
    config = ApplicationConfig()

    # For each AWG, define the list of HVI Actions to be executed and add such list to its own HVI Action Collection
    for engine_name in module_dict.keys():
        for ch_index in range(1, module_dict[engine_name].num_channels + 1):
            # Actions need to be added to the engine's action list so that they can be executed
            if module_dict[engine_name].instrument.getProductName() == "M3102A":
                action_name = config.dig_trigger_action_name + str(ch_index) # arbitrary user-defined name
                instrument_action = "daq{}_trigger".format(ch_index) # name decided by instrument API
            else:
                action_name = config.awg_trigger_action_name + str(ch_index) # arbitrary user-defined name
                instrument_action = "awg{}_trigger".format(ch_index) # name decided by instrument API
            action_id = getattr(module_dict[engine_name].instrument.hvi.actions, instrument_action)
            sys_def.engines[engine_name].actions.add(action_id, action_name)
        for i in range(8):
            action_name = f"FPGATrigger{i}"
            instrument_action = f"fpga_user_{i}"
            action_id = getattr(module_dict[engine_name].instrument.hvi.actions, instrument_action)
            sys_def.engines[engine_name].actions.add(action_id, action_name)

def define_hvi_triggers(sys_def, module_dict):
    " Defines and configure the FP trigger output of each AWG "
    # Previously defined resources
    config = ApplicationConfig()
    
    # Add to the HVI Trigger Collection of each HVI Engine the FP Trigger object of that same instrument
    for engine_name in module_dict.keys():
        fp_trigger_id = module_dict[engine_name].instrument.hvi.triggers.front_panel_1
        fp_trigger = sys_def.engines[engine_name].triggers.add(fp_trigger_id, config.fp_trigger_name)
        # Configure FP trigger in each hvi.engines[index]
        fp_trigger.config.direction = kthvi.Direction.OUTPUT
        fp_trigger.config.polarity = kthvi.Polarity.ACTIVE_HIGH
        fp_trigger.config.sync_mode = kthvi.SyncMode.IMMEDIATE
        fp_trigger.config.hw_routing_delay = 0
        fp_trigger.config.trigger_mode = kthvi.TriggerMode.LEVEL 
        #NOTE: FP trigger pulse length is defined by the HVI Statements that control FP Trigger ON/OFF

def configureAWG(AWGModule, numChan=4):
    """ 
    awg_queue_waveform(awg_module, num_channels) queues a Gaussian waveform to all the num_channels channels of
    the AWG object awg_module that is passed to the function
    """  
    # AWG settings for all channels
    syncMode = keysightSD1.SD_SyncModes.SYNC_NONE # (SYNC_NONE / SYNC_CLK10) OR (0 / 1)
    queueMode = keysightSD1.SD_QueueMode.ONE_SHOT # (ONE_SHOT / CYCLIC)
    startDelay = 0
    prescaler = 0
    nCycles = 0
    amplitude = 1
    # Trigger settings
    triggerMode = keysightSD1.SD_TriggerModes.SWHVITRIG_CYCLE # (AUTOTRIG / SWHVITRIG / SWHVITRIG_CYCLE / EXTTRIG / EXTTRIG_CYCLE)
    # Load waveform to AWG memory
    AWGModule.waveformFlush() #memory flush
    
    for i in range(1, numChan+1):
        AWGModule.AWGqueueConfig(i + 1, 1)
        AWGModule.channelWaveShape(i + 1, keysightSD1.SD_Waveshapes.AOU_AWG)
        AWGModule.channelAmplitude(i + 1, 1)
        AWGModule.channelPhaseReset(i + 1)
    return

def uploadAWG(AWGModule, wfDictUpload, queUpload):
    for k,v in wfDictUpload.items(): # v[0]: index, v[1]: numpy.array
        tWave = keysightSD1.SD_Wave()
        tWave.newFromArrayDouble(0, v[1])
        AWGModule.waveformLoad(tWave, v[0], 0)
    return

def uploadWaveform(wfDict: dict):
    """Transfer from wfDict to upload dictioanry.
    
    Args:
        wfDict (dict): Waveform dictionary. User upload for each sequence including all waveforms.
    
    Returns:
        wfDictUpload (dict): Dictionary including index and waveform format uploading to AWG.
    """
    index = 0
    wfDictUpload = {}
    for k,v in wfDict.items():
        wfDictUpload[k] = [index, v]
        index += 1
    return wfDictUpload

module_dict = open_modules() # Initialize all modules
waveformDict = {}
testWave = signal.gaussian(100, std=51) * 0.1

for i in range(5):
    waveformDict[f'box{i}'] = signal.gaussian(84, std=21) * 0.5 * (i+1)
wfDictUpload = uploadWaveform(waveformDict)
configureAWG(module_dict['A1'].instrument)
uploadAWG(module_dict['A1'].instrument, wfDictUpload, 0)

awg_module = module_dict['A1'].instrument
for i in range(1, 5):
    # AWG queue flush 
    awg_module.AWGstop(i)
    awg_module.AWGflush(i)
    # Set AWG mode
    awg_module.channelWaveShape(i, keysightSD1.SD_Waveshapes.AOU_AWG)
    awg_module.channelAmplitude(i, 1)
    # AWG configuration
    awg_module.AWGqueueConfig(i, keysightSD1.SD_QueueMode.ONE_SHOT)
    awg_module.AWGqueueSyncMode(i, keysightSD1.SD_SyncModes.SYNC_NONE)
    # Queue waveform to channel nAWG
    awg_module.AWGqueueWaveform(i, 0, 1, 0, 1, 0)
    awg_module.AWGqueueWaveform(i, 1, 0, 30, 1, 0)

awg_module.AWGstartMultiple(0b1111) # AWG starts and wait for trigger


pxi_sync_trigger_resources = [
    kthvi.TriggerResourceId.PXI_TRIGGER0, 
    kthvi.TriggerResourceId.PXI_TRIGGER1, 
    kthvi.TriggerResourceId.PXI_TRIGGER2,
    kthvi.TriggerResourceId.PXI_TRIGGER3]
chassis_list = [1]
M9031_descriptors = []
sys_def = kthvi.SystemDefinition("systemTest") 
# Define your system, HW platform, add HVI engines
define_hvi_resources(sys_def, module_dict, chassis_list, M9031_descriptors, pxi_sync_trigger_resources)
sequencer = kthvi.Sequencer("MySequencer", sys_def)
sync_block = sequencer.sync_sequence.add_sync_multi_sequence_block("TriggerAWGs", 30)
seqA1 = sync_block.sequences[0]
seqD1 = sync_block.sequences[3]
# action_listA1 = seqA1.engine.actions
action_listA1 = [seqA1.engine.actions['AwgTrigger1'], seqA1.engine.actions['AwgTrigger4']]

instructionA1 = seqA1.add_instruction("AWG trigger", 0, seqA1.instruction_set.action_execute.id)
instructionA1.set_parameter(seqA1.instruction_set.action_execute.action, action_listA1)


action_listA1FPGA = [seqA1.engine.actions["FPGATrigger7"]]
instructionA1FPGA = seqA1.add_instruction("AWG FPGA trigger", 500, seqA1.instruction_set.action_execute.id)
instructionA1FPGA.set_parameter(seqA1.instruction_set.action_execute.action, action_listA1FPGA)


action_listD1 = [seqD1.engine.actions[f'DigTrigger{i}'] for i in range(1, 5)]
instructionD1 = seqD1.add_instruction("DAQ trigger", 0, seqD1.instruction_set.action_execute.id)
instructionD1.set_parameter(seqD1.instruction_set.action_execute.action, action_listD1)

action_listD1_FPGA = [seqD1.engine.actions['FPGATrigger7']]
instructionD1_2 = seqD1.add_instruction("FPGA Trigger7", 100, seqD1.instruction_set.action_execute.id)
instructionD1_2.set_parameter(seqD1.instruction_set.action_execute.action, action_listD1_FPGA)

action_listD1_FPGA2 = [seqD1.engine.actions['FPGATrigger7']]
instructionD1_3 = seqD1.add_instruction("FPGA Trigger7_2", 100, seqD1.instruction_set.action_execute.id)
instructionD1_3.set_parameter(seqD1.instruction_set.action_execute.action, action_listD1_FPGA2)


for i in range(1, 5):
    module_dict['D1'].instrument.channelInputConfig(i, 2, 1, 1)
    module_dict['D1'].instrument.channelPrescalerConfig(i, 0)
    module_dict['D1'].instrument.DAQconfig(i, 1000, 1, 0, keysightSD1.SD_TriggerModes.SWHVITRIG)

hvi = sequencer.compile()
print("HVI Compiled")
print("This HVI application needs to reserve {} PXI trigger resources to execute".format(len(hvi.compile_status.sync_resources)))
hvi.load_to_hw()    
print("HVI Loaded to HW")
module_dict['D1'].instrument.DAQstartMultiple(0b1111)
hvi.run(hvi.no_timeout)
test = module_dict['D1'].instrument.DAQread(4, 1000 * 1, 1000)
print(test)
d40 = test[::5]
d41 = test[1::5]+1
d42 = test[2::5]+2
d43 = test[3::5]+3
d44 = test[4::5]+4

plt.plot(np.arange(200)*5, d40)
plt.plot(np.arange(200)*5, d41)
plt.plot(np.arange(200)*5, d42)
plt.plot(np.arange(200)*5, d43)
plt.plot(np.arange(200)*5, d44)

# Release HW resources once HVI execution is completed
hvi.release_hw()
print("Releasing HW...")

for engine_name in module_dict:
    module_dict[engine_name].instrument.close()
print("Modules closed\n")

plt.show()