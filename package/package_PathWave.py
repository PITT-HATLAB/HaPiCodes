import keysight_hvi as kthvi
import keysightSD1
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
import warnings
import json

import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
sys.path.append('C:\HatCode\PXIe')

'''
Convention:
variable, list, class: camelName
dict, function: underscore_dict
'''


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
        instObjOptions = descriptor.options + \
            ',simulate=true' if config.hardwareSimulated else descriptor.options
        id = instrObj.openWithOptions(
            descriptor.modelNumber, descriptor.chassisNumber, descriptor.slotNumber, instObjOptions)
        if id < 0:
            raise Exception("Error opening instrument in chassis: {}, slot: {}! Error code: {} - {}. Exiting...".format(
                descriptor.chassisNumber, descriptor.slotNumber, id, keysightSD1.SD_Error.getErrorMessage(id)))
        nCh = instrObj.getOptions("hvi")
        if nCh == "HVI":
            hviSupport = 1
        elif nCh == "none":
            hviSupport = 0
        else:
            raise Exception("PXI module in chassis {}, slot {} returned number of channels = {} which is incorrect. Exiting... ".format(
                instrObj.getChassis(), instrObj.getSlot(), nCh))
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


def configAWG(awgModule, numChan=4, amplitude=[1.5, 1.5, 1.5, 1.5], offset=[0, 0, 0, 0]):
    """
    awg_queue_waveform(awg_module, num_channels) queues a Gaussian waveform to all the num_channels channels of
    the AWG object awg_module that is passed to the function
    """
    # AWG settings for all channels
    # (SYNC_NONE / SYNC_CLK10) OR (0 / 1)
    syncMode = keysightSD1.SD_SyncModes.SYNC_CLK10
    queueMode = keysightSD1.SD_QueueMode.CYCLIC  # (ONE_SHOT / CYCLIC)
    # Load waveform to AWG memory
    awgModule.instrument.waveformFlush()  # memory flush
    awgModule.instrument.channelPhaseResetMultiple(0b1111)
    for i in range(1, numChan + 1):
        awgModule.instrument.AWGflush(i)
        awgModule.instrument.AWGqueueSyncMode(i, syncMode)
        awgModule.instrument.AWGqueueConfig(i, queueMode)
        awgModule.instrument.channelWaveShape(
            i, keysightSD1.SD_Waveshapes.AOU_AWG)
        awgModule.instrument.channelAmplitude(i, amplitude[i - 1])
        awgModule.instrument.channelOffset(i, offset[i - 1])
    return


def moduleFpgaLoad(module, fpgaName):
    loadId = module.instrument.FPGAload(fpgaName)
    if loadId < 0:
        raise NameError('Need to complete the code here:' + str(loadId))
    with open(r"sysInfo.json") as file_:
        info = json.load(file_)
        fpgaOld = info['FPGA'][module.moduleName]
    info['FPGA'][module.moduleName] = fpgaName
    with open(r"sysInfo.json", 'w') as file_:
        json.dump(info, file_)


def configDig(digModule, fpgaName, manualReloadFPGA=0):
    with open(r"sysInfo.json") as file_:
        info = json.load(file_)
        fpgaOld = info['FPGA'][digModule.moduleName]

    if not manualReloadFPGA:
        if fpgaName == fpgaOld:
            print(digModule.moduleName + ' has same FPGA, no need to reload')
        else:
            moduleFpgaLoad(digModule, fpgaName)
    else:
        moduleFpgaLoad(digModule, fpgaName)

    for i in range(1, 5):
        # channel, fullscale, impedance, coupling
        digModule.instrument.channelInputConfig(i, 2, 1, 1)
        digModule.instrument.channelPrescalerConfig(i, 0)


# chan order: ch4, ch3, ch2, ch1
def digAcceptData(digModule, hvi, pointPerCycle, cycles, triggerDelay=0, chan="1111", timeout=1000):
    data_receive = {}
    triggerMode = keysightSD1.SD_TriggerModes.SWHVITRIG
    if type(triggerDelay) == int:
        triggerDelay = [triggerDelay] * 4
    if len(triggerDelay) == 1:
        triggerDelay = triggerDelay * 4
    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            data_receive[str(chanNum)] = np.zeros(pointPerCycle * cycles)
            digModule.instrument.DAQconfig(
                chan, pointPerCycle, cycles, triggerDelay[chanNum - 1], triggerMode)
        else:
            data_receive[str(chanNum)] = []
        chanNum -= 1
    digModule.instrument.DAQstartMultiple(int(chan, 2))
    hvi.run(hvi.no_timeout)
    chanNum = 4
    for chan_ in chan:
        if int(chan_):
            print('receive data from channel' + str(chanNum))
            data_receive[str(chanNum)] = digModule.instrument.DAQread(
                chanNum, pointPerCycle * cycles, timeout)
        chanNum -= 1
    return data_receive


def uploadAWG(AWGModule, wfDictUpload, queUpload):
    for k, v in wfDictUpload.items():  # v[0]: index, v[1]: numpy.array
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
    for k, v in wfDict.items():
        wfDictUpload[k] = [index, v]
        index += 1
    return wfDictUpload


if __name__ == '__main__':
    with open(r"sysinfo.json") as file_:
        info = json.load(file_)
    print(info)
