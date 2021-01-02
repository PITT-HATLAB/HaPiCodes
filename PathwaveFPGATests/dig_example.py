import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1

# MODULE CONSTANTS

PRODUCT = "M3102A"
CHASSIS = 1
SLOT = 5
CHANNEL = 1

# CREATE AND OPEN MODULE
module = keysightSD1.SD_AIN()
moduleID = module.openWithSlot(PRODUCT, CHASSIS, SLOT)
if moduleID < 0:
    print("Module open error:", moduleID)
else:
    print("Module opened:", moduleID)
    print("Module name:", module.getProductName())
    print("slot:", module.getSlot())
    print("Chassis:", module.getChassis())
    print()

# CONFIGURE AND START DAQ
POINTS_PER_CYCLE = 350
CYCLES = 2
TRIGGER_DELAY = 0
module.DAQconfig(CHANNEL, POINTS_PER_CYCLE, CYCLES, TRIGGER_DELAY, keysightSD1.SD_TriggerModes.SWHVITRIG)
module.DAQstart(CHANNEL)

input("Press any key to provide trigger")
module.DAQtrigger(CHANNEL)

# READ DATA
TIMEOUT = 1
dataRead = module.DAQread(CHANNEL, POINTS_PER_CYCLE * CYCLES, TIMEOUT)
print(dataRead)

# exiting...
module.close()
print()
print("AIN closed")




# © Keysight Technologies, 2020

# All rights reserved.

# You have a royalty-free right to use, modify, reproduce and distribute this Sample Application (and/or any modified # version) in any way you find useful,

# provided that you agree that Keysight Technologies has no warranty, obligations or liability for any Sample Application Files.

#

# Keysight Technologies provides programming examples for illustration only.

# This sample program assumes that you are familiar with the programming language being demonstrated and the tools used to create and debug procedures.

# Keysight Technologies support engineers can help explain the functionality of Keysight Technologies software components and associated commands,

# but they will not modify these samples to provide added functionality or construct procedures to meet your specific needs.

# ----------
