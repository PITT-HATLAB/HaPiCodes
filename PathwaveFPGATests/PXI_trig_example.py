import time
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1

# CONFIGURATION CONSTANTS
FULL_SCALE = 2  # half peak to peak voltage
DELAY_IN = 0  # 250
READ_TIMEOUT = 100  # 0 means infinite timeout
DAQ_CH = 1  # DAQ channel
WAITER_TIMEOUT_SECONDS = 0.1


def waitUntilPointsRead(module, DAQchannel, totalPoints, timeOut):
    t0 = time.time()
    timeElapsed = 0
    totalPointsRead = 0
    while (totalPointsRead < totalPoints) and (timeElapsed < timeOut):
        totalPointsRead = module.DAQcounterRead(DAQchannel)
        if (totalPointsRead < totalPoints):
            time.sleep(.1)  # input argument of time.sleep is in seconds
        timeElapsed = time.time() - t0


# MODULE CONSTANTS
PRODUCT = "M3102A"
CHASSIS = 1
SLOT_IN = 5  # digitizer slot in chassis

# CREATE AND OPEN MODULE IN
digitizer = keysightSD1.SD_AIN()
digitizerID = digitizer.openWithSlot(PRODUCT, CHASSIS, SLOT_IN)
if digitizerID < 0:
    print("Module open error:", digitizerID)
else:
    print("===== Digitizer =====")
    print("ID:\t\t", digitizerID)
    print("Product name:\t", digitizer.getProductName())
    print("Serial number:\t", digitizer.getSerialNumber())
    print("Chassis:\t", digitizer.getChassis())
    print("Slot:\t\t", digitizer.getSlot())
    print()




plt.ion()  # interactive mode
plt.show(block=False)
print("Test: PXI external trigger, 1000 total points, 1 cycle...")
NUM_CYCLES = 1
TOTAL_POINTS = 1000

# DAQ CONFIGURATION
digitizer.channelInputConfig(DAQ_CH, FULL_SCALE, keysightSD1.AIN_Impedance.AIN_IMPEDANCE_50,
                             keysightSD1.AIN_Coupling.AIN_COUPLING_DC)
digitizer.DAQconfig(DAQ_CH, TOTAL_POINTS, NUM_CYCLES, DELAY_IN, keysightSD1.SD_TriggerModes.EXTTRIG)
digitizer.DAQdigitalTriggerConfig(DAQ_CH, keysightSD1.SD_TriggerExternalSources.TRIGGER_PXI2,
                                  keysightSD1.SD_TriggerBehaviors.TRIGGER_RISE)

# DAQ ACQUISITION
digitizer.DAQflush(DAQ_CH)
digitizer.DAQstart(DAQ_CH)
waitUntilPointsRead(digitizer, DAQ_CH, TOTAL_POINTS, WAITER_TIMEOUT_SECONDS)

print("total points read before trigger:", digitizer.DAQcounterRead(DAQ_CH))

# PXI2 Trigger
PXI2 = 2
digitizer.PXItriggerWrite(PXI2, 1);
digitizer.PXItriggerWrite(PXI2, 0);
digitizer.PXItriggerWrite(PXI2, 1);
waitUntilPointsRead(digitizer, DAQ_CH, TOTAL_POINTS, WAITER_TIMEOUT_SECONDS)
print("total points read after trigger:", digitizer.DAQcounterRead(DAQ_CH))

# read points

readPoints = digitizer.DAQread(DAQ_CH, TOTAL_POINTS, READ_TIMEOUT)
print("total points read: {}".format(readPoints.size))

# STOP DAQ
digitizer.DAQstop(DAQ_CH)

# PLOT
print("Plotting test...")
plt.clf()
plt.plot(readPoints, 'r-')
plt.show()
plt.pause(2)

# exiting...
digitizer.close()

print()

print("AIN closed")

# ----------------------------------

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
