import sys
import time
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1

# Set product details
product = 'M3102A'
chassis = 1
slot = 5

# open a module
module = keysightSD1.SD_Module()
moduleID = module.openWithSlot(product, chassis, slot)

if moduleID < 0:
    print("Module open error: ", moduleID)
else:
    print("Module is: ", moduleID)
# Loading FPGA sandbox using .K7z file
error = module.FPGAload(
    r'C:\PXI_FPGA\Projects\Tests\MemReader\MemReader.data\bin\MemReader_2020-11-29T16_48_52\MemReader.k7z')
numRegisters = 3
# Get Registers list
registers = module.FPGAgetSandBoxRegisters(numRegisters)
# Print the register properties in register list
for register in registers:
    print(register.Name)
    print(register.Length)
    print(register.Address)
    print(register.AccessType)

registerName = 'Register_Bank_rd_trig_delay'
# Get Sandbox Register with name "Register_Bank_rd_trig_delay"
registerA = module.FPGAgetSandBoxRegister(registerName)
# Write data to Register_Bank_A
error = registerA.writeRegisterInt32(3)
registerNameB = 'Register_Bank_rd_length'
# Get Sandbox Register with name "Register_Bank_rd_length"
registerB = module.FPGAgetSandBoxRegister(registerNameB)
# Write data to Register_Bank_B
error = registerB.writeRegisterInt32(10)

# Read data from Register_Bank_B
regB = registerB.readRegisterInt32()


memoryMap = 'Host_mem_1'
# # Get Sandbox memoryMap with name "Host_mem_1"
memory_Map = module.FPGAgetSandBoxRegister(memoryMap)
# # Write buffer to memory map
a = list(range(1000))
memory_Map.writeRegisterBuffer(0, a, keysightSD1.SD_AddressingMode.AUTOINCREMENT, keysightSD1.SD_AccessMode.DMA)
# # Read buffer from memory map
c_value = memory_Map.readRegisterBuffer(0, 1000, keysightSD1.SD_AddressingMode.AUTOINCREMENT, keysightSD1.SD_AccessMode.NONDMA)
print(c_value)
