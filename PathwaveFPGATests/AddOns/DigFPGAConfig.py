import time
from typing import List, Callable, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray
import keysightSD1 as SD1

from .SD1AddOns import write_FPGA_memory, write_FPGA_register


def config_mem_reader(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      mem_name: str = "Host_mem_1", delay_reg_name: str = "MemReaderRegisters_rd_trig_delay",
                      length_reg_name: str = "MemReaderRegisters_rd_length"):
    write_FPGA_memory(module_, mem_name, input_data)
    write_FPGA_register(module_, delay_reg_name, trig_delay)
    write_FPGA_register(module_, length_reg_name, rd_length)


def config_IQIntegrator(module_: SD1.SD_Module, integ_start: int, integ_stop: int,
                        truncate_demod: List[int] = None,
                        truncate_integ: List[int] = None):
    write_FPGA_register(module_, "IntegRegisters_integ_trig_delay", integ_start)
    write_FPGA_register(module_, "IntegRegisters_integ_length", integ_stop)

    if truncate_demod is None:
        truncate_demod = [18, 18, 18, 18]
        #TODO: This needs a validator. The possible value should be 0-18 (for current FPGA), and this value needs to be in a markup file)
    if truncate_integ is None:
        truncate_integ = [6, 7, 7, 7] #TODO: Again, validator. The possible value should be 0-15 (for current FPGA)
    for i in range(4):
        write_FPGA_register(module_, f"TruncationRegisters_truncate_demod_{i}", truncate_demod[i])
        write_FPGA_register(module_, f"TruncationRegisters_truncate_integ_{i}", truncate_integ[i])


def config_mem_reader_1(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      mem_name: str = "Host_mem_1", delay_reg_name: str = "MemReaderRegisters_1_rd_trig_delay",
                      length_reg_name: str = "MemReaderRegisters_1_rd_length"):
    write_FPGA_memory(module_, mem_name, input_data)
    write_FPGA_register(module_, delay_reg_name, trig_delay)
    write_FPGA_register(module_, length_reg_name, rd_length)

def config_IQIntegrator_1(module_: SD1.SD_Module, integ_start: int, integ_stop: int,
                        truncate_demod: List[int] = None,
                        truncate_integ: List[int] = None):
    write_FPGA_register(module_, "IntegRegisters_1_integ_trig_delay", integ_start)
    write_FPGA_register(module_, "IntegRegisters_1_integ_length", integ_stop)

    if truncate_demod is None:
        truncate_demod = [18, 18, 18, 18]
        #TODO: This needs a validator. The possible value should be 0-18 (for current FPGA), and this value needs to be in a markup file)
    if truncate_integ is None:
        truncate_integ = [6, 7, 7, 7] #TODO: Again, validator. The possible value should be 0-15 (for current FPGA)
    for i in range(4):
        write_FPGA_register(module_, f"TruncationRegisters_1_truncate_demod_{i}", truncate_demod[i])
        write_FPGA_register(module_, f"TruncationRegisters_1_truncate_integ_{i}", truncate_integ[i])


def get_recommended_truncation(data_I: NDArray[float], data_Q:NDArray[float],
                               integ_start: int, integ_stop: int, current_demod_trunc: int = 18,
                               fault_tolerance_factor:float = 1.01) -> Tuple[int, int]:
    """ get recommended truncation point for both demodulation and integration from the cavity response data trace.

    :param data_I: cavity response I data. 1 pt/10 ns. The shape should be (DAQ_cycles, points_per_cycle).
    :param data_Q: cavity response Q data. 1 pt/10 ns. The shape should be (DAQ_cycles, points_per_cycle).
    :param integ_start: integration start point, unit: ns
    :param integ_stop: integration stop point (not integrated), unit: ns
    :param current_demod_trunc: the demodulation truncation point used to get data_I and data_Q
    :param fault_tolerance_factor: a factor that will be multiplied onto the data to make sure overflow will not happen
    :return: demod_trunc_point, integ_trunc_point
    """
    data_I = data_I.astype(float)
    data_Q = data_Q.astype(float) # to avoid overflow in calculation
    max_mag = np.max(np.sqrt(data_I ** 2 + data_Q ** 2)) * fault_tolerance_factor
    bits_available = 15 - int(np.ceil(np.log2(max_mag+1)))
    demod_trunc = current_demod_trunc - bits_available
    data_I_new = data_I * 2 ** bits_available
    data_Q_new = data_Q * 2 ** bits_available

    #TODO: validate integ_start and integ_stop
    integ_I = np.sum(data_I_new[:, integ_start // 10: integ_stop // 10])
    integ_Q = np.sum(data_Q_new[:, integ_start // 10: integ_stop // 10])
    max_integ = np.max(np.sqrt(integ_I ** 2 + integ_Q ** 2)) * fault_tolerance_factor
    integ_trunc = np.clip(int(np.ceil(np.log2(max_integ+1))) - 15, 0, 15)
    return demod_trunc, integ_trunc

