import time
from typing import List, Callable, Union, Tuple, Dict
from dataclasses import dataclass
import warnings

import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray
import yaml

import keysightSD1 as SD1
from .SD1AddOns import write_FPGA_memory, write_FPGA_register

def config_demodulator(module_: SD1.SD_Module, truncate_demod: List[int] = None):
    if truncate_demod is None:
        truncate_demod = [18, 18, 18, 18]
        #TODO: This needs a validator. The possible value should be 0-18 (for current FPGA), and this value needs to be in a markup file)
    for i in range(4):
        write_FPGA_register(module_, f"TruncationRegisters_truncate_lower_bit_{i+1}", truncate_demod[i])


def config_weight_func(module_: SD1.SD_Module, channel: int, wf_data: List[int], trig_delay: int, wf_length: int = None):
    if wf_length is None:
        wf_length = len(wf_data)
    write_FPGA_memory(module_, f"WeightFunc_{channel}", wf_data)
    write_FPGA_register(module_, f"WeightFuncRegisters_rd_trig_delay_{channel}", trig_delay)
    write_FPGA_register(module_, f"WeightFuncRegisters_rd_length_{channel}", wf_length)





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



# ----------------------------For Test----------------------------------------------------------------------------------

def config_mem_reader(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      mem_name: str = "Host_mem_1", delay_reg_name: str = "DataGenRegisters_trig_delay",
                      length_reg_name: str = "DataGenRegisters_length"):
    write_FPGA_memory(module_, mem_name, input_data)
    write_FPGA_register(module_, delay_reg_name, trig_delay)
    write_FPGA_register(module_, length_reg_name, rd_length)

def config_mem_reader_1(module_: SD1.SD_Module, input_data: List[int], trig_delay: int, rd_length: int,
                      mem_name: str = "Host_mem_1", delay_reg_name: str = "DataGenRegisters_1_trig_delay",
                      length_reg_name: str = "DataGenRegisters_1_length"):
    write_FPGA_memory(module_, mem_name, input_data)
    write_FPGA_register(module_, delay_reg_name, trig_delay)
    write_FPGA_register(module_, length_reg_name, rd_length)





#------------------------- Change FPGA register names before implement these-------------------------------------------
CONFIG_DICT = yaml.safe_load(open(r"C:\Users\PXIe1\PycharmProjects\PathwaveFPGATests\AddOns\DigFPGAMarkup_Demodulate.yaml", 'r'))

def check_register_constraint(reg_name: str, reg_val: int, constraint_dict: Dict, warning_message: str = None):
    lower_bound = constraint_dict[reg_name]["min"]
    upper_bound = constraint_dict[reg_name]["max"]
    min_eff = constraint_dict[reg_name].get("min_eff")  # minimum value to make the corresponding block functional
    if (reg_val < lower_bound) or (reg_val > upper_bound):
        raise ValueError(f"register {reg_name} does not satisfy the constraint [{lower_bound}, {upper_bound}]")
    if (min_eff is not None) and (reg_val < min_eff):
        warnings.warn(warning_message)

@dataclass
class FPGARegisters:
    wf_trig_delay: int
    wf_length: int
    demod_trunc: int = 18

    def __post_init__(self):
        cstr_dict = CONFIG_DICT["register_constraints"]
        for k, v in self.__dict__.items():
            if k == "wf_trig_delay":
                min_eff_ = cstr_dict[k]["min_eff"]
                wf_idle_ = CONFIG_DICT["wf_idle_val"]
                warn_msg = f"weight function trigger delay is smaller than {min_eff_}, weight function will be constant {wf_idle_}."
            else:
                warn_msg = None
            check_register_constraint(k, v, cstr_dict, warn_msg)


def config_FPGA_registers(module_: SD1.SD_Module, channel:int, registers: FPGARegisters):
    for k, v in registers.__dict__.items():
        write_FPGA_register(module_, f"FPGARegisters_{channel}_{k}", v)

#---------------------------------------------------------------------------------------------------------------------------------