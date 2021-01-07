from typing import List, Callable, Union, Tuple, Dict
from dataclasses import dataclass
import warnings
import pathlib

import numpy as np
from nptyping import NDArray
import yaml

import keysightSD1 as SD1
from  sd1_api.SD1AddOns import write_FPGA_memory, write_FPGA_register
import FPGA



fpga_path = pathlib.Path(FPGA.__path__[0])
CONFIG_DICT = yaml.safe_load(open(str(fpga_path)+r"\Demodulate_showWeight\Demodulate_showWeight_Markup.yaml", 'r'))


@dataclass
class FPGARegisters:
    wf_trig_delay: int
    wf_length: int
    demod_trunc: int = 18

def check_register_constraint(reg_name: str, reg_val: int, constraint_dict: Dict, warning_message: str = None):
    lower_bound = constraint_dict[reg_name]["min"]
    upper_bound = constraint_dict[reg_name]["max"]
    min_eff = constraint_dict[reg_name].get("min_eff")  # minimum value to make the corresponding block functional
    if (reg_val < lower_bound) or (reg_val > upper_bound):
        raise ValueError(f"register {reg_name} does not satisfy the constraint [{lower_bound}, {upper_bound}]")
    if (min_eff is not None) and (reg_val < min_eff):
        warnings.warn(warning_message)

def config_FPGA_registers(module_: SD1.SD_Module, channel:int, registers: FPGARegisters, module_name: str = None):
    cstr_dict = CONFIG_DICT["register_constraints"]
    if module_name is None:
        module_name = f"at chassis{module_.getChassis()}_slot{module_.getSlot()}"
    for k, v in registers.__dict__.items():
        if k == "wf_trig_delay":
            min_eff_ = cstr_dict[k]["min_eff"]
            wf_idle_ = CONFIG_DICT["wf_idle_val"]
            warn_msg = f"weight function trigger delay {v} is smaller than {min_eff_}, " \
                       f"weight function will be constant {wf_idle_} for module {module_name} channel {channel}"
        else:
            warn_msg = None
        check_register_constraint(k, v, cstr_dict, warn_msg)
        write_FPGA_register(module_, f"FPGARegisters_{channel}_{k}", v)


def configFPGA(module: SD1.SD_Module, channel:int,
               demod_trunc: int,  wf_data: List[int] = None, wf_start: int = None, wf_stop:int = None,
               module_name: str = None):
    """

    :param module_: AIN module
    :param channel: channel to configure
    :param demod_trunc: lower bit offset for truncating the 35bit data after demodulation to 16 bit
    :param wf_data: weight function data, 10 ns per point
    :param wf_start: in ns, weight function starting point
    :param wf_stop: in ns, weight function stop point
    :param module_name: user defined name of module, if None, physical location of module will be sued
    :return:
    """
    if wf_data is None:
        wf_data = [32767]
        wf_start = 1 # no wait function
        wf_stop = 2
    else:
        if wf_start is None:
            wf_start = 2
        if wf_stop is None:
            wf_stop = wf_start + len(wf_data)

    wf_length = wf_stop - wf_start
    FPGA_regs = FPGARegisters(wf_start, wf_length, demod_trunc)
    config_FPGA_registers(module, channel, FPGA_regs, module_name)

    write_FPGA_memory(module, f"WeightFunc_{channel}", wf_data)


