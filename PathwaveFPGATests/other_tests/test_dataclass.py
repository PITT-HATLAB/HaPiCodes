import time
from typing import List, Callable, Union, Tuple, Dict
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray
import yaml

import keysightSD1 as SD1

CONFIG_DICT = yaml.safe_load(open(r"C:\Users\PXIe1\PycharmProjects\PathwaveFPGATests\AddOns\DigFPGAMarkup_QubitMSMT.yaml", 'r'))


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
    integ_trig_delay: int
    integ_length: int
    wf_trig_delay: int
    wf_length: int
    demod_trunc: int = 18
    integ_trunc: int = 7

    def __post_init__(self):
        cstr_dict = CONFIG_DICT["register_constraints"]
        for k, v in self.__dict__.items():
            if k == "integ_trig_delay":
                min_eff_ = cstr_dict[k]["min_eff"]
                warn_msg = f"integration trigger delay is smaller than{min_eff_} , integration will not happen."
            elif k == "wf_trig_delay":
                min_eff_ = cstr_dict[k]["min_eff"]
                wf_idle_ = CONFIG_DICT["wf_idle_val"]
                warn_msg = f"weight function trigger delay is smaller than {min_eff_}, weight function will be constant {wf_idle_}."
            else:
                warn_msg = None
            check_register_constraint(k, v, cstr_dict, warn_msg)


if __name__ == "__main__":
    fr = FPGARegisters(1,2,1,4)