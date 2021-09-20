import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import pathlib
import os


def operator_creator(dim_list_, annihilation=0):
    '''
    returns I, A, Adag, B, Bdag,...
    '''
    op_num = len(dim_list_)
    des_oplist = [qt.destroy(dim_list_[i]) for i in range(op_num)]
    iden_oplist = [qt.identity(dim_list_[i]) for i in range(op_num)]
    tensored_ops = tuple()
    id_op = iden_oplist[0]
    for i in range(op_num):
        if i == 0:
            tmp_op = des_oplist[0]
        else:
            tmp_op = iden_oplist[0]
            id_op = qt.tensor(id_op, iden_oplist[i])
        for j in range(1, op_num):
            if i == j:
                tmp_op = qt.tensor(tmp_op, des_oplist[j])
            else:
                tmp_op = qt.tensor(tmp_op, iden_oplist[j])

        if annihilation:
            tensored_ops += (tmp_op, )
        else:
            tensored_ops += (tmp_op, tmp_op.dag())

    return (id_op,) + tensored_ops


def generateDecayAndDecoherence(t1, t2, op):
    return np.sqrt(1 / t1) * op, np.sqrt(1 / t2) * op.dag() * op