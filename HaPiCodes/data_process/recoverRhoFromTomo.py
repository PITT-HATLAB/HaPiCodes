# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:42:14 2021

@author: Chao
"""
from typing import Dict, List

import qutip as qt
import numpy as np
from itertools import product, combinations 



oDict = {"I": qt.identity(2),
         "Z": qt.sigmaz(),
         "X": qt.sigmax(),
         "Y": qt.sigmay()}


def orderedTomoSeries(nbits):
    allComb = list(product(list(oDict.keys()),repeat=nbits))[1:]
    tomoNames = []
    # sort all the combinations in the order of number of "I"s
    for nI in range(nbits-1, -1, -1):
        Ipositions = combinations(range(nbits), nI)
        xyz_combs = list(product(list(oDict.keys())[1:], repeat=nbits-nI))
        for Iposi in Ipositions:
            for comb_ in xyz_combs:
                i_comb = iter(comb_)
                tomo_ = ["I" ]*nbits
                for i in range(nbits):
                    if i not in Iposi:
                        tomo_[i] = next(i_comb)

                tomoNames.append("".join(tomo_))
    return tomoNames

def tomoOperatorDict(tomo_series: List[str]):
    tomos = {}
    for t in tomo_series:
        ops = [oDict[o] for o in t]
        tomos[t] = qt.tensor(*ops)
    return tomos

def recoverRhoDicts(nbits:int, tomo_dict:Dict[str, qt.Qobj]):
    tomo_series = list(tomo_dict.keys())
    tomo_series.append("III")
    dim = 2 ** nbits
    qdims = [[2] * nbits, [2] * nbits]
    rho0 = np.zeros((dim, dim), dtype=np.complex)
    
    coeff_matrix_diag = np.zeros((dim ** 2, dim))
    coeff_matrix_upR = np.zeros((dim ** 2, dim*(dim-1)//2))
    coeff_matrix_upI = np.zeros((dim ** 2, dim*(dim-1)//2))
    
    # find coefficients for the diagnal elemets    
    for i in range(dim):
        rho_ = rho0.copy()
        rho_[i,i] = 1
        dm_ = qt.Qobj(rho_, qdims)
        for j, operator in enumerate(tomo_dict.values()):
            coeff_matrix_diag[j, i] = (dm_ * operator).tr()
    coeff_matrix_diag[-1] = np.array([1]*dim)
            
    # find coefficients for the off-diagnal elemets
    l=0         
    for i in range(dim):
        for j in range(i+1, dim):
            rho_R = rho0.copy()
            rho_I = rho0.copy()
            rho_R[i,j] = 1
            rho_R[j,i] = 1
            rho_I[i,j] = 1j
            rho_I[j,i] = -1j            
            dm_R = qt.Qobj(rho_R, qdims)
            dm_I = qt.Qobj(rho_I, qdims)
            
            for k, operator in enumerate(tomo_dict.values()):
                coeff_matrix_upR[k, l] = (dm_R * operator).tr()
                coeff_matrix_upI[k, l] = (dm_I * operator).tr()
            l += 1
    
    coeff_matrix = np.concatenate((coeff_matrix_diag, coeff_matrix_upR,
                                   coeff_matrix_upI), axis=1)
    
    inv_cm = np.linalg.inv(coeff_matrix)
    inv_cm_diag = inv_cm[:dim]
    inv_cm_upR = inv_cm[dim: dim*(dim+1)//2]
    inv_cm_upI = inv_cm[dim*(dim+1)//2:]
    
    rho = [[None for i in range(dim)] for j in range(dim)]
    # diagnal elements
    for i in range(dim):
        rho[i][i] = dict(zip(tomo_series, inv_cm_diag[i]))
    
    # off-diagnal elements
    l = 0
    for i in range(dim):
        for j in range(i+1, dim):
            rho[i][j] = dict(zip(tomo_series, inv_cm_upR[l] + 1j * inv_cm_upI[l]))
            rho[j][i] = dict(zip(tomo_series, inv_cm_upR[l] - 1j * inv_cm_upI[l]))
            l += 1
    return rho

def recoverRhoFromTomo(tomoResults:Dict[str, float], recoverRhoDicts:List[List[Dict]]):
    tomoResults["III"] = 1
    dim = (np.array(recoverRhoDicts).shape)[0]
    nbits = int(np.log2(dim))
    qdims = [[2] * nbits, [2] * nbits]
    rho = np.zeros((dim, dim), dtype=np.complex)
    for i in range(dim):
        for j in range(dim):
            coeff_dict = recoverRhoDicts[i][j]
            rho[i, j] = np.sum([coeff_dict[t] * tomoResults[t] for t in coeff_dict])
    rho = qt.Qobj(rho, qdims)
    return rho
    

def emulateTomoMSMT(rho, tomoDict):
    results = {}
    for tomo, op in tomoDict.items():
        results[tomo] = (rho * op).tr()
    return results        

if __name__ == '__main__':
    nBits = 3
    tomoSeries = orderedTomoSeries(nBits)   
    tomoDict = tomoOperatorDict(tomoSeries)
    rhoDicts_recover = recoverRhoDicts(nBits, tomoDict)  
    # --------------tests-------------------------------
    rho_test = qt.ket2dm(qt.ghz_state(3))
    # rho_test = qt.ket2dm(qt.w_state(3))
    # rho_test = qt.ket2dm(qt.bell_state("11"))
    tomoResultDict = emulateTomoMSMT(rho_test, tomoDict)
    rho = recoverRhoFromTomo(tomoResultDict, rhoDicts_recover)
    print(tomoSeries)


