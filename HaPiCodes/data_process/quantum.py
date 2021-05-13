from typing import List, Dict
from itertools import product, combinations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt
from qutip.qip.operations import gate_expand_1toN as e2N
from qutip.qip.operations.gates import rz
from tqdm import tqdm

from HaPiCodes.data_process.sliderPlot import sliderBarPlot


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
    


def cal_CHSH(jd):
    result = -2 * jd["ZZ"] - 2 * jd["XX"]
    result /= np.sqrt(2)
    return result


def cal_density_matrix(x, y, z):
    rho = np.zeros((2, 2), dtype=complex)
    rho[0, 0] = (z + 1) / 2.0
    rho[0, 1] = (x - 1j * y) / 2.0
    rho[1, 0] = rho[0, 1].conjugate()
    rho[1, 1] = 1 - rho[0, 0]
    return rho


def cal_density_matrix_joint(jd):
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = (1 + jd['IZ'] + jd['ZI'] + jd['ZZ']) / 4.0
    rho[1, 1] = (1 - jd['IZ'] + jd['ZI'] - jd['ZZ']) / 4.0
    rho[2, 2] = (1 + jd['IZ'] - jd['ZI'] - jd['ZZ']) / 4.0
    rho[3, 3] = (1 - jd['IZ'] - jd['ZI'] + jd['ZZ']) / 4.0
    rho[0, 1] = (jd['IX'] + jd['ZX']) / 4.0 - 1j * (jd['IY'] + jd['ZY']) / 4.0
    rho[0, 2] = (jd['XI'] + jd['XZ']) / 4.0 - 1j * (jd['YI'] + jd['YZ']) / 4.0
    rho[0, 3] = (jd['XX'] - jd['YY']) / 4.0 - 1j * (jd['XY'] + jd['YX']) / 4.0
    rho[1, 2] = (jd['XX'] + jd['YY']) / 4.0 + 1j * (jd['XY'] - jd['YX']) / 4.0
    rho[1, 3] = (jd['XI'] - jd['XZ']) / 4.0 - 1j * (jd['YI'] - jd['YZ']) / 4.0
    rho[2, 3] = (jd['IX'] - jd['ZX']) / 4.0 - 1j * (jd['IY'] - jd['ZY']) / 4.0
    for i in [1, 2, 3]:
        for j in range(i):
            rho[i, j] = rho[j, i].conjugate()
    return rho


def cal_Entropy(rho):
    eigen_vals = np.linalg.eigvals(rho)
    entropy = 0
    for ev in eigen_vals:
        if ev != 0:
            entropy -= ev * np.log2(ev)
    return entropy


def cal_MI(jd):
    rho_AB = cal_density_matrix_joint(jd)
    rho_A = cal_density_matrix(jd["XI"], jd["YI"], jd["ZI"])
    rho_B = cal_density_matrix(jd["IX"], jd["IY"], jd["IZ"])

    SAB = cal_Entropy(rho_AB)
    SA = cal_Entropy(rho_A)
    SB = cal_Entropy(rho_B)
    return SAB - SA - SB


def calFidelityofBellState(rho, plot=0, type="odd"):
    rho = np.matrix(rho)
    phaseArray = np.linspace(-np.pi, np.pi, 1001)
    fidelityList = np.zeros(len(phaseArray))
    for i, iPhase in enumerate(phaseArray):
        phaseA = np.exp(1j * iPhase)
        phaseB = np.exp(-1j * iPhase)
        if type == "odd":
            rhoBell = np.matrix([[0, 0, 0, 0], [0, 0.5, 0.5 * phaseA, 0], [0, 0.5 * phaseB, 0.5, 0], [0, 0, 0, 0]])
        elif type == "even":
            rhoBell = np.matrix([[0.5, 0, 0, 0.5 * phaseB], [0, 0, 0, 0], [0, 0, 0, 0], [0.5 * phaseA, 0, 0, 0.5]])
        else:
            raise NameError("invalid bell state type")
        rho=qt.Qobj(rho, [[2,2],[2,2]])
        rhoBell=qt.Qobj(rhoBell, [[2,2],[2,2]])

        fidelityList[i] = (qt.fidelity(rho, rhoBell)) ** 2

    if plot:
        plt.figure()
        plt.plot(phaseArray, fidelityList)

    print('fidelity of bell state is', np.max(fidelityList))
    return np.max(fidelityList), phaseArray[np.argmax(fidelityList)]

def calFidelityofGHZState(rho, nqbuits=3, hilberTruncated=2, plot=0):
    rho = qt.Qobj(rho)
    phaseArray = np.linspace(-np.pi, np.pi, 1001)
    fidelityList = np.zeros(len(phaseArray))

    for i, iPhase in enumerate(phaseArray):
        targetState = qt.ket2dm(1/np.sqrt(2) * (qt.ket("0"*nqbuits, hilberTruncated) + np.exp(-1j * iPhase) * qt.ket("1"*nqbuits, hilberTruncated)))
        fidelityList[i] = qt.fidelity(rho, targetState)**2

    if plot:
        plt.figure()
        plt.plot(phaseArray, fidelityList)

    print('fidelity of GHZ state is', np.max(fidelityList))
    return np.max(fidelityList), phaseArray[np.argmax(fidelityList)]


def calFidelityofTempState(rho, plot=0):
    rho = qt.Qobj(rho)
    phaseArray = np.linspace(-np.pi, np.pi, 1001)
    fidelityList = np.zeros(len(phaseArray))
    for i, iPhase in enumerate(phaseArray):
        targetState = qt.ket2dm(1 / np.sqrt(2) * (qt.ket("100", 2) + np.exp(-1j * iPhase) * qt.ket("011", 2)))
        fidelityList[i] = qt.fidelity(rho, targetState)**2

    if plot:
        plt.figure()
        plt.plot(phaseArray, fidelityList)

    print('fidelity of GHZ state is', np.max(fidelityList))
    return np.max(fidelityList), phaseArray[np.argmax(fidelityList)]

def calFidelityofWState(rho, plot=0):
    rho = qt.Qobj(rho)
    phaseArray = np.linspace(-np.pi, np.pi, 101)
    fidelityList = np.zeros([len(phaseArray), len(phaseArray)])
    for i, iPhase in enumerate(phaseArray):
        for j, jPhase in enumerate(phaseArray):
            targetState = qt.ket2dm(1 / np.sqrt(3) * (qt.ket("001", 2) + np.exp(-1j * iPhase) * qt.ket("010", 2) + np.exp(-1j * jPhase) * qt.ket("100", 2)))
            fidelityList[i, j] = qt.fidelity(targetState, rho)**2#np.abs((rho * targetState).tr())

    if plot:
        plt.figure()
        plt.pcolormesh(phaseArray, phaseArray, fidelityList)

    print('fidelity of W state is', np.max(fidelityList))
    return np.max(fidelityList)
def plotSingleQubitTomo(g_pcts_, xdata=["x", "y", "z"], plot=1):
    tomoNum_ = 2 * g_pcts_ - 1
    x_ = tomoNum_[1]
    y_ = tomoNum_[2]
    z_ = tomoNum_[0]

    rho_ = cal_density_matrix(x_, y_, z_)
    print(xdata, (x_, y_, z_))
    print('Tr(rho**2) = ', np.trace(rho_.dot(rho_)))
    print('SQRT(x**2 + y**2):', np.sqrt(x_ ** 2 + y_ ** 2))
    purity_ = np.trace(rho_.dot(rho_))

    if plot:
        plt.figure()
        plt.bar(xdata, 2 * g_pcts_ - 1)
        plt.ylim(-1, 1)
        plt.title('purity: ' + str(purity_))
    return (x_, y_, z_), (rho_, purity_, np.sqrt(x_ ** 2 + y_ ** 2))


def plotRho3D(rho, sim=0):
    column_names = ['gg', 'ge', 'eg', 'ee']
    row_names = ['gg', 'ge', 'eg', 'ee']

    fig = plt.figure('real')
    ax = Axes3D(fig)
    lx = len(rho[0])  # Work out matrix dimensions
    ly = len(rho[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.1, ypos + 0.1)
    xpos, ypos = xpos.ravel(), ypos.ravel()

    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = np.real(rho).flatten()
    if sim:
        cs = ['#66c2a520', '#fc8d6220', '#8da0cb20', '#e78ac320'] * ly
    else:
        cs = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'] * ly

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)

    ax.w_xaxis.set_ticks(np.arange(4) + 0.5)
    ax.w_yaxis.set_ticks(np.arange(4) + 0.5)
    ax.w_xaxis.set_ticklabels(column_names)
    ax.w_yaxis.set_ticklabels(row_names)
    ax.set_zlim(-0.5, 0.5)

    fig = plt.figure('imag')
    ax = Axes3D(fig)

    lx = len(rho[0])  # Work out matrix dimensions
    ly = len(rho[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.1, ypos + 0.1)
    xpos, ypos = xpos.ravel(), ypos.ravel()

    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = np.imag(rho).flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)

    ax.w_xaxis.set_ticks(np.arange(4) + 0.5)
    ax.w_yaxis.set_ticks(np.arange(4) + 0.5)
    ax.w_xaxis.set_ticklabels(column_names)
    ax.w_yaxis.set_ticklabels(row_names)
    ax.set_zlim(-0.5, 0.5)


def generateJointMsmtRes(g_list1, g_list2, plot=0, rho3dPlot=0):
    tomoList = ["ZI", "XI", "YI", "IZ", "IX", "IY", "ZZ", "ZX", "ZY", "XZ", "XX", "XY", "YZ", "YX", "YY"]
    res = [np.average(np.concatenate(g_list1[0:3])), np.average(np.concatenate(g_list1[3:6])),
           np.average(np.concatenate(g_list1[6:9])),
           np.average(np.concatenate(g_list2[::3])), np.average(np.concatenate(g_list2[1::3])),
           np.average(np.concatenate(g_list2[2::3])),
           np.average(g_list1[0] * g_list2[0]), np.average(g_list1[1] * g_list2[1]),
           np.average(g_list1[2] * g_list2[2]), np.average(g_list1[3] * g_list2[3]),
           np.average(g_list1[4] * g_list2[4]), np.average(g_list1[5] * g_list2[5]),
           np.average(g_list1[6] * g_list2[6]), np.average(g_list1[7] * g_list2[7]),
           np.average(g_list1[8] * g_list2[8])]
    if plot:
        plt.figure(figsize=(9, 4))
        plt.bar(tomoList, res, color='black')
        plt.ylim(-1, 1)
        plt.axvspan(-0.5, 2.5, alpha=0.4, color='red')
        plt.axvspan(2.5, 5.5, alpha=0.4, color='blue')
        plt.axvspan(5.5, 14.5, alpha=0.4, color='violet')

    if rho3dPlot:
        joint_dict = {tomoList[i]: res[i] for i in range(len(tomoList))}
        rho = cal_density_matrix_joint(joint_dict)
        plotRho3D(rho)
    return res

def generateJointMsmtRes_nQubit(full_tomo_series: List[str], experiment_series:List[str],
                           experiment_results:List[List], plot=0, target_rho=None):
    full_tomo = dict(zip(full_tomo_series, np.zeros(len(full_tomo_series))))

    def find_matched_tomo(target_tomo:str, available_tomo:List[str]):
        pick_idx = []
        for idx_, t_ in enumerate(available_tomo):
            match_ = True
            for m1, m2 in zip(target_tomo, t_):
                if m1 != "I" and m1 != m2:
                    match_ = False
            if match_:
                pick_idx.append(idx_)
        return pick_idx

    for tomo in full_tomo_series:
        matched_tomo_indices = find_matched_tomo(tomo, experiment_series)
        nonIqubits = [i for i, msmt in enumerate(tomo) if msmt!="I"]

        results = []
        for tomo_idx in matched_tomo_indices:
            r_ = 1
            for q_idx in nonIqubits:
                r_ *= experiment_results[q_idx][tomo_idx]
            results.append(r_)
        result = np.average(np.concatenate(results))
        full_tomo[tomo] = result


    if plot:
        plt.figure(figsize=(len(full_tomo)//2, 4))
        plt.bar(full_tomo_series, list(full_tomo.values()), color='black')
        plt.ylim(-1.1, 1.1)
        plt.xlim(-0.5, 62.5)
        plt.axvspan(-0.5, 2.5, alpha=0.3, color='red')
        plt.text(1, 0.5, "Q3")
        plt.axvspan(2.5, 5.5, alpha=0.1, color='red')
        plt.text(4, 0.5, "Q2")
        plt.axvspan(5.5, 8.5, alpha=0.3, color='red')
        plt.text(7, 0.5, "Q1")
        plt.axvspan(8.5, 17.5, alpha=0.3, color='blue')
        plt.text(13, 0.5, "Q2Q3")
        plt.axvspan(17.5, 26.5, alpha=0.1, color='blue')
        plt.text(22, 0.5, "Q1Q3")
        plt.axvspan(26.5, 35.5, alpha=0.3, color='blue')
        plt.text(31, 0.5, "Q1Q2")
        plt.axvspan(35.5, 62.5, alpha=0.3, color='yellow')
        plt.text(49, 0.5, "Q1Q2Q3")
        if target_rho is not None:
            emu_result = emulateTomoMSMT(target_rho, tomoOperatorDict(full_tomo_series))
            plt.bar(full_tomo_series, list(emu_result.values()), color=(0.8, 0.0, 0.0, 0.5), edgecolor='black')
    return full_tomo


def calculateRhoFromTomoRes(res_dict, plot=0):
    nBits = len(list(res_dict.keys())[0])
    tomoSeries = orderedTomoSeries(nBits)   
    tomoDict = tomoOperatorDict(tomoSeries)
    rhoDicts_recover = recoverRhoDicts(nBits, tomoDict)  
    rho = recoverRhoFromTomo(res_dict, rhoDicts_recover)
    if plot:
        qt.visualization.matrix_histogram_complex(rho)
    return rho

def emulateTomoMSMT(rho, tomoDict):
    results = {}
    for tomo, op in tomoDict.items():
        results[tomo] = (rho * op).tr()
    return results


def sweepPhasePlotSlider_3bit(res_rho):
    qdims = [[2, 2, 2], [2, 2, 2]]
    res_rho = qt.Qobj(res_rho, dims=qdims)
    full_tomo_series = orderedTomoSeries(3)
    phaseArray = np.linspace(-np.pi, np.pi, 101)
    new_state_list = np.zeros((len(phaseArray),len(phaseArray), 8, 8), dtype=complex)
    new_exp_list = np.zeros((len(phaseArray),len(phaseArray), 63))
    exop = list(tomoOperatorDict(full_tomo_series).values())
    for i, p1 in enumerate(tqdm(phaseArray)):
        for j, p2 in enumerate(phaseArray):
            rot_op = e2N(rz(p1), 3, 0) * e2N(rz(p2), 3, 1)
            newState =  rot_op * res_rho * rot_op.conj()
            new_state_list[i, j] = newState
            for k, op in enumerate(exop):
              new_exp_list[i, j, k] = np.real(qt.expect(newState, op))
    sld = sliderBarPlot(new_exp_list, dict(phase1=phaseArray / np.pi * 180, phase2=phaseArray / np.pi * 180), bar_labels=full_tomo_series)
    return sld

def sweepPhasePlotSlider_2bit(res_rho):
    qdims = [[2, 2], [2, 2]]
    res_rho = qt.Qobj(res_rho, dims=qdims)
    full_tomo_series = orderedTomoSeries(2)
    phaseArray = np.linspace(-np.pi, np.pi, 101)
    new_state_list = np.zeros((len(phaseArray),len(phaseArray), 4, 4), dtype=complex)
    new_exp_list = np.zeros((len(phaseArray), 15))
    exop = list(tomoOperatorDict(full_tomo_series).values())
    for j, p2 in enumerate(phaseArray):
        rot_op = e2N(rz(p2), 2, 1)
        print(res_rho, rot_op)
        newState =  rot_op * res_rho * rot_op.conj()
        new_state_list[j] = newState
        for k, op in enumerate(exop):
          new_exp_list[j, k] = np.real(qt.expect(newState, op))
    sld = sliderBarPlot(new_exp_list, dict(phase1=phaseArray / np.pi * 180), bar_labels=full_tomo_series)
    return sld

if __name__ == '__main__':
    ghz = 1/np.sqrt(2) * (qt.ket('000', 2) + np.exp(-1j * np.pi/2) * qt.ket('111', 2))
    w = qt.ket2dm(1/np.sqrt(3) * (qt.ket('001', 2) + qt.ket('010', 2) + qt.ket('100', 2)))
    badBell = 1/2* qt.ket2dm( (qt.ket('01', 2))) + 1/2* qt.ket2dm( (qt.ket('10', 2)))
    # calFidelityofWState(w, plot=0)
    # randomState =  1/3* qt.ket2dm( (qt.ket('001', 2))) + 1/3* qt.ket2dm( (qt.ket('010', 2))) + 1/3* qt.ket2dm( (qt.ket('100', 2)))
    xstate = (qt.ket('0') + qt.ket('1'))/np.sqrt(2)
    randomState = qt.ket2dm(qt.tensor(xstate, qt.bell_state('10'))) * 1/3 + qt.ket2dm(qt.tensor(qt.bell_state('10'), xstate)) * 1/3 + qt.ket2dm(1/np.sqrt(2) * (qt.tensor(qt.ket('0'), xstate, qt.ket('1')) + qt.tensor(qt.ket('1'), xstate, qt.ket('0')))) * 1/3
    print(qt.fidelity(w, randomState))
    # qt.visualization.matrix_histogram_complex(qt.ket2dm(ghz))
    plt.show()
