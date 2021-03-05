import numpy as np

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
    rho[0, 3] = (jd['XX'] - jd['YY']) / 4.0 - 1j * (jd['XX'] + jd['YY']) / 4.0
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
    return  SAB - SA - SB

def calFidelityofBellState(rho, plot=0):
    rho = np.matrix(rho)
    phaseArray = np.linspace(-np.pi, np.pi, 1001)
    fidelityList = np.zeros(len(phaseArray))
    for i, iPhase in enumerate(phaseArray):
        phaseA = np.exp(1j * iPhase)
        phaseB = np.exp(-1j * iPhase)
        rhoBell = np.matrix([[0, 0, 0, 0], [0, 0.5, 0.5 * phaseA, 0], [0, 0.5 * phaseB, 0.5, 0], [0, 0, 0, 0]])

        fidelityList[i] = np.trace(rho * rhoBell)

    if plot:
        plt.figure()
        plt.plot(phaseArray, fidelityList)

    print('fidelity of bell state is', np.max(fidelityList))
    return np.max(fidelityList), phaseArray[np.argmax(fidelityList)]

