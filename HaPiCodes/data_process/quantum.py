import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        fidelityList[i] = np.abs(np.trace(rho * rhoBell))

    if plot:
        plt.figure()
        plt.plot(phaseArray, fidelityList)

    print('fidelity of bell state is', np.max(fidelityList))
    return np.max(fidelityList), phaseArray[np.argmax(fidelityList)]


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
