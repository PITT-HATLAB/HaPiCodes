import os
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

import qutip as qt
from qutip.parallel import parallel_map
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as color
from tqdm import tqdm
import h5py

PI = np.pi

opts = qt.Options(nsteps=1e6, atol=1e-14, rtol=1e-12)
p_bar = None  # qt.ui.TextProgressBar()

dim = 10


def fock_n(n):
    return qt.fock(dim, n)


def exp_N(n, state_):
    return (qt.expect(qt.ket2dm(fock_n(n)), state_))
    # return(qt.expect(qt.ket2dm(eigenS[n]), state_))


psi0 = fock_n(0)
a = qt.destroy(dim)
N0 = qt.num(dim)

omega_a = 4.5 * 2 * PI
alpha = -0.15 * 2 * PI
omega_d = omega_a / 3
drive_coeff = 12

H0 = (omega_a - 2 * alpha) * a.dag() * a + alpha / 6 * (a.dag() + a) ** 4 - alpha / 1200 * (a.dag() * a) ** 6
eigenS = H0.eigenstates()[1]


# psi0 = eigenS[0]


def sweepDriveFreqAndTime(omega_d_, drive_length):
    def drive(t, args):
        return drive_coeff * np.cos(omega_d_ * t) * boxDrive(t, 50, 0.001, drive_length)

    t_list = np.linspace(0, drive_length, int(drive_length * 10 + 1))
    H = [H0, [a.dag() + a, drive]]
    solve_result = qt.mesolve(H, psi0, t_list, options=opts, progress_bar=p_bar)
    result_states_ = solve_result.states
    exp_n = np.zeros(dim)
    for j in range(dim):
        exp_n[j] = exp_N(j, result_states_[-1])
    return exp_n


def boxDrive(t, rampWidth, cutOffset, width):
    """

    :param t: time list
    :param rampWidth: number of points from cutOffset to 0.95 amplitude
    :param cutOffset: the initial offset to cut on the tanh Function
    :param width:
    :return:
    """
    c0_ = np.arctanh(2 * cutOffset - 1)
    c1_ = np.arctanh(2 * 0.95 - 1)
    k_ = (c1_ - c0_) / rampWidth
    return 0.5 * (np.tanh(k_ * t + c0_) -
                  np.tanh(k_ * (t - width) - c0_))


def saveData(filePath, fileName, **data):
    f = h5py.File(filePath + fileName, "w")
    for s, var in data.items():
        f.create_dataset(s, data=var)
    f.close()


def sweepDriveFreqAndTime2D(omegaAndTime):
    omega_d_ = omegaAndTime[0]
    drive_length = omegaAndTime[1]

    def drive(t, args):
        return drive_coeff * np.cos(omega_d_ * t) * boxDrive(t, 20, 0.001, drive_length)

    t_list = np.linspace(0, drive_length, int(drive_length * 10 + 1))
    H = [H0, [a.dag() + a, drive]]
    solve_result = qt.mesolve(H, psi0, t_list, options=opts, progress_bar=p_bar)
    result_states_ = solve_result.states
    exp_n = np.zeros(dim)
    for j in range(dim):
        exp_n[j] = exp_N(j, result_states_[-1])
    return exp_n

def loadData(filePath, fileName):
    file_ = h5py.File(filePath + fileName, 'r')
    dataDict = {}
    for k, v in file_.items():
        dataDict[k] = v[()]
    file_.close()
    return dataDict


if __name__ == "__main__":

    dataDict = loadData(r"./SimData//",
                 f"DriveFactor_12_50Ramp_with6Order_frequencyShift")
    exp = dataDict['data']
    dim = dataDict['dim']
    time_list = dataDict['time_list']
    o_d_list = (omega_d + np.linspace(-0.115, -0.075, 41) * 2 * PI)

    res2D = np.array(exp).reshape(len(time_list), len(o_d_list), dim)
    plt.figure()
    plt.pcolormesh((o_d_list - omega_d) / 2 / PI, time_list, res2D[:, :, 0], shading='auto')
    plt.colorbar()
    #
    # # linecut------------------------------------
    # plt.figure()
    # drive_freq_idx = np.argmin(np.abs((o_d_list-omega_d)/2/PI+0.09))
    # for i in range(dim):
    #     plt.plot(time_list, res2D[:, drive_freq_idx, i])
    #

    # ----------drive-------------------
    # time_list = np.linspace(0, 400, 201)
    # pulseData = boxDrive(time_list,  20, 0.001, time_list[-1])
    # plt.figure()
    # plt.plot(time_list, pulseData)

    # plt.figure()
    # plt.rcParams["axes.linewidth"] = 4
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #
    # ax.tick_params(direction='in', width=4, length=12, color='black',
    #                labelsize=20)
    # colors = [color.hex2color('#FF0000'), color.hex2color('#FFFFFF'),
    #           color.hex2color('#0000FF')]
    # colors = colors[::-1]
    # cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
    # plt.pcolormesh((o_d_list - omega_d) / 2 / PI * 1000, time_list,
    #                exp_n_list[:, :, 0].T * 2 - 1, cmap=cmap, vmin=-1, vmax=1)
    # ax.set_xticks(np.array([-37, -36, -35, -34, -33, -32]))
    # #    ax.set_yticks(np.array([np.min(populationPitch), np.max(populationPitch)]))
    # ax.set_yticks(np.array([200, 400, 600, ]))
    # plt.colorbar()
