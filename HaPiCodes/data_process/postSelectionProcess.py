from typing import List, Callable, Union, Tuple, Dict
from typing_extensions import Literal
import warnings
import time

import matplotlib.pyplot as plt
import h5py
import lmfit as lmf
import math
import yaml
import numpy as np
from matplotlib.widgets import Slider

from nptyping import NDArray
import h5py
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib.patches import Circle, Wedge, Polygon
from scipy.ndimage import gaussian_filter as gf

from HaPiCodes.data_process import fittingAndDataProcess as fdp


class PostSelectionData_Base():
    def __init__(self, data_I: NDArray, data_Q: NDArray, msmtInfoDict: dict= None, selPattern: List = [1, 0]):
        """ A base class fro post selection data. doesn't specify the size of Hilbert space of the qubit.
        :param data_I:  I data
        :param data_Q:  Q data
        :param msmtInfoDict: dictionary from the measurement information yaml file
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        :param geLocation:  [g_x, g_y, e_x, e_y, g_r, e_r]
        """
        if msmtInfoDict is None:
            msmtInfoDict = {}

        self.data_I_raw = data_I
        self.data_Q_raw = data_Q
        self.selPattern = selPattern
        self.msmtInfoDict = msmtInfoDict

        data_max = np.max(np.abs(np.array([data_I, data_Q])))
        autoHistRange = [[-data_max, data_max],[-data_max, data_max]]
        self.msmtInfoDict["histRange"] = self.msmtInfoDict.get('histRange', autoHistRange)

        n_avg = len(data_I)
        pts_per_exp = len(data_I[0])
        msmt_per_sel = len(selPattern)
        if pts_per_exp % msmt_per_sel != 0:
            raise ValueError(f"selPattern is not valid. the length of selPattern {len(selPattern)} is no a factor of "
                             f"points per experiment {pts_per_exp}")
        n_sweep = int(np.round(pts_per_exp // msmt_per_sel))  # e.g. len(xData)

        self.sel_data_msk = np.array(selPattern, dtype=bool)
        self.exp_data_msk = ~ self.sel_data_msk
        self.data_I = data_I.reshape((n_avg, n_sweep, msmt_per_sel))
        self.data_Q = data_Q.reshape((n_avg, n_sweep, msmt_per_sel))

        self.I_sel = self.data_I[:, :, self.sel_data_msk]  # gather selection data
        self.Q_sel = self.data_Q[:, :, self.sel_data_msk]
        self.I_exp = self.data_I[:, :, self.exp_data_msk]  # gather experiment data
        self.Q_exp = self.data_Q[:, :, self.exp_data_msk]

    def state_split_line(self, x1, y1, x2, y2, x):
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        k_ = -(x1 - x2) / (y1 - y2)
        return k_ * (x - center_x) + center_y

    def mask_state_by_circle(self, sel_idx: int, state_x: float, state_y: float, state_r: float,
                             plot: Union[bool, int] = True, state_name: str = ""):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param state_x: x position of the state on IQ plane
        :param state_y: y position of the state on IQ plane
        :param state_r: size of the selection circle
        :param plot: if true, plot selection
        :param state_name: name of the state, will be used in the plotting title.
        :return:
        """
        if self.selPattern[sel_idx] != 1:
            raise ValueError(f"sel_idx must be a position with value 1 in selPattern {self.selPattern}")
        idx_ = np.where(np.where(np.array(self.selPattern) == 1)[0] == sel_idx)[0][0]
        I_sel_ = self.I_sel[:, :, idx_]
        Q_sel_ = self.Q_sel[:, :, idx_]
        mask = (I_sel_ - state_x) ** 2 + (Q_sel_ - state_y) ** 2 < (state_r) ** 2
        if plot:
            plt.figure(figsize=(7, 7))
            plt.title(f'{state_name} state selection range')
            plt.hist2d(I_sel_.flatten(), Q_sel_.flatten(), bins=101, range=self.msmtInfoDict['histRange'])
            theta = np.linspace(0, 2 * np.pi, 201)
            plt.plot(state_x + state_r * np.cos(theta), state_y + state_r * np.sin(theta), color='r')
        return mask

    def sel_data(self, mask, plot=True):
        self.I_vld = []
        self.Q_vld = []
        for i in range(self.I_exp.shape[1]):
            for j in range(self.I_exp.shape[2]):
                self.I_vld.append(self.I_exp[:, i, j][mask[:, i]])
                self.Q_vld.append(self.Q_exp[:, i, j][mask[:, i]])
        if plot:
            plt.figure(figsize=(7, 7))
            selNum = np.average(list(map(len, self.I_vld)))
            plt.title('experiment pts after selection\n'+"sel%: "+ str(selNum / len(self.data_I_raw)))
            plt.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=101, range=self.msmtInfoDict['histRange'])
            print("sel%: " + str(selNum / len(self.data_I_raw)))

        selNum = np.average(list(map(len, self.I_vld)))
        print("sel%: " + str(selNum / len(self.data_I_raw)))
        return self.I_vld, self.Q_vld


class PostSelectionData(PostSelectionData_Base):
    def __init__(self, data_I: NDArray, data_Q: NDArray, msmtInfoDict: dict=None, selPattern: List = [1, 0],
                 geLocation: List[float] = "AutoFit", plotGauFitting=True, fitGuess: dict=None, histRange=None):
        super().__init__(data_I, data_Q, msmtInfoDict, selPattern)
        """ A post selection data class that assumes a qubit has two possible states
        :param data_I:  I data
        :param data_Q:  Q data
        :param msmtInfoDict: dictionary from the measurement information yaml file
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        :param geLocation:  [g_x, g_y, e_x, e_y, g_r, e_r]
        :param fitGuess: 
        """

        # fit for g, e gaussian if g/e state location is not provided
        if geLocation == "AutoFit":
            fitData = np.array([self.I_sel.flatten(), self.Q_sel.flatten()])
            if plotGauFitting == 1:
                mute_ = 0
            else:
                mute_ = 1
            fitRes = fdp.fit_Gaussian(fitData, plot=plotGauFitting, mute=mute_, fitGuess=fitGuess, histRange=histRange)
            sigma_g = np.sqrt(fitRes[4] ** 2 + fitRes[5] ** 2)
            sigma_e = np.sqrt(fitRes[6] ** 2 + fitRes[7] ** 2)

            if plotGauFitting:
                print(fitRes[0], ',', fitRes[1], ',', fitRes[2], ',', fitRes[3], ',', sigma_g, ',', sigma_e)
            geLocation = [*fitRes[:4], sigma_g, sigma_e]
        self.geLocation = geLocation
        self.g_x, self.g_y, self.e_x, self.e_y, self.g_r, self.e_r = self.geLocation

    def ge_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.e_x, self.e_y, x)

    def mask_g_by_circle(self, sel_idx: int = 0, circle_size: float = 1, plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of g_r (sigma of g state gaussion blob)
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.g_x, self.g_y, self.g_r * circle_size,
                                         plot, "g")
        return mask

    def mask_e_by_circle(self, sel_idx: int = 0, circle_size: float = 1, plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of e_r (sigma of e state gaussion blob)
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.e_x, self.e_y, self.e_r * circle_size,
                                         plot, "e")
        return mask

    def mask_g_by_line(self, sel_idx: int = 0, line_rotate: float = 0, line_shift: float = 0,
                       plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param line_rotate: rotation angle the split line in counter clockwise direction, in unit of rad. Zero angle
            is the the perpendicular bisector of g and e.
        :param line_shift: shift the split line along the e -> g direction, in unit of half ge distance
        :param plot:
        :return:
        """
        if self.selPattern[sel_idx] != 1:
            raise ValueError(f"sel_idx must be a position with value 1 in selPattern {self.selPattern}")
        idx_ = np.where(np.where(np.array(self.selPattern) == 1)[0] == sel_idx)[0][0]
        I_sel_ = self.I_sel[:, :, idx_]
        Q_sel_ = self.Q_sel[:, :, idx_]

        def rotate_ge_line_(x, theta):
            k_ = -(self.g_x - self.e_x) / (self.g_y - self.e_y)
            x0_ = (self.g_x + self.e_x) / 2
            y0_ = (self.g_y + self.e_y) / 2
            return (x - x0_) * np.tan(np.arctan(k_) + theta) + y0_

        rotate_split_line = lambda x: rotate_ge_line_(x, line_rotate)
        shift_split_line = lambda x: rotate_split_line(x - line_shift * 0.5 * (self.g_x - self.e_x)) \
                                     + line_shift * 0.5 * (self.g_y - self.e_y)
        if self.g_y < self.e_y:
            mask = Q_sel_ < shift_split_line(I_sel_)
        else:
            mask = Q_sel_ > shift_split_line(I_sel_)

        if plot:
            plt.figure(figsize=(7, 7))
            plt.title('g state selection range')
            h, xedges, yedges, image = plt.hist2d(I_sel_.flatten(), Q_sel_.flatten(), bins=101,
                                                  range=self.msmtInfoDict['histRange'])
            plt.plot(xedges, shift_split_line(xedges), color='r')
            plt.plot([(self.g_x + self.e_x) / 2], [(self.g_y + self.e_y) / 2], "*")
        return mask

    def cal_g_pct(self, plot=False, correct=False):
        g_pct_list = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            if self.g_y < self.e_y:
                mask = Q_v < self.ge_split_line(I_v)
            else:
                mask = Q_v > self.ge_split_line(I_v)
            try:
                g_pct_list.append(len(I_v[mask]) / n_pts)
            except ZeroDivisionError:
                warnings.warn("! no valid point, this is probably because the gaussian fitting, please double check system and fitting function")
                g_pct_list.append(1)
        if plot:
            plt.figure(figsize=(7, 7))
            h, xedges, yedges, image = plt.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=101,
                                                  range=self.msmtInfoDict['histRange'], cmap='hot')
            plt.plot(xedges, self.ge_split_line(xedges), color='r')
            plt.plot([(self.g_x + self.e_x) / 2], [(self.g_y + self.e_y) / 2], "*")

        if correct:
            e0, e1 = self.msmtInfoDict["MSMTError"]
            g_pct_list = (np.array(g_pct_list) - e1) / (e0 - e1)
        return np.array(g_pct_list)

    def cal_stateForEachMsmt(self):
        stateForEachMsmt = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            if self.g_y < self.e_y:
                mask = Q_v < self.ge_split_line(I_v)
            else:
                mask = Q_v > self.ge_split_line(I_v)
            state = mask * 2 - 1
            stateForEachMsmt.append(state)

        return stateForEachMsmt


class PostSelectionData_gef(PostSelectionData_Base):
    def __init__(self, data_I: NDArray, data_Q: NDArray, msmtInfoDict: dict=None, selPattern: List = [1, 0],
                 gefLocation: List[float] = "AutoFit", plotGauFitting=True, fitGuess=None):
        super().__init__(data_I, data_Q, msmtInfoDict, selPattern)
        """ A post selection data class that assumes a qubit has three possible states
        :param data_I:  I data
        :param data_Q:  Q data
        :param msmtInfoDict: dictionary from the measurement information yaml file
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        :param geLocation:  [g_x, g_y, e_x, e_y, f_x, f_y, g_r, e_r, f_r]
        """

        # fit for g, e, f gaussian if g/e/f state location is not provided
        if gefLocation == "AutoFit":
            fitData = np.array([self.I_sel.flatten(), self.Q_sel.flatten()])
            fitRes = fdp.fit_Gaussian(fitData, blob=3, plot=plotGauFitting, mute=1, fitGuess=fitGuess)
            sigma_g = np.sqrt(fitRes[6] ** 2 + fitRes[7] ** 2)
            sigma_e = np.sqrt(fitRes[8] ** 2 + fitRes[9] ** 2)
            sigma_f = np.sqrt(fitRes[10] ** 2 + fitRes[11] ** 2)
            gefLocation = [*fitRes[:6], sigma_g, sigma_e, sigma_f]
            print(fitRes[0], ',', fitRes[1], ',', fitRes[2], ',', fitRes[3], ',', fitRes[4], ',', fitRes[5], ',', sigma_g, ',', sigma_e, ',', sigma_f)
        self.gefLocation = gefLocation
        self.g_x, self.g_y, self.e_x, self.e_y, self.f_x, self.f_y, self.g_r, self.e_r, self.f_r = self.gefLocation

        # find the circumcenter of the three states
        d_ = 2 * (self.g_x * (self.e_y - self.f_y) + self.e_x * (self.f_y - self.g_y)
                  + self.f_x * (self.g_y - self.e_y))
        self.ext_center_x = ((self.g_x ** 2 + self.g_y ** 2) * (self.e_y - self.f_y)
                         + (self.e_x ** 2 + self.e_y ** 2) * (self.f_y - self.g_y)
                         + (self.f_x ** 2 + self.f_y ** 2) * (self.g_y - self.e_y)) / d_
        self.ext_center_y = ((self.g_x ** 2 + self.g_y ** 2) * (self.f_x - self.e_x)
                         + (self.e_x ** 2 + self.e_y ** 2) * (self.g_x - self.f_x)
                         + (self.f_x ** 2 + self.f_y ** 2) * (self.e_x - self.g_x)) / d_

        self.in_center_x = np.mean([self.g_x, self.e_x, self.f_x])
        self.in_center_y = np.mean([self.g_y, self.e_y, self.f_y])

    def ge_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.e_x, self.e_y, x)

    def ef_split_line(self, x):
        return self.state_split_line(self.e_x, self.e_y, self.f_x, self.f_y, x)

    def gf_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.f_x, self.f_y, x)

    def mask_g_by_circle(self, sel_idx: int = 0, circle_size: float = 1, plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of g_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.g_x, self.g_y, self.g_r * circle_size,
                                         plot, "g")
        return mask

    def mask_e_by_circle(self, sel_idx: int = 0, circle_size: float = 1, plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of e_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.e_x, self.e_y, self.e_r * circle_size,
                                         plot, "e")
        return mask

    def mask_f_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of f_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.f_x, self.f_y, self.f_r * circle_size,
                                         plot, "f")
        return mask

    def cal_g_pct(self, plot=True):
        g_pct_list = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)

        if plot:
            plt.figure(figsize=(7, 7))
            h, xedges, yedges, image = plt.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=101,
                                                  range=self.msmtInfoDict['histRange'])

            def get_line_range_(s1, s2):
                """get the x range to plot for the line that splits three states"""
                x12 = np.mean([getattr(self, f"{s1}_x"), getattr(self, f"{s2}_x")])
                y12 = np.mean([getattr(self, f"{s1}_y"), getattr(self, f"{s2}_y")])
                
                v1 = [self.ext_center_x- x12, self.ext_center_y-y12 ]
                v2 = [self.in_center_x- x12, self.in_center_y-y12 ]
                if (np.dot(v1, v2) > 0 and v1[0] >0)  or (np.dot(v1, v2) < 0 and v1[0] < 0):
                    return np.array([xedges[0], self.ext_center_x])
                else:
                    return np.array([self.ext_center_x, xedges[-1]])

            x_l_ge = get_line_range_("g", "e")
            x_l_ef = get_line_range_("e", "f")
            x_l_gf = get_line_range_("g", "f")

            plt.plot(x_l_ge, self.ge_split_line(x_l_ge), color='r')
            plt.plot(x_l_ef, self.ef_split_line(x_l_ef), color='g')
            plt.plot(x_l_gf, self.gf_split_line(x_l_gf), color='b')
            plt.plot([self.ext_center_x], [self.ext_center_y], "*")
        return np.array(g_pct_list)

    def cal_gef_pct(self, plot=True):
        g_pct_list = []
        e_pct_list = []
        f_pct_list = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)
            e_mask = np.where(state_ == 1)[0]
            e_pct_list.append(len(e_mask) / n_pts)
            f_mask = np.where(state_ == 2)[0]
            f_pct_list.append(len(f_mask) / n_pts)

        if plot:
            plt.figure(figsize=(7, 7))
            h, xedges, yedges, image = plt.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=101,
                                                  range=self.msmtInfoDict['histRange'])

            def get_line_range_(s1, s2):
                """get the x range to plot for the line that splits three states"""
                x12 = np.mean([getattr(self, f"{s1}_x"), getattr(self, f"{s2}_x")])
                y12 = np.mean([getattr(self, f"{s1}_y"), getattr(self, f"{s2}_y")])

                v1 = [self.ext_center_x - x12, self.ext_center_y - y12]
                v2 = [self.in_center_x - x12, self.in_center_y - y12]
                if (np.dot(v1, v2) > 0 and v1[0] > 0) or (np.dot(v1, v2) < 0 and v1[0] < 0):
                    return np.array([xedges[0], self.ext_center_x])
                else:
                    return np.array([self.ext_center_x, xedges[-1]])

            x_l_ge = get_line_range_("g", "e")
            x_l_ef = get_line_range_("e", "f")
            x_l_gf = get_line_range_("g", "f")

            plt.plot(x_l_ge, self.ge_split_line(x_l_ge), color='r')
            plt.plot(x_l_ef, self.ef_split_line(x_l_ef), color='g')
            plt.plot(x_l_gf, self.gf_split_line(x_l_gf), color='b')
            plt.plot([self.ext_center_x], [self.ext_center_y], "*")
        return np.array([np.array(g_pct_list), np.array(e_pct_list), np.array(f_pct_list)])

    def cal_stateForEachMsmt(self, gef=0):
        warnings.warn("now we consider f as e, didn't implement 3 states calculation yet")
        g_pct_list = []
        e_pct_list = []
        f_pct_list = []
        stateForEachMsmt = []

        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)
            e_mask = np.where(state_ == 1)[0]
            e_pct_list.append(len(e_mask) / n_pts)
            f_mask = np.where(state_ == 2)[0]
            f_pct_list.append(len(f_mask) / n_pts)

            stateForSingleMsmt = state_.copy()
            stateForSingleMsmt[np.where(state_ == 0)[0]] = 1
            stateForSingleMsmt[np.where(state_ != 0)[0]] = -1
            stateForEachMsmt.append(stateForSingleMsmt)
        return stateForEachMsmt

if __name__ == "__main__":
    directory = r'N:\Data\Tree_3Qubits\QCSWAP\Q3C3\20210111\\'
    # directory = r'D:\Lab\Code\PostSelProcess_dev\\'
    fileName = '10PiPulseTest'
    f = h5py.File(directory + fileName, 'r')
    Idata = np.real(f["rawData"])[:-1]
    Qdata = np.imag(f["rawData"])[:-1]
    msmtInfoDict = yaml.safe_load(open(directory + fileName + ".yaml", 'r'))

    t0 = time.time()
    IQsel = PostSelectionData_gef(Idata, Qdata, msmtInfoDict, [1, 0],
                                  gefLocation=[-9000, -9500, -11000, -1500, -500, -700, 3000, 3000, 3000])

    # mask0 = IQsel.mask_g_by_line(0, line_shift=0, plot=True)
    mask0 = IQsel.mask_g_by_circle(0, circle_size=1, plot=True)
    I_vld, Q_vld = IQsel.sel_data(mask0, plot=True)
    # I_avg, Q_avg = fdp.average_data(I_vld, Q_vld, axis0_type="xData")
    # I_rot, Q_rot = fdp.rotateData(I_avg, Q_avg, plot=0)
    g_pct = IQsel.cal_g_pct()

    xData = np.arange(10)
    # plt.figure(figsize=(7, 7))
    # plt.plot(xData, I_avg)
    plt.figure(figsize=(7, 7))
    plt.plot(xData, g_pct)

    print("time: ", time.time() - t0)