# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:04:54 2019

@author: chao
"""
from typing import Union, List, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
from matplotlib.animation import FuncAnimation


def _indexData(data: Union[List, np.array], dim: Union[List, np.array]):
    """ get the data at dimension 'dim' of the input data. Basically just make
        the list data can be indexed like np.array data.

    :param data: input data
    :param dim: list of indexs for each dimension
    :return:
    """
    d = data
    for i in dim:
        d = d[int(i)]
    return d


def sliderHist2d(data_I: Union[List, np.array], data_Q: Union[List, np.array],
           axes_dict: dict, callback: Callable = None, **hist2dArgs) -> List[Slider]:
    """Create a slider plot widget. The caller need to maintain a reference to
    the returned Slider objects to keep the widget activate

    :param data_I:
    :param data_Q:
    :param axes_dict: a dictionary that contains the data of each axis
    :param hist2dArgs:
    :return: list of Slider objects.
    """
    hist2dArgs["bins"] = hist2dArgs.get("bins", 101)
    hist2dArgs["range"] = hist2dArgs.get("range", [[-2e4, 2e4], [-2e4, 2e4]])

    # initial figure
    nAxes = len(axes_dict)
    dataI0 = _indexData(data_I, np.zeros(nAxes))
    dataQ0 = _indexData(data_Q, np.zeros(nAxes))
    fig = plt.figure(figsize=(7, 7 + nAxes * 0.3))
    callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
    plt.subplots_adjust(bottom=nAxes * 0.3 / (7 + nAxes * 0.3) + 0.1)
    plt.subplot(1, 1, 1)
    plt.hist2d(dataI0, dataQ0, **hist2dArgs)
    ax = plt.gca()
    ax.set_aspect(1)
    # generate sliders
    axcolor = 'lightgoldenrodyellow'
    sld_list = []
    for idx, (k, v) in enumerate(axes_dict.items()):
        ax_ = plt.axes([0.2, (nAxes - idx) * 0.04, 0.6, 0.03], facecolor=axcolor)
        sld_ = Slider(ax_, k, 0, len(v) - 1, valinit=0, valstep=1)
        sld_list.append(sld_)

    # update funtion
    def update(val):
        sel_dim = []
        ax_val_list = []
        for i in range(nAxes):
            ax_name = sld_list[i].label.get_text()
            ax_idx = int(sld_list[i].val)
            sel_dim.append(int(ax_idx))
            ax_val = np.round(axes_dict[ax_name][ax_idx], 5)
            ax_val_list.append(ax_val)
            sld_list[i].valtext.set_text(str(ax_val))
        newI = _indexData(data_I, sel_dim)
        newQ = _indexData(data_Q, sel_dim)
        ax.cla()
        ax.hist2d(newI, newQ, **hist2dArgs)
        # print callback result on top of figure
        if callback is not None:
            result = callback(newI, newQ, *ax_val_list)
            callback_text.set_text(callback.__name__ + f": {result}")
        fig.canvas.draw_idle()

    for i in range(nAxes):
        sld_list[i].on_changed(update)
    return sld_list


def sliderPColorMesh(xdata, ydata, zdata,
                     axes_dict: dict, callback: Callable = None, **pColorMeshArgs):
    # raise NotImplementedError("this function is still under developing")
    pColorMeshArgs["shading"] = pColorMeshArgs.get("shading", "auto")
    pColorMeshArgs["vmin"] = pColorMeshArgs.get("vmin", np.min(zdata))
    pColorMeshArgs["vmax"] = pColorMeshArgs.get("vmax", np.max(zdata))
    # initial figure
    nAxes = len(axes_dict)
    zdata0 = _indexData(zdata, np.zeros(nAxes))
    fig = plt.figure(figsize=(7, 7 + nAxes * 0.3))

    callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
    plt.subplots_adjust(bottom=nAxes * 0.3 / (7 + nAxes * 0.3) + 0.1)
    plt.subplot(1, 1, 1)
    pcm = plt.pcolormesh(xdata, ydata, zdata0.T, **pColorMeshArgs)
    ax1 = plt.gca()
    fig.colorbar(pcm, ax=ax1)
    axcolor = 'lightgoldenrodyellow'

    # generate sliders
    sld_list = []
    for idx, (k, v) in enumerate(axes_dict.items()):
        ax_ = plt.axes([0.15, (nAxes - idx) * 0.04, 0.6, 0.03], facecolor=axcolor)
        sld_ = Slider(ax_, k, 0, len(v) - 1, valinit=0, valstep=1)
        sld_list.append(sld_)

    # update funtion
    def update(val):
        sel_dim = []
        ax_val_list = []
        for i in range(nAxes):
            ax_name = sld_list[i].label.get_text()
            ax_idx = int(sld_list[i].val)
            sel_dim.append(int(ax_idx))
            ax_val = np.round(axes_dict[ax_name][ax_idx], 5)
            ax_val_list.append(ax_val)
            sld_list[i].valtext.set_text(str(ax_val))
        newZdata = _indexData(zdata, sel_dim)
        ax1.cla()
        pcm = ax1.pcolormesh(xdata, ydata, newZdata.T, **pColorMeshArgs)
        # print callback result on top of figure
        if callback is not None:
            result = callback(xdata, ydata, newZdata, *ax_val_list)
            callback_text.set_text(callback.__name__ + f": {result}")
        fig.canvas.draw_idle()

    for i in range(nAxes):
        sld_list[i].on_changed(update)

    return sld_list

def AnimatePColorMesh(xdata, ydata, zdata,
                         axes_dict: dict, fileName="", **pColorMeshArgs):
    if len(axes_dict.keys()) > 1:
        raise NotImplementedError("this function (axis > 1) is still under developing")
    pColorMeshArgs["shading"] = pColorMeshArgs.get("shading", "auto")
    pColorMeshArgs["vmin"] = pColorMeshArgs.get("vmin", np.min(zdata))
    pColorMeshArgs["vmax"] = pColorMeshArgs.get("vmax", np.max(zdata))
    # initial figure
    nAxes = len(axes_dict)
    zdata0 = _indexData(zdata, np.zeros(nAxes))
    fig = plt.figure(figsize=(7, 7 + nAxes * 0.3))

    callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
    plt.subplots_adjust(bottom=nAxes * 0.3 / (7 + nAxes * 0.3) + 0.1)
    plt.subplot(1, 1, 1)
    pcm = plt.pcolormesh(xdata, ydata, zdata0.T, **pColorMeshArgs)
    ax1 = plt.gca()
    fig.colorbar(pcm, ax=ax1)
    axcolor = 'lightgoldenrodyellow'
    for k, v in axes_dict.items():
        sweepLabel = k
        sweepValue = v
    # update funtion
    def update(val):
        sel_dim = val
        newZdata = _indexData(zdata, [sel_dim])
        ax1.cla()
        pcm = ax1.pcolormesh(xdata, ydata, newZdata.T, **pColorMeshArgs)
        ax1.set_title(sweepLabel + ": " + str(sweepValue[val]))
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, update, frames=np.arange(len(sweepValue)), interval=500)
    if fileName != "":
        anim.save(fileName+".gif", dpi=80, writer='imagemagick')
    return anim

if __name__ == '__main__':
    axis1 = np.arange(10)
    axis2 = np.arange(10.124564, 20)*1e6
    axis3 = np.arange(20, 30)
    axis4 = np.arange(30, 40)
    data_len = 100
    rdata_I = np.random.rand(len(axis1), len(axis2), len(axis3), len(axis4), data_len)
    rdata_Q = np.random.rand(len(axis1), len(axis2), len(axis3), len(axis4), data_len)

    def avgIQ(dataI, dataQ, *args):
        return(np.average(dataI), np.average(dataQ))

    axes_dict = dict(axis1=axis1, axis2=axis2, axis3=axis3, axis4=axis4)
    slds = sliderHist2d(rdata_I, rdata_Q, axes_dict, avgIQ, range=[[0, 1], [0, 1]])

