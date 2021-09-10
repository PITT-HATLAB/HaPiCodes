import h5py
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
# from plottr.data.datadict import DataDictBase, MeshgridDataDict
from HaPiCodes.data_process.IQdata import IQData

def saveArbitraryData(fileFullDir, **data):
    file_name_splt = fileFullDir.split("\\")
    file_path = "\\".join(file_name_splt[:-1])
    file_name = file_name_splt[-1]
    #create dir if not exist
    try:
        path = pathlib.Path(file_path)
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(error)
    # add duplicate index if file name exist
    duplicate_idx = 0
    save_name = file_name
    while save_name in os.listdir(path):
        save_name = file_name + f"_{duplicate_idx}"
        duplicate_idx += 1

    fileSave = h5py.File(file_path + "\\" + save_name, 'w-')
    for k, v in data.items():
        v = np.array(v)
        fileSave.create_dataset(k, data=v)

    fileSave.close()

def loadH5toDict(fileFullDir):
    fileLoad = h5py.File(fileFullDir, 'r')
    data_dict = {}
    for k in fileLoad.keys():
        data_dict[k] = fileLoad[k][()]
    return data_dict
