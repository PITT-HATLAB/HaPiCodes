import h5py
import os
import time
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



# ---------------------- data saving helpers -------------------------------------
def getPulseParams(infoDict, pulseName, args):
    pulseParams = infoDict["pulseParams"][pulseName]
    return {k: np.round(pulseParams[k],5) for k in args}

def paramDictToStr(paramDict, tag=""):
    s = ""
    for k, v in paramDict.items():
        s += f"-{tag}_{k}_{np.round(v, 6)}"
    return s

def get_date():
    return time.strftime("%Y-%m-%d")

def toYamlFriendly(v):
    if type(v) == str:
        vv = v
        strName = v
        return vv, strName
    if type(v) == dict:
        vv = {}
        strName = "{"
        for k_, v_ in v.items():
            vv_, sn = toYamlFriendly(v_)
            strName += f"{k_}_{sn}-"
            vv[k_] = vv_
        strName = strName[:-1] + "}"
        return vv, strName
    try:
        if len(v) > 0:
            try:
                vv = v.tolist()
            except AttributeError:
                vv = v
            try:
                start = np.round(v[0], 5)
                stop = np.round(v[-1], 5)
            except TypeError:
                start = v[0]
                stop = v[-1]
            strName = f"{start}To{stop}"
            return vv, strName
    except TypeError:
        vv = float(v)
        strName = f"{v}"
        return vv, strName

def createSweepDir(baseDir, preFix="", msmtInfoDict=None, **sweepAxes):
    folderName = preFix
    sweepDict = {}
    for k, v in sweepAxes.items():
        vv, strName = toYamlFriendly(v)
        folderName += f"-{k}_{strName}"
        sweepDict[k] = vv

    fullDir = baseDir + rf"{folderName}\\"

    try:
        path = pathlib.Path(fullDir)
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(error)

    with open(fullDir+'sweepAxes.yml', 'w') as outfile:
        yaml.dump(sweepDict, outfile, default_flow_style=True)

    if msmtInfoDict is not None:
        with open(fullDir+'msmtInfo.yml', 'w') as outfile:
            yaml.dump(msmtInfoDict, outfile, default_flow_style=True)

    return fullDir


def yamlItem2npArray(v):
    if type(v) == list:
        if type(v[0]) == str:
            return v
        else:
            return np.array(v)
    elif type(v) == dict:
        dd = {}
        for k_, v_ in v.items():
            dd[k_] = yamlItem2npArray(v_)
        return dd
    else:
        return v

def getSweepDict(dataDir):
    yamlDict = yaml.safe_load(open(dataDir + 'sweepAxes.yml', "r"))
    sweepAxes = {}
    for k, v in yamlDict.items():
        sweepAxes[k] = yamlItem2npArray(v)
    return sweepAxes

def getSweepDictAndMSMTInfo(dataDir):
    sweepAxes = getSweepDict(dataDir)
    msmtInfo = yaml.safe_load(open(dataDir + 'msmtInfo.yml', "r"))
    return sweepAxes, msmtInfo

if __name__ == "__main__":
    sa = {"a1": np.linspace(0, 10, 101), "b1": [12,3], "c":1, "c1":"hello", "d": {"d1":1, "d2":np.linspace(0,10,11)}}
    baseDir = r"L:\Data\SNAIL_Pump_Limitation\test\2022-09-08\\"
    preFix ="test11"
    dd = createSweepDir(baseDir, preFix, **sa)
    sa_get = getSweepDict(dd)

