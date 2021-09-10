from __future__ import annotations
from typing import List, Union
from inspect import getfullargspec
from dataclasses import dataclass
import os
import h5py
import numpy as np
from nptyping import NDArray
import pathlib

yamlFile = '1224Q5_info.yaml'


def getIQDataFromDataReceive(dataReceive: dict, dig_name: str, channel: int, subbuffer_used: bool):
    IQdata = IQData()
    if subbuffer_used:
        IQdata.I_raw = dataReceive[dig_name][f"ch{channel}"][:, 0::5]
        IQdata.Q_raw = dataReceive[dig_name][f"ch{channel}"][:, 1::5]
        IQdata.I_rot = dataReceive[dig_name][f"ch{channel}"][:, 2::5]  # should can be None
        IQdata.Q_rot = dataReceive[dig_name][f"ch{channel}"][:, 3::5]  # should can be None
    else:
        IQdata.I_trace_raw = dataReceive[dig_name][f"ch{channel}"][:, :, 0::5]
        IQdata.Q_trace_raw = dataReceive[dig_name][f"ch{channel}"][:, :, 1::5]
        IQdata.Mag_trace = dataReceive[dig_name][f"ch{channel}"][:, :, 2::5]
    return IQdata





def loadH5IntoIQData(directory: str = os.getcwd() + "\\", fileName: str = "temp"):
    fileOpen = h5py.File(directory + fileName, "r")
    param_dict = {}
    IQdata_ditc ={}
    for k, v in fileOpen.items():
        if k in getfullargspec(IQData.__init__).args:
            IQdata_ditc[k] = v[()]
        else:
            param_dict[k] = v[()]
    # print(IQdata_ditc)
    IQdata = IQData(**IQdata_ditc)
    IQdata.time_trace = param_dict.get("time_trace", None)
    fileOpen.close()
    return IQdata, param_dict


@dataclass
class IQData:
    I_raw: Union[List, np.ndarray] = np.array([])
    Q_raw: Union[List, np.ndarray] = np.array([])
    I_rot: Union[List, np.ndarray] = np.array([])
    Q_rot: Union[List, np.ndarray] = np.array([])
    I_trace_raw: Union[List, np.ndarray] = np.array([])
    Q_trace_raw: Union[List, np.ndarray] = np.array([])
    I_trace_rot: Union[List, np.ndarray] = np.array([])
    Q_trace_rot: Union[List, np.ndarray] = np.array([])
    Mag_trace: Union[List, np.ndarray] = np.array([])

    def integ_IQ_trace(self, integ_start: int, integ_stop: int, ref_data: IQData = None, timeUnit=10):
        demod_sigI = self.I_trace_raw
        demod_sigQ = self.Q_trace_raw
        I_raw_ = np.sum(demod_sigI[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        Q_raw_ = np.sum(demod_sigQ[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        IQ_raw_max = np.max(np.sqrt(I_raw_ ** 2 + Q_raw_ ** 2))
        truncation_factor = IQ_raw_max / 2 ** 15
        self.I_raw = I_raw_ / truncation_factor
        self.Q_raw = Q_raw_ / truncation_factor

        if type(ref_data) is not IQData:
            return

        demod_refI = ref_data.I_trace_raw
        demod_refQ = ref_data.Q_trace_raw
        ref_data.I_raw = np.sum(demod_refI[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        ref_data.Q_raw = np.sum(demod_refQ[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        ref_mag = np.sqrt(ref_data.I_raw ** 2 + ref_data.Q_raw ** 2)
        ref_I_raw_3 = ref_data.I_raw[:, :, np.newaxis]
        ref_Q_raw_3 = ref_data.Q_raw[:, :, np.newaxis]
        ref_mag_3 = ref_mag[:, :, np.newaxis]
        self.I_trace_rot = (demod_sigI * ref_I_raw_3 + demod_sigQ * ref_Q_raw_3) / ref_mag_3
        self.Q_trace_rot = (-demod_sigI * ref_Q_raw_3 + demod_sigQ * ref_I_raw_3) / ref_mag_3
        self.I_rot = np.sum(self.I_trace_rot[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        self.Q_rot = np.sum(self.Q_trace_rot[:, :, integ_start // timeUnit:integ_stop // timeUnit], axis=2)
        self.I_rot /= truncation_factor
        self.Q_rot /= truncation_factor

    def saveIQDataIntoH5(self, directory: str = os.getcwd() + "\\", fileName: str = "temp", **kwargs):
        try:
            path = pathlib.Path(directory)
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(error)
        duplicateIndex = 0
        saveSuccess = 0
        saveName = fileName

        while not saveSuccess:
            try:
                fileSave = h5py.File(directory + saveName, 'x')
                saveSuccess = 1
            except OSError:
                saveName = fileName + "_" + str(duplicateIndex)
                duplicateIndex += 1

        for k, v in self.__dict__.items():
            if k[0] == "_":
                k = k[1:]

            if np.sum(v) == None:
                pass
            else:
                fileSave.create_dataset(k, data=v)

        for k, v in kwargs.items():
            fileSave.create_dataset(k, data=v)
        fileSave.close()
        print(directory + saveName + " file saved successfully")
        return
    """
    @property
    def I_raw(self) -> NDArray[float]:
        return self._I_raw

    @I_raw.setter
    def I_raw(self, data: NDArray[float]) -> None:  # TODO: warning when asign to NonNone value
        self._I_raw = data

    @property
    def Q_raw(self) -> NDArray[float]:
        return self._Q_raw

    @Q_raw.setter
    def Q_raw(self, data: NDArray[float]) -> None:
        self._Q_raw = data

    @property
    def I_rot(self) -> NDArray[float]:
        return self._I_rot

    @I_rot.setter
    def I_rot(self, data: NDArray[float]) -> None:
        self._I_rot = data

    @property
    def Q_rot(self) -> NDArray[float]:
        return self._Q_rot

    @Q_rot.setter
    def Q_rot(self, data: NDArray[float]) -> None:
        self._Q_rot = data

    @property
    def I_trace_rot(self) -> NDArray[float]:
        return self._I_trace_rot

    @I_trace_rot.setter
    def I_trace_rot(self, data: NDArray[float]) -> None:
        self._I_trace_rot = data

    @property
    def Q_trace_rot(self) -> NDArray[float]:
        return self._Q_trace_rot

    @Q_trace_rot.setter
    def Q_trace_rot(self, data: NDArray[float]) -> None:
        self._Q_trace_rot = data
    """

if __name__ == "__main__":
    a = np.array([1, 2, 3])
    a3 = [[a, a * 2]] * 5
    a3 = np.array(a3)
