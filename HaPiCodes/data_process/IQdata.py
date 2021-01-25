from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from nptyping import NDArray

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


@dataclass
class IQData:
    _I_raw = None
    _Q_raw = None
    _I_rot = None
    _Q_rot = None
    I_trace_raw = None
    Q_trace_raw = None
    _I_trace_rot = None
    _Q_trace_rot = None
    Mag_trace = None

    def integ_IQ_trace(self, integ_start: int, integ_stop: int, ref_data: IQData = None):
        demod_sigI = self.I_trace_raw
        demod_sigQ = self.Q_trace_raw
        I_raw_ = np.sum(demod_sigI[:, :, integ_start // 10:integ_stop // 10], axis=2)
        Q_raw_ = np.sum(demod_sigQ[:, :, integ_start // 10:integ_stop // 10], axis=2)
        IQ_raw_max = np.max(np.sqrt(I_raw_ ** 2 + Q_raw_ ** 2))
        truncation_factor = IQ_raw_max/2**15
        self.I_raw = I_raw_/truncation_factor
        self.Q_raw = Q_raw_/truncation_factor

        if type(ref_data) is not IQData:
            return

        demod_refI = ref_data.I_trace_raw
        demod_refQ = ref_data.Q_trace_raw
        ref_data.I_raw = np.sum(demod_refI[:, :, integ_start // 10:integ_stop // 10], axis=2)
        ref_data.Q_raw = np.sum(demod_refQ[:, :, integ_start // 10:integ_stop // 10], axis=2)
        ref_mag = np.sqrt(ref_data.I_raw ** 2 + ref_data.Q_raw ** 2)
        ref_I_raw_3 = ref_data.I_raw[:, :, np.newaxis]
        ref_Q_raw_3 = ref_data.Q_raw[:, :, np.newaxis]
        ref_mag_3 = ref_mag[:, :, np.newaxis]
        self.I_trace_rot = (demod_sigI * ref_I_raw_3 + demod_sigQ * ref_Q_raw_3) / ref_mag_3
        self.Q_trace_rot = (-demod_sigI * ref_Q_raw_3 + demod_sigQ * ref_I_raw_3) / ref_mag_3
        self.I_rot = np.sum(self.I_trace_rot[:, :, integ_start // 10:integ_stop // 10], axis=2)
        self.Q_rot = np.sum(self.Q_trace_rot[:, :, integ_start // 10:integ_stop // 10], axis=2)
        self.I_rot /= truncation_factor
        self.Q_rot /= truncation_factor

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


a = np.array([1, 2, 3])
a3 = [[a, a * 2]] * 5
a3 = np.array(a3)
b = np.average(a3, axis=2)
b1 = b[:, :, np.newaxis]
