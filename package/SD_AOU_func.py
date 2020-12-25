import keysightSD1


def AWGconfig(self, offset: list = [0, 0, 0, 0], amplitude: list = [1.5, 1.5, 1.5, 1.5], numChan: int = 4):
    """basic configure AOU module to fitinto our purpose for AWG

    Args:
        offset (list, optional): DC offset to compensate any potential leakage
        amplitude (list, optional): full amplitude for output voltage
        numChan (int, optional): number of channel
    """
    syncMode = keysightSD1.SD_SyncModes.SYNC_CLK10
    queueMode = keysightSD1.SD_QueueMode.ONE_SHOT  # (ONE_SHOT / CYCLIC)
    self.waveformFlush()  # memory flush
    self.channelPhaseResetMultiple(0b1111)
    for i in range(1, numChan + 1):
        self.AWGflush(i)
        self.AWGqueueSyncMode(i, syncMode)
        self.AWGqueueConfig(i, queueMode)
        self.channelWaveShape(i, keysightSD1.SD_Waveshapes.AOU_AWG)
        self.channelAmplitude(i, amplitude[i - 1])
        self.channelOffset(i, offset[i - 1])


def AWGuploadWaveform(self, w_dict: dict):
    """upload all waveform into AWG module and return the index for correspondin pulse

    Args:
        w_dict (dict): {pulseName (str): pulseArray (np.ndarray)}

    Returns:
        dict: {pulseName (str): index (int)}
    """
    w_index = {}
    paddingMode = keysightSD1.SD_Wave.PADDING_ZERO
    index = 0
    for waveformName, waveformArray in w_dict.items():
        tWave = keysightSD1.SD_Wave()
        tWave.newFromArrayDouble(0, waveformArray)
        self.waveformLoad(tWave, index, paddingMode)
        w_index[waveformName] = index
        index += 1

    return w_index


def AWGqueueWaveform(self, w_index: dict, queueCollection):
    """upload all queue into module from queueCollection

    Args:
        w_index (dict): the index corresponding to each waveform. Generated from AWGuploadWaveform
        queueCollection (queueCollection): queueCollection
    """
    triggerMode = keysightSD1.SD_TriggerModes.SWHVITRIG
    for chan in range(1, 5):
        for seqOrder, seqInfo in getattr(queueCollection, f'chan{chan}').items():
            for singlePulse in seqInfo:
                triggerDelay = 0
                # nAWG, waveformNumber, triggerMode, startDelay, cycles, prescaler)
                self.AWGqueueWaveform(chan, w_index[singlePulse[0]], triggerMode, triggerDelay, 1, 0)  # singlePulse = ['pulseName', timeDelay]
    self.AWGstartMultiple(0b1111)  # TODO: not necessary
