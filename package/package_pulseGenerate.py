import numpy as np
import matplotlib.pyplot as plt
import package_sequenceGenerate as sG
import time

gauCondition = {'amp': 0.1,
				'ssbFreq': 0.1,
				'iqScale': 1,
				'phase': 0,
				'skewPhase': 0,
				'sigma': 20,
				'sigmaMulti': 6,
				'dragFactor': 0}

boxCondition = {'amp': 0.5,
				'width': 200,
				'ssbFreq': 0.0,
				'iqScale': 1,
				'phase': 0,
				'skewPhase': 0,
				'rampSlope': 0.5,
				'cutFactor': 3}

module_dict = {"A1": 0,
              "M1": 0,
              "D1": 0,
              "A2": 0,
              "A3": 0,
              "M2": 0,
              "M3": 0}

ampArrayEg = np.linspace(0, 1, 11)[:10]

def piPulseTuneUp(ampArray = ampArrayEg):

	# define all waveforms we're going to use into W
	W = sG.waveformModulesCollection(module_dict)
	gPulse = sG.gau(gauCondition)
	bPulse = sG.box(boxCondition)
	for i in range(len(ampArray)):
		gPulse.amp = ampArray[i]
		W.A1[f'gau{i}.I'] = gPulse.x().I_data
		W.A1[f'gau{i}.Q'] = gPulse.x().Q_data
	W.A1['msmtBox.I'] = bPulse.smooth().I_data
	W.A1['msmtBox.Q'] = bPulse.smooth().Q_data
	W.M1['gau.M'] = gPulse.marker().I_data
	W.M1['msmtBox.M'] = bPulse.marker().I_data

	# define Queue here
	Q = sG.queueModulesCollection(module_dict)
	for i in range(len(ampArray)):
		Q.add('A1', 1, i, f'gau{i}.I', 10)
		Q.add('A1', 1, i, f'gau{i}.I', 50)
		Q.add('A1', 2, i, f'gau{i}.Q', 10)
		Q.add('A1', 2, i, f'gau{i}.Q', 50)
		Q.add('A1', 3, i, 'msmtBox.I', 110)
		Q.add('A1', 4, i, 'msmtBox.Q', 110)
		Q.add('M1', 1, i, 'gau.M', 0)
		Q.add('M1', 2, i, 'msmtBox.M', 100)
		Q.add('D1', 1, i, 'data!', 100)
	return W, Q

if __name__ == '__main__':
	start = time.time()
	W, Q = piPulseTuneUp()
	print(Q.A1.chan1)
	index = 0
	wf_index = {}
	for k in W.A1.keys():
		wf_index[k] = index
		index += 1
	print(wf_index)

	print(time.time() - start)