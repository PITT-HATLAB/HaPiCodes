import time
from pathwave import HVI_config as pwh
import package_allMsmtPulses as amp
from AddOns.DigFPGAConfig_QubitMSMT import *

triggerAwgDigDelay = 330 #info['sysConstants']['triggerAwgDigDelay']

pulse_general_dict = {'relaxingTime': 600, # float; precision: 0.01us; relax after the start of the firt pulse
                      'avgNum': 300}


timeArray = np.linspace(0, 300000, 101)[:100]
if __name__ == '__main__':

    # pointPerCycle = 50000
    # cycles = 3
    start = time.time()
    FPGA_file = r'N:\Chao\HaPiCodes\PathwaveFPGATests\FPGA\Qubit_MSMT\Qubit_MSMT.k7z'


    module_dict = pwh.openAndConfigAllModules(FPGA_file) # TODO, I think this will look better if we write this as a class, "PXIModules"
    W, Q = amp.t1MsmtReal(module_dict)
    avg_num_per_hvi = pwh.autoConfigAllDAQ(module_dict, Q, pulse_general_dict['avgNum']) #PXIModules.autoConfigAllDAQ


    pulse_general_dict["avgNum"] = avg_num_per_hvi
    # TODO: this need to be changed. W, Q is no longer necessary to be returned here,
    hvi = pwh.uploadPulseAndQueue(W, Q, module_dict, pulse_general_dict, subbuffer_used=1) # PXIModules.defineAndUploadPulse



    configFPGA(module_dict["D1"].instrument, 1, 17, 100, 300, 5, 2, False) ## PXIModules.configDigitizer
    configFPGA(module_dict["D1"].instrument, 2, 18, 100, 300, 7, 2, False)

    dataReceive = pwh.runExperiment(module_dict, hvi, Q, timeout=20000)  # PXIModules.runExperiment

    Idata = np.average(dataReceive["D1"]['ch1'], axis=0)[2::5]
    Qdata = np.average(dataReceive["D1"]['ch1'], axis=0)[3::5]


    import fit_all as fa
    fa.t1_fit(Idata, Qdata, timeArray/1e3)

    pwh.releaseHviAndCloseModule(hvi, module_dict)  # PXIModules.releaseHviAndCloseModule