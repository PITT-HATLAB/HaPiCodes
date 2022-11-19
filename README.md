# HaPiCodes
Hatlab PXIe AWG and digitizer control package for using the Keysight M3 series 
boards for qubit experiments. 

This package works with the Keysight KS2201A PathWave Test Sync Executiv API. For 
the old M3602A HVI, check out https://github.com/hatlabcz/HatLab_PXI.

This package contains:
* FPGA firmware for the M3102A digitizer for realtime qubit readout signal 
  demodualtation and state discrimation.
* Python pacakge for pulse programing and sequencing for the M3201A/M3202A AWGs 
  based on the KS2201A HVI.
* Data processing and analysis package for qubit characteraization, gate 
  calibration, and etc.
  
## Get started
Download and install the package.

Check out "qubit_msmt_demos" for exmaples of how to write qubit measurement 
experiments with this pacakge.

For any questions, please contact Pinlei (pil9@pitt.edu) or Chao (chz78@pitt.edu).

