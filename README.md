# Code from "**A Neural Network Method for Diffusion-Ordered NMR Spectroscopy**"

This repository contains information and code from the paper "A Neural Network Method for Diffusion-Ordered NMR Spectroscopy", which is published at *Analytical Chemistry* **2022**, 94(6), 2699-2705, by Enping Lin, Nannan Zou, Yuqing Huang, Zhong Chen, and Yu Yang. 

## Requirements

Here is a list of libraries you might need to install to execute the code:

- python (=3.6)
- tensorflow (=1.14.0)
- numpy
- scipy
- h5py
- pandas
- matplotlib
- jupyter notebook

## Data

A .mat file for DOSY data should be prepared before running the python code. Examples for the .mat file are presented in the 'data' folder. To generate this .mat file, one could use one of the following two methods:<br><br>
(1) Use DOSYToolbox to export the .mat file, which contains a structure named "NmrData". Make sure that these variables exist in NmrData: NmrData.SPECTRA (to-be-process data matrix, the same as "S" in the paper), NmrData.Gzlvl (pulse field gradients, as "g" in the paper), NmrData.ngrad (the number of gradients), NmrData.dosyconstant (gamma.^2\*delts^2\*DELTAprime).<br>
<br>
(2) Define a .mat file including "S" (to-be-process data matrix), "b" (the gradient-related vector), "idx_peaks" (the indices of the selected spectral points), and "ppm" (the chemical shift coordinates in ppm of the original data). It should be mentioned that only "S" and "b" are used for the DOSY parameter estimation while "idx_peaks" and "ppm" are used for the subsequent spectrum reconstruction and display.<br><br>

- The subfolder 'simulation' contains a simple example of simulation data and the generation code written in Matlab.<br>
- The subfolder 'QGC' contains the data that are applied in the paper as the first example. The corresponding .mat file (QGC_net_input.mat) contains data that are transformed and extracted from the original DOSY experimental data. If you want to use this data, please refer to (and cite) the original paper: Foroozandeh, M.; Castanar, L.; Martins, L. G.; Sinnaeve, D.; Poggetto, G. D.; Tormena, C. F.; Adams, R. W.; Morris, G. A.; Nilsson, M. Ultrahigh-Resolution Diffusion-Ordered Spectroscopy, *Angewandte Chemie* **2016**, *128*, 15808-15811.<br>
- The subfolder 'GSP' contains the data that are applied in the paper as the second example. The corresponding .mat file (GSP_net_input.mat) contains data that are transformed and extracted from the original DOSY experimental data. If you want to use this data, please refer to (and cite) the original paper: Yuan, B.; Ding, Y.; Kamal, G. M.; Shao, L.; Zhou, Z.; Jiang, B.; Sun, P.; Zhang, X.; Liu, M. Reconstructing diffusion ordered NMR spectroscopy by simultaneous inversion of Laplace transform, *Journal of Magnetic Resonance* **2017**, *278*, 1-7.<br>

## Python Code (to generate parameters for DOSY spectrum)

- utils.py provides the script for loading the data, displaying and saving the results.
- train_DOSYEst.py is the code for setting and training the neural network model to generate the parameters for the fitting. 
- examples.ipynb is the example code written in Jupyter Notebook to present how to use this neural-network-based optimizer and generate the parameters for DOSY spectra.

After running the code (demo.py or examples.ipynb) with defaulted setting, a folder named "Net_Results" would be created and expanded with new results, including the estimated diffusion coefficients D(l) and the spectral coefficients C(l, f). To show the results, the following Matlab codes are provided:


## Matlab Code (to show the DOSY spectrum in a prettier way)

The Matlab code for spectrum reconstruction and display is included in the folder "Matlab_codes_for_spectra_display". 

## Others

Email me if you have any questions: yuyang15@xmu.edu.cn
