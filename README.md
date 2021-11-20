# Code from "**A Neural Network Method for Diffusion-Ordered NMR Spectroscopy**"

This repository contains information and code from the paper "A Neural Network Method for Diffusion-Ordered NMR Spectroscopy", *which is unpublished yet*, by Enping Lin, Nannan Zou, Yuqing Huang, Yu Yang, and Zhong Chen. 

## Requirements

Here is a list of libraries you might need to install to execute the code:

- python (=3.6)
- tensorflow (=1.14.0)
- numpy
- scipy
- pandas
- matplotlib
- jupyter notebook

## Data

A .mat file for DOSY data should be prepared before running the python code. Examples for the .mat file are presented in the 'data' folder.

- The subfolder 'simulation' contains a simple example of simulation data and the generation code written in Matlab.
- The subfolder 'QGC' contains the data that are applied in the paper as the first example. The corresponding .mat file (QGC_net_input.mat) contains data that are transformed and extracted from the original DOSY experimental data. If you want to use this data, please refer to (and cite) the original paper: Foroozandeh, M.; Castanar, L.; Martins, L. G.; Sinnaeve, D.; Poggetto, G. D.; Tormena, C. F.; Adams, R. W.; Morris, G. A.; Nilsson, M. Ultrahigh-Resolution Diffusion-Ordered Spectroscopy, *Angewandte Chemie* **2016**, *128*, 15808-15811.
- The subfolder 'GSP' contains the data that are applied in the paper as the second example. The corresponding .mat file (GSP_net_input.mat) contains data that are transformed and extracted from the original DOSY experimental data. If you want to use this data, please refer to (and cite) the original paper: Yuan, B.; Ding, Y.; Kamal, G. M.; Shao, L.; Zhou, Z.; Jiang, B.; Sun, P.; Zhang, X.; Liu, M. Reconstructing diffusion ordered NMR spectroscopy by simultaneous inversion of Laplace transform, *Journal of Magnetic Resonance* **2017**, *278*, 1-7.

## Python Code (to generate parameters for DOSY spectrum)

- utils.py provides the script for loading the data, displaying and saving the results.
- train_DOSYEst.py is the code for setting and training the neural network model to generate the parameters for the fitting. 
- examples.ipynb is the example code written in Jupyter Notebook to present how to use this neural-network-based optimizer and generate the parameters for DOSY spectra.

After running the code (train_DOSYEst.py or examples.ipynb) with defaulted setting, a folder named "Net_Results" would be created and expanded with new results, including the estimated diffusion coefficients D(l) and the spectral coefficients C(l, f).


## Matlab Code (to show the DOSY spectrum in a prettier way)

The Matlab code for spectrum display is included in the folder "Matlab_codes_for_spectra_display". 

## Others

Email me if you have any questions: yuyang15@xmu.edu.cn
