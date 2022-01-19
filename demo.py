# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:05:35 2021

@author: Yu
"""

from train_DOSYEst import train, options

args = options()

idx_case = 1

if idx_case == 1:
    ## Case 1: simulation
    #args.input_file = 'data/simulation/testdataSigma0.015.mat'
    args.input_file = 'data/simulation/testdataSigma0.1.mat'
    args.output_path = 'Net_Results/sim/'
    args.diff_range = []
    args.reg_A = 0.3
    args.max_iter = 20000
    args.learning_rate_1 = 1e-4
    args.learning_rate_2 = 5e-6
    args.fidelity = 'norm-2'
    args.n_decay = 3
    
elif idx_case == 2:
    ## Case 2: QGC
    args.input_file = 'data/QGC/QGC_net_input.mat'
    #args.input_file = 'data/QGC/QGC.mat'
    args.threshold = 0.0138
    args.output_path = 'Net_Results/QGC/'
    args.diff_range = []
    #args.diff_range = [3.0, 12.0]
    args.reg_A = 0.1
    args.max_iter = 30000
    args.learning_rate_1 = 2e-4 # 1e-4 is also fine 
    args.learning_rate_2 = 1e-5 # 5e-6 is also fine but may need a larger "max_iter"
    args.fidelity = 'norm-2'
    args.n_decay = 3
    
elif idx_case == 3:
    ## Case 3: GSP
    args.input_file = 'data/GSP/GSP_net_input.mat'
    #args.input_file = 'data/GSP/GSP.mat'
    args.threshold = 0.03
    args.output_path = 'Net_Results/GSP/'
    args.diff_range = []
    #args.diff_range = [1.0, 6.0]
    args.reg_A = 0.8
    args.max_iter = 30000
    args.learning_rate_1 = 2e-4 # 1e-4 is also fine 
    args.learning_rate_2 = 1e-5 # 5e-6 is also fine but may need a larger "max_iter"    
    args.fidelity = 'norm-2'
    args.n_decay = 3


dr, Sp = train(args)