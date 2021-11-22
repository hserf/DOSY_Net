# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:22:19 2021

@author: Yu & Nannan & Enping
"""
import numpy as np
import scipy.io as scio
import time
import os
import pandas as pd
import matplotlib.pyplot as plt


def __clear_env():
    for key in list(globals().keys()):
        if not key.startswith("__"):
            globals().pop(key)


def read_mat_dosy(pathname):
    """
    Output: 
        label_data: DOSY spectral data with size [N_freq, N_grad]
        b_data: vector related to the gradients with size [1, N_grad]
    
    """
    read_data = scio.loadmat(str(pathname))
    label_data = read_data['S']
    b_data = read_data['b']
    return label_data, b_data


def make_folder(BaseDir):
    Name_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    subsubFolderName = str(Name_time)
    FolderName = '%s%s/' % (BaseDir,subsubFolderName)
    if not os.path.isdir(BaseDir):
        os.mkdir(BaseDir)
    if not os.path.isdir(BaseDir):
        os.mkdir(BaseDir)
    os.mkdir(FolderName)
    os.mkdir('%smodel/' % (FolderName))
    return FolderName


def save_csv(data, FolderName, file_name, shape, is_real):
    if is_real == 0:
        y2 = np.concatenate([np.squeeze(np.real(data)),np.squeeze(np.imag(data))])
        y2 = np.reshape(y2,[2,-1])
        df = pd.DataFrame(y2)
        df.to_csv(str(FolderName) + str(file_name),index_label='0rl\\1im')
    else:
        data = np.reshape(data, shape)
        df = pd.DataFrame(data)
        df.to_csv(str(FolderName) + str(file_name), index=0,header=0)
    
    
def save_param_dosy(aout1, Akout1, alpha_num, peak_num, FolderName):
    save_csv(aout1, FolderName, file_name='diffusion_coeffs.csv', shape=[-1,alpha_num], is_real=1)
    save_csv(Akout1, FolderName, file_name='Sp.csv', shape=[-1,peak_num * alpha_num], is_real=1)
        

def draw_netoutput_DOSY(aout, Akout, step=0, save=0, FolderName=None):
    n_peaks = Akout.shape[0]
    n_dr = Akout.shape[1]
    # Akout = np.transpose(Akout)
    index = np.arange(1,n_peaks+1)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')
    
    for i in range(n_peaks):
        x = aout
        y = np.ones(Akout.shape[1])*(i+1)
        z = Akout[i,:]
        ax.scatter(x, y, z, c='red',s=20)
        #ax.plot3D(x,y,z,'red')
        
    plt.subplot(122)            

    for j in range(len(aout)):
        indexj = index[Akout[:, j] > 1e-3]
        plt.plot(indexj, [aout[j]] * len(indexj), 'ro')

    if save == 1:
        fnm = '%sPara_%s.png' % (FolderName,step)
        plt.savefig(fnm)
        plt.close()

    plt.show()


def draw_loss(i,loss_train, save=0, FolderName=None):
    plt.figure()
    plt.plot(loss_train, 'g--', label='Train loss')
    plt.legend()
    plt.title('MSE loss: step=%s' % (i))
    if save == 1:
        fnm = '%sloss_%s.png' % (FolderName,i)
        plt.savefig(fnm)
        plt.close()
    else:
        plt.show()
    

def draw_recDOSY(diff_data, Sp):
    n_peaks = Sp.shape[0]
    y = np.squeeze(diff_data)
    x = np.arange(1,n_peaks+1,1)
    plt.figure()
    plt.contour(x,y,Sp.transpose())
    plt.show()

    
