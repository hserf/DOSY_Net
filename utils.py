# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:22:19 2020

@author: Yu
"""
import numpy as np
import scipy.io as scio
import h5py
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def __clear_env():
    for key in list(globals().keys()):
        if not key.startswith("__"):
            globals().pop(key)


def read_mat_dosy(args):
    """
    Input:
        pathname: the path of the .mat file 
        thr: threshold value (only effective when the input is an DOSYToolbox exported .mat file)
    Output: 
        label_data: DOSY spectral data with size [N_freq, N_grad]
        b_data: vector related to the gradients with size [1, N_grad]
        ppm: the chemical shift coordinates (in ppm) of the original data
        idx_peaks: the indices of the selected spectral points
    """
    pathname = args.input_file
    thr = args.threshold
    
    nmrdata_save = dict()
    try:
        read_data = scio.loadmat(str(pathname))
        if read_data.__contains__('S') & read_data.__contains__('b'):
            print('Read from a user-defined Mat file.')
            nmrdata_save['S'] = read_data['S']
            nmrdata_save['b'] = read_data['b'].astype(float)
            if read_data.__contains__('ppm'):
                nmrdata_save['ppm'] = np.squeeze(read_data['ppm'])
            if read_data.__contains__('idx_peaks'):
                nmrdata_save['idx_peaks'] = read_data['idx_peaks']
            
        elif read_data.__contains__('NmrData'):
            print('Read from a DOSY-Toolbox exported file using scio.')
            nmrdata = read_data['NmrData']
            # b-vector
            expfactor = np.squeeze(nmrdata['dosyconstant'][0,0])*1e-10
            ngrad = int(nmrdata['ngrad'][0,0])
            g = nmrdata['Gzlvl'][0,0]
            if g.shape[0]>g.shape[1]:
                g = np.transpose(g)
            nmrdata_save['b'] = expfactor*g**2
            
            # Spectrum data
            if args.data_matrix_proc == 'real':
                specdata = np.real(nmrdata['SPECTRA'][0,0]) 
            elif args.data_matrix_proc == 'abs':
                specdata = np.abs(nmrdata['SPECTRA'][0,0])
            else:
                specdata = np.abs(nmrdata['SPECTRA'][0,0])
                warnings.warn('Unrecognized input setting: data_matrix_proc. Apply ''abs'' instead.')
                
            if specdata.shape[1] != ngrad:
                specdata = np.transpose(specdata)
            specdata = specdata/specdata.max()
            
            thr = max(thr, 0.0)
            spec0 = specdata[:,0]
            idx_peaks = np.asarray(np.where(spec0>thr))
            nmrdata_save['idx_peaks'] = idx_peaks
            nmrdata_save['S'] = specdata[np.squeeze(idx_peaks),:]
            nmrdata_save['ppm'] = np.squeeze(nmrdata['Specscale'][0,0])
            
        else:
            print('Reading File Error: make sure there are ''S'', ''b'' or ''NmrData'' in your .mat file')
               
    except NotImplementedError:
        read_data = h5py.File(str(pathname),'r')
        nmrdata = read_data.get('NmrData')
        print('Read from a DOSY-Toolbox exported file using h5py.')
        # b-vector
        expfactor = nmrdata.get('dosyconstant')[0,0]*1e-10
        ngrad = int(nmrdata.get('ngrad')[0,0])
        g = nmrdata.get('Gzlvl')[()]
        if g.shape[0]>g.shape[1]:
            g = np.transpose(g)
        nmrdata_save['b'] = expfactor*g**2
        
        # Spectrum data
        if args.data_matrix_proc == 'real':
            specdata = nmrdata['SPECTRA']['real']
        else:
            specdata_r = nmrdata['SPECTRA']['real']
            specdata_i = nmrdata['SPECTRA']['imag']
            specdata = np.sqrt(specdata_r**2 + specdata_i**2)
            if args.data_matrix_proc != 'abs':
                warnings.warn('Unrecognized input setting: data_matrix_proc. Apply ''abs'' instead.')
            
        if specdata.shape[1] != ngrad:
            specdata = np.transpose(specdata)
        specdata = specdata/specdata.max()
        
        thr = max(thr, 0.0)
        spec0 = specdata[:,0]
        idx_peaks = np.asarray(np.where(spec0>thr))
        nmrdata_save['idx_peaks'] = idx_peaks
        nmrdata_save['S'] = specdata[np.squeeze(idx_peaks),:]
        nmrdata_save['ppm'] = np.squeeze(nmrdata['Specscale'])
        
            
    return nmrdata_save


def make_folder(BaseDir):
    Name_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    subsubFolderName = str(Name_time)
    FolderName = '%s%s/' % (BaseDir,subsubFolderName)
    if not os.path.isdir(BaseDir):
        os.makedirs(FolderName)
    else:
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
    
    
def save_param_dosy(aout1, Akout1, nmrdata, alpha_num, peak_num, FolderName):
    save_csv(aout1, FolderName, file_name='diffusion_coeffs.csv', shape=[-1,alpha_num], is_real=1)
    save_csv(Akout1, FolderName, file_name='Sp.csv', shape=[-1,peak_num * alpha_num], is_real=1)
    matfile = '%sdata_org.mat'%FolderName
    scio.savemat(file_name=matfile, mdict=nmrdata)
        

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
    plt.yscale('log')
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

    