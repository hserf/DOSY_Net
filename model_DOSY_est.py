# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:50:59 2021

@author: Yu & Nannan & Enping

"""

import tensorflow as tf
import numpy as np


def param_gen_dosy_simple(input_r, n_alphas, n_peaks, t_in, diff_range=[]):
    with tf.compat.v1.variable_scope('param_gen_dosy_experiment'):
        conv = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=2, activation=tf.nn.leaky_relu)(input_r)
        conv_flat = tf.reshape(conv, [1, -1])
        par_a = tf.keras.layers.Dense(units=n_alphas)(conv_flat)
        par_A = tf.keras.layers.Dense(units=n_alphas*n_peaks)(conv_flat)
        
        Ak = tf.reshape(tf.cos(par_A)+1.0, [n_peaks, n_alphas])
        
        if not diff_range:
            z = tf.reshape(tf.nn.sigmoid(par_a), [n_alphas, 1])
            a = -tf.math.log(z)
            output, C = harmonic_gen_dosy_z(z, Ak, t_in)
        else:
            z_min = np.exp(-diff_range[1])
            z_max = np.exp(-diff_range[0])
            tmp = tf.nn.sigmoid(par_a)*(z_max-z_min)+z_min
            z = tf.reshape(tmp, [n_alphas, 1])
            a = -tf.math.log(z)
            output, C = harmonic_gen_dosy_z(z, Ak, t_in)
        
        norm_C = tf.square(tf.norm(C, axis=1, keepdims=True))        
        
    return a, Ak, output, norm_C


def harmonic_gen_dosy_z(z, Ak, t):
    # size of z: [n_decay, 1]
    # size of Ak: [n_freq, n_decay]
    # size of t: [1, n_grad]
    # size of output: [n_freq, n_grad]
    C = tf.math.pow(z, t)        
    output = tf.matmul(Ak, C)
    return output, C

