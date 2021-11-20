# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:10:19 2020

@author: Yu
"""

import time
import tensorflow as tf
import numpy as np
#import random
#from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import argparse

from model_DOSY_est import param_gen_dosy_simple, param_gen_dosy_mostsimple
import utils


def options():
    # Set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate_1", default=1e-4, type=float, help='Learning rate for the first 15000 steps')
    parser.add_argument("--learning_rate_2", default=5e-6, type=float, help='Learning rate after 15000 steps')
    parser.add_argument("--input_file", default='data/simulation/testdataSigma0.015.mat', help='File path of the input data')
    parser.add_argument("--output_path", default='Net_Results/', help='Output file path')
    parser.add_argument("--diff_range", default=[], type=float, help='The range of the diffusion coefficient')
    parser.add_argument("--n_decay", default=3, type=int, help='The number of different decay components')
    parser.add_argument("--dim_in", default=100, type=int, help='Dimension of the input of the network')
    parser.add_argument("--reg_A", default=0.01, type=float, help='Regularization parameter')
    parser.add_argument("--max_iter", default=20000, type=int, help='Maximum iterations')
    parser.add_argument("--fidelity", default='norm-2', help='Fidelity term setting, could be norm-1 or norm-2')
    parser.add_argument("--save_process", default=True, help='If you want to save the intermediate process parameters?')
    parser.add_argument("--save_interval", default=500, type=int, help='The step size to save the intermediate results')
    parser.add_argument("--display", default=True, help='Do you want to show the convergence process in the terminal?')
       
    args = parser.parse_args()            
    return args


def train(args):
    # 1. load data & generate the input
    label_data0, b = utils.read_mat_dosy(args.input_file)
    n_grad = b.shape[1]
    n_freq = label_data0.shape[0]
    input_r = np.random.randn(1,args.dim_in)
    input_r = np.expand_dims(input_r,axis=2)

    # 2. Build graph
    tf.compat.v1.reset_default_graph()
    net1_input = tf.compat.v1.placeholder(tf.float32, shape=[1, args.dim_in, 1], name="net_input")
    label1_output = tf.compat.v1.placeholder(tf.float32, shape=[n_freq, n_grad], name="output_label")
    b_in = tf.compat.v1.placeholder(tf.float32, shape=[1, n_grad], name='b_in')
    
    # load the neural network model
    dr1_out, sp1_out, X_output, normC_out = param_gen_dosy_simple(net1_input, args.n_decay, n_freq, b_in, args.diff_range)  # 
    #dr1_out, sp1_out, X_output, normC_out = param_gen_dosy_mostsimple(net1_input, args.n_decay, n_freq, b_in, args.diff_range)  #
    
    norm_label_tensor = tf.math.reduce_max(label1_output,axis=1,keepdims=True)
    
    with tf.name_scope('loss'):
        err_tmp = label1_output - X_output
        if args.fidelity == 'norm-2':
            fidelity_loss = tf.square(tf.norm(err_tmp, ord='euclidean'))
        elif args.fidelity == 'norm-1':
            fidelity_loss = tf.reduce_sum(tf.math.abs(err_tmp))
        else:
            print('Undefined fidelity term. Please input: norm-1 or norm-2 for "fidelity"')
            return
            
        sp1_weighted = (sp1_out*norm_label_tensor)/tf.transpose(normC_out)
        LW1_out = tf.reduce_sum(tf.reshape(sp1_weighted,[-1]))
        train_loss = fidelity_loss + args.reg_A * LW1_out
         
    learning_rate = tf.compat.v1.placeholder(tf.float32, [])
    train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(train_loss)
    
    # 3. Prepare to save the results
    Folder_name = utils.make_folder(args.output_path)
    tf_log_dir = str(Folder_name)+'model'
    
    ### observed variabel #####
    if args.save_process == True:
        loss_train = []
        dr_save = []
        Sp_save = []
    
    # summary_writer = tf.compat.v1.summary.FileWriter(tf_log_dir, graph=sess.graph)
    
    # 4. Training: optimizing the output parameters
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(max_to_keep=5)
        saver.save(sess, tf_log_dir+'/model.ckpt-done')
        
        #checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # t0 = time.perf_counter()
        t0 = time.clock()

        for i in range(args.max_iter):
            if i<1.5e4:
                lr = args.learning_rate_1
            else:
                lr = args.learning_rate_2
        
            feed_dict = {
                    net1_input:input_r,
                    label1_output:label_data0,
                    b_in:b,
                    learning_rate:lr,
                    }
            gen_loss_tmp = sess.run(train_loss,feed_dict=feed_dict)
            loss_train.append(gen_loss_tmp)
                    
            if args.display or args.save_process:
                if (i % args.save_interval) == 0:
                    dr, Sp, fl = sess.run([dr1_out,sp1_out,fidelity_loss], feed_dict=feed_dict)
                    if args.save_process:
                        dr_save = np.concatenate([dr_save,np.reshape(dr,[-1])])
                        Sp_save = np.concatenate([Sp_save,np.reshape(Sp,[-1])])
                    if args.display:
                        print('step: %d, total loss: %f, fidelity loss: %f' %(i + 1, gen_loss_tmp, fl))    
                        t1 = time.clock()
                        print('time cost: %s' % (t1 - t0))
                        #utils.draw_netoutput_DOSY(dr, Sp)
        
            _ = sess.run(train_opt, feed_dict=feed_dict)
               
        
        print("Loop over")
        traintime = time.clock() - t0
        print('Total time cost: %s' % traintime)
        
        dr, Sp, gen_loss_tmp, fl = sess.run([dr1_out,sp1_out,train_loss,fidelity_loss], feed_dict=feed_dict)
        print('Final step: %d, total loss: %f, fidelity loss: %f' %(i + 1, gen_loss_tmp, fl))    
        dr_save = np.concatenate([dr_save,np.reshape(dr,[-1])])
        Sp_save = np.concatenate([Sp_save,np.reshape(Sp,[-1])])

        utils.save_param_dosy(dr_save, Sp_save, args.n_decay, n_freq, Folder_name)
        
        if args.display:
            utils.draw_loss(i, loss_train)
            dr_save = np.reshape(dr_save,[-1,args.n_decay])
            plt.figure()
            plt.plot(dr_save)
            plt.show()
        
        return dr, Sp


if __name__ == '__main__':
    args = options()
    print(args)
   
    dr, Sp = train(args)


