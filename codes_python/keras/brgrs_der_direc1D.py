#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import os
import csv
import sys 
import time
import os.path as osp

import numpy as np
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, warnings, argparse

## Import de la classe CVC ##
cases_folder = osp.abspath(osp.dirname("../cases/"))
sys.path.append(cases_folder)

import solvers
import tensorflow as tf

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, MaxPool1D, AveragePooling1D, UpSampling1D, Activation, Conv1D, concatenate
from keras.layers.normalization import BatchNormalization

import keras.optimizers as optimizers

import Roe_inference_burgers as rib
rib = reload(rib)
solvers = reload(solvers)

from visu_strides import glide1D

import time

try :
    tf.reset_default_graph()
    print ("Graph reseted")
except :
    pass

def build_dataset(roe) :
    X = np.zeros((roe.Nx - 2)) # roe.Nx
    y = np.zeros_like(range(roe.Nx - 2)) # roe.Nx
    
    final_X = np.zeros((roe.Nx - 2)) # roe.Nx
    final_y = np.zeros((roe.Nx - 2)) # roe.Nx
    final_d = np.zeros((roe.Nx - 2)) # roe.Nx
    
    deru = np.zeros((roe.Nx - 2))
    
    print "X.shape init : {}".format(X.shape)
    
    header_file = lambda amp, U_adv, init, harm, phase, it, real : \
        "../cases/data/roe/Nt100_Nx250_CFL0_40_amp%s_U_adv%s_%s_harm%s_phase%s_u_it%d_%d.npy"% \
        (amp, U_adv, init, harm, phase, it, real)
    
    init = 'sin'
    
    cnt = 0
    dx  = roe.dx
    
    load_file = lambda abspath_ffile : np.load(abspath_ffile)
    calc_deri = lambda field : np.array([(field[j+1] - field[j-1]) / (2. * dx) \
                                         for j in range(1, roe.Nx - 1)]).ravel()
    
    times = []
    
    CFL = '0_40'
    for amp in ['0_50', '1_00', '1_50']:
        for u in ['0_30', '0_60', '0_90', '1_20', '1_50'] :
            for h in ['1_00' , '2_00', '3_00'] :
                for ph in ['0_00', '0_17', '0_33', '0_50'] :
                    for it in range(180) :
                        
                        # For X                    
                        ffile = header_file(amp, u, init, h, ph, it, 0)
                        curr_u = load_file(ffile)
                        curr_der = calc_deri(curr_u)
#                        print "curr_u :\n{}".format(curr_u)
#                        print "curr_der :\n{}".format(curr_der)
                        
#                        print np.shape(np.array([curr_u[1:-1], curr_der]))
#                        print np.array([curr_u[1:-1], curr_der])
                        
                        # For y
                        ffile = header_file(amp, u, init, h, ph, it+1, 0)
                        next_u = load_file(ffile)
                        
                        # Effective build of X and y 
#                        print curr_u[1:-1]
#                        print X
                        
                        time1 = time.time()
                        X = np.block([ [X], [curr_u[1:-1]] ]) #, curr_der] ])
                        y = np.block([ [y], [next_u[1:-1]] ])
                        deru = np.block([ [deru], [curr_der] ])
                        
                        times.append(abs(time.time() - time1))
                        
                        if cnt % 500 == 0 :
                            print cnt
                        cnt += 1
                        
                    final_X = np.block([ [final_X], [X] ])
                    final_y = np.block([ [final_y], [y] ])
                    final_d = np.block([ [final_d], [deru] ])

                    X = np.zeros((roe.Nx - 2))
                    y = np.zeros((roe.Nx - 2))
                    deru = np.zeros((roe.Nx - 2))
                
    print cnt
    
    final_X = np.delete(final_X, 0, axis=0)
    final_y = np.delete(final_y, 0, axis=0)
    final_d = np.delete(final_d, 0, axis=0)
    
    final_X = np.delete(final_X, 0, axis=0)
    final_y = np.delete(final_y, 0, axis=0)
    final_d = np.delete(final_d, 0, axis=0)
    
    return final_X, final_d, final_y 
    
#    for j in range(1, len(self.line_x)-1)
    
#    return X, y, (cnt, X_keys, y_keys, u_means)

def split_data(X, der, y):
    permutation = np.random.permutation(len(y))
    
    X, y = np.copy(X), np.copy(y)
    
    X_rand = X[permutation]
    y_rand = y[permutation]
    der_rand = der[permutation]
    
    xtr = X_rand[:int(len(y)*0.8)] 
    ytr = y_rand[:int(len(y)*0.8)]
    dertr = der_rand[:int(len(y)*0.8)]    
    
    xte = X_rand[int(len(y)*0.8):int(len(y)*0.9)]
    yte = y_rand[int(len(y)*0.8):int(len(y)*0.9)]
    derte = der_rand[int(len(y)*0.8):int(len(y)*0.9)]
        
    xval = X_rand[int(len(y)*0.9):]
    yval = y_rand[int(len(y)*0.9):]
    derval = der_rand[int(len(y)*0.9):]
    
    return xtr, ytr, dertr, xte, yte, derte, xval, yval, derval

def see_pred(roe, unet, u_means, key, indice) :
    u_n = xval[indice]
    beta_pred = unet.predict(u_n.reshape(1, roe.Nx, 1)).ravel()
    
    u_n = u_n.ravel()
    true_beta = yval[indice].ravel()

#    u_beta(self, beta, u_n, verbose=False)
    u_npred = cb.u_beta(beta_pred, u_n)
    
#    u_nexact = 
    

if __name__ == '__main__':
#    run brgrs_der_direc1D.py -num_real 1 -itmax 180 -CFL 0.4 -Nx 250 -Nt 100 -typeJ "u" -dp "./../cases/data/2019_burger_data/"
    parser = rib.parser()
    roe = rib.Class_Roe_BFGS(parser, call=True)

    #    unet = model(roe.Nx, 2)
    #    X, y, utils = build_dataset(roe)

    print (" ")    
    
    der_u_input_shape = (roe.Nx-2, 1)  
    u_input_shape = (roe.Nx-2, 1)

    u_input_data = Input(shape=u_input_shape, name='u-input')
    
    der_u_input_data = Input(shape=der_u_input_shape, name='deru-input')

    x = Conv1D(3, kernel_size=3, padding='same')(u_input_data)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv1D(3, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    
    print x.shape

    deru = Conv1D(3, kernel_size=3, padding='same')(der_u_input_data)
    deru = BatchNormalization()(deru)
    deru = Activation('elu')(deru)

    deru = Conv1D(3, kernel_size=3, padding='same')(deru)
    deru = BatchNormalization()(deru)
    deru = Activation('elu')(deru)
    
    print deru.shape
    
    xx = concatenate([x, deru])

    to_concatene_first = xx

    in_pool_shape = np.shape(xx)
     
    x = AveragePooling1D(pool_size=2, padding='same',name='Max-convo')(xx)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")

    #	Out_shape = (?, 124, 3)

        
    in_pool_shape = np.shape(x)

    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    to_concatene_second = x

    in_pool_shape = np.shape(x)
     
    x = AveragePooling1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")

    in_pool_shape = np.shape(x)

    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")

    # Out_shape = (?, 124, 12)

    y = concatenate([to_concatene_second, x])

    in_pool_shape = np.shape(y)

    x = Conv1D(6, kernel_size=2, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(6,2) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    print ("in {}--> Conv1D(6,2) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")

    y = concatenate([to_concatene_first, x])

    x = Conv1D(3, kernel_size=1, padding='same')(y)
    out = Conv1D(1, kernel_size=3, padding='same')(x)

    print ("Output shape = {}".format(np.shape(x)))

    unet = Model(inputs = [u_input_data, der_u_input_data], outputs=out, name='u_net_out')
        
    from keras.callbacks import TensorBoard

    adam = optimizers.Adam(lr=1e-2, decay=1e-4)
    unet.compile(loss='mean_squared_error', optimizer=adam)

#    X, y = build_dataset(roe)
    X = np.load("Xder248.npy")
    y = np.load("yder248.npy")    
    d = np.load("dder248.npy")    
    
    from sklearn.preprocessing  import StandardScaler

    scaler = StandardScaler()
    #        
    #    X = np.load("X.npy")
    #    y = np.load("y.npy")

    #    X_scaled = scaler.fit_transform(X)    

    xtr, ytr, dtr, xte, yte, dte, xval, yval, dval = split_data(X, d, y)
    
    x_train = np.reshape(xtr, (len(xtr), 248, 1))
    y_train = np.reshape(ytr, (len(ytr), 248, 1))
    d_train = np.reshape(dtr, (len(ytr), 248, 1))
    
    x_test = np.reshape(xte, (len(xte), 248, 1))
    y_test = np.reshape(yte, (len(yte), 248, 1))
    d_test = np.reshape(dte, (len(yte), 248, 1))

    x_validation = np.reshape(xval, (len(xval), 248, 1))
    y_validation = np.reshape(yval, (len(yval), 248, 1))
    d_validation = np.reshape(dval, (len(yval), 248, 1))
    
    unet.fit(x=[x_train, d_train], y=y_train,
            epochs=10,
            batch_size=256,
            shuffle=True,
            validation_data=([x_test, d_test], y_test),
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder2')])

    
#    tf.get_default_graph().finalize()

header_file = lambda Nt, Nx, CFL, amp, U_adv, init, harm, phase, it, real : \
        "Nt%d_Nx%d_CFL%s_amp%s_U_adv%s_%s_harm%s_phase%s_u_it%d_%d.npy" % \
        (Nt, Nx, CFL, amp, U_adv, init, harm, phase, it, real)

u = np.load('../cases/data/roe/' + header_file(100, 250, '0_40', '0_50', '0_90', 'sin', '1_00', '0_00', 0, 0))


def compare(u, U_adv, trained_model, roe=roe):
    f = lambda x : U_adv*x
    fprime = lambda x : U_adv
    
    u_for_ke = np.copy(u[1:-1])
    u_for_roe = np.copy(u)

    u_pred_next = np.zeros_like(u[1:-1]) 
    u_next = np.zeros_like(u) 
    
    for i in range(0, 200, 5) :
        for j in range(1, 249) :
            u_next[j] = solvers.timestep_roe(u_for_roe, j, 0.4, f, fprime)
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[2]
        
        u_pred_next = trained_model.predict(u_for_ke.reshape(1, 248, 1))
        
        plt.figure("Evolution comparaison")
        plt.clf()
#        plt.plot(roe.line_x, u_next, label="true next state it = %d" %(i+1))
        plt.plot(roe.line_x[1:-1], u_pred_next.ravel(), label="Prediction", fillstyle='none', linestyle='-.', marker='o' )
        plt.legend()
        plt.pause(0.1)
        
        u_for_ke = u_pred_next.ravel()
        u_for_roe = u_next 
        
