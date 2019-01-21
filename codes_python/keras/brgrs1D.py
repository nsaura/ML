#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

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

import os
import sys 
import os.path as osp

import time

## Import de la classe CVC ##
cases_folder = osp.abspath(osp.dirname("../cases/"))
sys.path.append(cases_folder)

import tensorflow as tf

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, MaxPool1D, UpSampling1D, Activation, Conv1D

import Class_Vit_Choc as cvc
cvc = reload(cvc)

from visu_strides import glide1D

try :
    tf.reset_default_graph()
except :
    pass

def model():
    # Unet 
    input_shape = (80,1)

    input_data = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, padding='same')(input_data)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("in --> Conv1D(32, 2, same) = {}".format(x.get_shape())) # (?, 81, 32)

    x = Conv1D(32, kernel_size=2, padding='same')(input_data)
    x = Activation('relu')(x)   
    print ("Conv1D --> Conv1D(32, 2, same). Out_shape = {} \n".format(x.get_shape()))
    #print x.get_shape() # (?, 81, 32)

    in_pool_shape = x.get_shape()

    x = MaxPool1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")
    # 

    x = Conv1D(64, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Pooled --> Conv1D(64, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(64, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Conv1D --> Conv1D(64, 2, same). Out_shape = {} \n".format(x.get_shape()))

    in_pool_shape = x.get_shape()

    x = MaxPool1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(2nd enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")

    x = Conv1D(128, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Pooled --> Conv1D(128, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(128, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Conv1D --> Conv1D(128, 2, same). Out_shape = {} \n".format(x.get_shape()))

    in_samp_shape = x.get_shape()

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(1st dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
    print (" ")

    x = Conv1D(64, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Upsampled --> Conv1D(64, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(64, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Conv1D --> Conv1D(64, 2, same). Out_shape = {} \n".format(x.get_shape()))

    in_samp_shape = x.get_shape()

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
    print (" ")

    x = Conv1D(32, kernel_size=3, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Upsampled --> Conv1D(32, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(1, kernel_size=1, padding='same')(x)
    #Rajouter une couche BN
    output = Activation('relu')(x)
    print ("Conv1D --> Conv1D(1, 1, same) (Output). Out_shape = {}".format(x.get_shape()))

    unet = Model(input_data, output, name='u_net_out')
    
    
    return unet
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# u_it139_0_Nt52_Nx82_CFL0_4_nu0_05_sin.npy



parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

def build_dataset(cb) :
    nu52_avail = [0.05, 0.035, 0.025, 0.015]
    Nt = 52
    Nx = 82
    CFL = '0_4'
    str_nu52_avail = [str(i).replace('.', '_') for i in nu52_avail]
    
#    u_it168_4_Nt52_Nx82_CFL0_4_nu0_05_sin.npy
    
    true_data =lambda it,neval,s_nu : "u_it%d_%d_Nt52_Nx82_CFL0_4_nu%s_sin.npy" % (it, neval, s_nu)
    
    warp_name = lambda it, neval, s_nu : osp.join(cb.datapath, true_data(it, neval, s_nu))
    
    u_means = dict()
    
    
    uu = np.zeros((Nx))
    tempu = np.zeros(Nx-2)
    
    path_dataset = osp.join(cb.datapath, 'burger_matrices/cas_u')
    
    std_name = lambda std, it, s_nu : '%s_Nx_82_Nt_52_nu_%s_typei_sin_CFL_0_4_it_%03d.npy' % (std, s_nu, it)
    
    key_dict = lambda it, s_nu : "u_it%d_Nt52_Nx_82_nu%s" %(it, s_nu)

    ### Calcul de la moyenne 
    for s in str_nu52_avail :
        for i in range(cb.itmax) :
            tempu = np.zeros(Nx-2)
            for ev in range(cb.num_real) :
                uu = np.load(warp_name(i, ev, s))
                for j in range(1, Nx-1) :
                    tempu[j-1] += uu[j] / cb.num_real
                
            u_means[key_dict(i, s)] = tempu

    X, y = np.zeros_like(cb.line_x[1:-1]), np.zeros_like(cb.line_x[1:-1])
    cnt = 0
    
    X_keys, y_keys = dict(), dict()
    
    for s in str_nu52_avail :
        for i in range(cb.itmax) :
            betafile = os.path.join(cb.datapath, "burger_matrices/cas_u/betas", std_name("beta", i, s))
#            chol_file = os.path.join(cb.datapath, "burger_matrices/cas_u/cholesky", std_name(chol, i, s))
            
            vel_file = u_means[key_dict(i, s)]
            
            X = np.block([ [X], [vel_file] ])
            y = np.block([ [y], [np.load(betafile)[1:-1]] ])
            
            X_keys[cnt] = key_dict(i, s)
            y_keys[cnt] = betafile
            
            cnt += 1 
            
    X = np.delete(X, 0, axis=0) 
    y = np.delete(y, 0, axis=0)     
    return X, y, (cnt, X_keys, y_keys, u_means)

def split_data(X, y):
    permutation = np.random.permutation(len(y))
    
    X, y = np.copy(X), np.copy(y)
    
    X_rand = X[permutation]
    y_rand = y[permutation]
    
    xtr = X_rand[:int(len(y)*0.8)] 
    ytr = y_rand[:int(len(y)*0.8)]
    
    xte = X_rand[int(len(y)*0.8):int(len(y)*0.9)]
    yte = y_rand[int(len(y)*0.8):int(len(y)*0.9)]
    
    xval = X_rand[int(len(y)*0.9):]
    yval = y_rand[int(len(y)*0.9):]
    
    return xtr, ytr, xte, yte, xval, yval



if __name__ == '__main__':
    #    run brgrs1D.py -nu 5e-2 -itmax 180 -CFL 0.4 -num_real 5 -Nx 82 -Nt 32 -typeJ "u" -dp "./../cases/data/2019_burger_data/"
    parser = cvc.parser()
    cb = cvc.Vitesse_Choc(parser)
    
    unet = model()
    X, y, _ = build_dataset(cb)
    
    xtr, ytr, xte, yte, xval, yval = split_data(X, y)
    
    unet.compile(optimizer='adam', loss='mse')
    from keras.callbacks import TensorBoard
    
    x_train = np.reshape(xtr, (len(xtr), 80, 1))
    y_train = np.reshape(ytr, (len(ytr), 80, 1))
    
    x_test = np.reshape(xte, (len(xte), 80, 1))
    y_test = np.reshape(yte, (len(yte), 80, 1))
    
    x_validation = np.reshape(xval, (len(xval), 80, 1))
    y_validation = np.reshape(yval, (len(yval), 80, 1))
    
    unet.fit(x_train, y_train,
                epochs=1000,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, y_test))
#                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    
