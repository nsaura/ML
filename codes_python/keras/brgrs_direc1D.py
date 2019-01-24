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

import tensorflow as tf

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, MaxPool1D, UpSampling1D, Activation, Conv1D
import keras.optimizers as optimizers

import Roe_inference_burgers as rib
rib = reload(rib)

from visu_strides import glide1D

try :
    tf.reset_default_graph()
except :
    pass

def model(Nx)
    # Unet 
    input_shape = (Nx,1)

    input_data = Input(shape=input_shape)
    x = Conv1D(2, kernel_size=3, padding='same')(input_data)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("in --> Conv1D(32, 2, same) = {}".format(x.get_shape())) # (?, 81, 32)

    x = Conv1D(4, kernel_size=2, padding='same')(input_data)
    x = Activation('relu')(x)   
    print ("Conv1D --> Conv1D(32, 2, same). Out_shape = {} \n".format(x.get_shape()))
    #print x.get_shape() # (?, 81, 32)

    in_pool_shape = x.get_shape()

    x = MaxPool1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")
    # 

    x = Conv1D(8, kernel_size=2, padding='same')(x)
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

    x = Conv1D(16, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Pooled --> Conv1D(128, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(16, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Conv1D --> Conv1D(128, 2, same). Out_shape = {} \n".format(x.get_shape()))

    in_samp_shape = x.get_shape()

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(1st dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
    print (" ")

    x = Conv1D(8, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Upsampled --> Conv1D(64, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(4, kernel_size=2, padding='same')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Conv1D --> Conv1D(64, 2, same). Out_shape = {} \n".format(x.get_shape()))

    in_samp_shape = x.get_shape()

    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
    print (" ")

    x = Conv1D(2, kernel_size=3, padding='valid')(x)
    #Rajouter une couche BN
    x = Activation('relu')(x)
    print ("Upsampled --> Conv1D(32, 2, same). Out_shape = {}".format(x.get_shape()))

    x = Conv1D(1, kernel_size=1, padding='valid')(x)
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
cb = crb.Vitesse_Choc(parser)

def build_dataset(roe) :
    data_file = csv.reader(open("roe_inference_case_done.txt","r"), delimiter="\t")
    lines = []
    
    for line in data_file :
        lines.append(line)
    
    return lines
    
    n_params = np.shape(p)[1]
    n_case = np.shape(p)[1] - 1 
    
    p = [[lines[j][i].replace(' ','') for j in range(n_case)] for i in range(n_params)]
    
    for l in range(len(p)) :
        for j in range(1, n_case + 1) : #+1 because key is still in lists
            try :
                p[l][j] = p[l][j]
            except :
                pass
    
    keys = []
    cases = []
    
    keys = [p[j][0] for j in range(n_params)]
    cases = [p[j][1:] for j in range(n_params)] 
    
    return p, keys, cases
    
    header_file = lambda Nx, Nt, CFL, amp, harm, phase, it, real : \
        "Nt%d_Nx%d_CFL%s_sin_amp%s_harm%s_phase%s_U_adv%s_u_it%d_%d.npy" % \
        (Nx, Nt, CFL, amp, harm, phase, it, real)
        
    
    
    
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

def see_pred(cb, unet, u_means, key, indice) :
    u_n = xval[indice]
    beta_pred = unet.predict(u_n.reshape(1, 82, 1)).ravel()
    
    u_n = u_n.ravel()
    true_beta = yval[indice].ravel()

#    u_beta(self, beta, u_n, verbose=False)
    u_npred = cb.u_beta(beta_pred, u_n)
    
#    u_nexact = 
    
    

if __name__ == '__main__':
    #    run brgrs1D_beta.py -nu 5e-2 -itmax 180 -CFL 0.4 -num_real 5 -Nx 82 -Nt 32 -typeJ "u" -dp "./../cases/data/2019_burger_data/"
    parser = rib.parser()
    roe = rib.Class_Roe_BFGS(parser)
    
    unet = model(roe.Nx)
    X, y, utils = build_dataset(roe)
    
    u_means = utils[-1]
    
    xtr, ytr, xte, yte, xval, yval = split_data(X, y)
    
    from keras.callbacks import TensorBoard
    
    adam = optimizers.Adam(lr=1e-2, decay=1e-5)
    unet.compile(loss='mean_squared_error', optimizer=adam)
    
    x_train = np.reshape(xtr, (len(xtr), 82, 1))
    y_train = np.reshape(ytr, (len(ytr), 82, 1))
    
    x_test = np.reshape(xte, (len(xte), 82, 1))
    y_test = np.reshape(yte, (len(yte), 82, 1))
    
    x_validation = np.reshape(xval, (len(xval), 82, 1))
    y_validation = np.reshape(yval, (len(yval), 82, 1))
    
    unet.fit(x_train, y_train,
                epochs=10,
#                batch_size=32,
                shuffle=True,
                validation_data=(x_test, y_test))
#                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    
