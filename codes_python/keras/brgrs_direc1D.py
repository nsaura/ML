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
from keras.layers import Input, Flatten, Dense, MaxPool1D, UpSampling1D, Activation, Conv1D, concatenate
from keras.layers.normalization import BatchNormalization

import keras.optimizers as optimizers

import Roe_inference_burgers as rib
rib = reload(rib)
solvers = reload(solvers)

from visu_strides import glide1D

try :
    tf.reset_default_graph()
    print ("Graph reseted")
except :
    pass

#def model(Nx, first_filter) :
#    # Unet 
#    input_shape = (Nx,1)

#    input_data = Input(shape=input_shape)
#    x = Conv1D(first_filter, kernel_size=3, padding='same')(input_data)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("in --> Conv1D(32, 2, same) = {}".format(x.get_shape())) # (?, 81, 32)

#    x = Conv1D(first_filter*2, kernel_size=2, padding='same')(input_data)
#    x = Activation('relu')(x)   
#    print ("Conv1D --> Conv1D(32, 2, same). Out_shape = {} \n".format(x.get_shape()))
#    #print x.get_shape() # (?, 81, 32)

#    in_pool_shape = x.get_shape()

#    x = MaxPool1D(pool_size=2, padding='same')(x)
#    print ("MaxPool1D pool_size = 2, padding same")
#    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
#    print (" ")
#    # 

#    x = Conv1D(first_filter*4, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Pooled --> Conv1D(64, 2, same). Out_shape = {}".format(x.get_shape()))

#    x = Conv1D(first_filter*8, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Conv1D --> Conv1D(64, 2, same). Out_shape = {} \n".format(x.get_shape()))

#    in_pool_shape = x.get_shape()

#    x = MaxPool1D(pool_size=2, padding='same')(x)
#    print ("MaxPool1D pool_size = 2, padding same")
#    print ("MaxPool1D(2nd enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
#    print (" ")

#    x = Conv1D(first_filter*16, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Pooled --> Conv1D(128, 2, same). Out_shape = {}".format(x.get_shape()))

#    x = Conv1D(first_filter*16, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Conv1D --> Conv1D(128, 2, same). Out_shape = {} \n".format(x.get_shape()))

#    in_samp_shape = x.get_shape()

#    x = UpSampling1D(2)(x)
#    print ("Upsampling pool_size = 2")
#    print ("UpSampling(1st dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
#    print (" ")

#    x = Conv1D(first_filter*8, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Upsampled --> Conv1D(64, 2, same). Out_shape = {}".format(x.get_shape()))

#    x = Conv1D(first_filter*4, kernel_size=2, padding='same')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Conv1D --> Conv1D(64, 2, same). Out_shape = {} \n".format(x.get_shape()))

#    in_samp_shape = x.get_shape()

#    x = UpSampling1D(2)(x)
#    print ("Upsampling pool_size = 2")
#    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_samp_shape, x.get_shape()))
#    print (" ")

#    x = Conv1D(first_filter*2, kernel_size=3, padding='valid')(x)
#    #Rajouter une couche BN
#    x = Activation('relu')(x)
#    print ("Upsampled --> Conv1D(32, 2, same). Out_shape = {}".format(x.get_shape()))

#    x = Conv1D(1, kernel_size=1, padding='valid')(x)
#    #Rajouter une couche BN
#    output = Activation('relu')(x)
#    print ("Conv1D --> Conv1D(1, 1, same) (Output). Out_shape = {}".format(x.get_shape()))

#    unet = Model(input_data, output, name='u_net_out')
#    
#    
#    return unet
##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# u_it139_0_Nt52_Nx82_CFL0_4_nu0_05_sin.npy

def build_dataset() :
    data_file = csv.reader(open("./../cases/data/roe/roe_inference_case_done.txt","r"), delimiter="\t")
    lines = []
    
#    for line in data_file :
#        lines.append(line)
#    
#    n_params = np.shape(lines)[1]
#    n_cases = np.shape(lines)[0] - 1 
#    
#    p = [[lines[j][i].replace(' ','') for j in range(n_cases)] for i in range(n_params)]
#    
#    for l in range(len(p)) :
#        for j in range(1, n_case + 1) : #+1 because key is still in lists
#            try :
#                p[l][j] = p[l][j]
#            except :
#                pass
#    
#    keys = []
#    cases = []
#    
#    keys = [p[j][0] for j in range(n_params)]
#    cases = [p[j][1:] for j in range(n_params)] 
#    
#    print np.shape(p)
#    
    
    
#    for c in range(n_cases) :
#        for it in range(200) :
#            for ind, kk in enumerate(keys[:-1]) :
#                if kk == 'init' :
#                    ll += "sin_" 
#                else :
#                    try :
#                        ll +='%s%s_' %(kk, v[ind][c])
#                    except :
#                        print "kk = {}\tind={}\tc={}".format(kk, ind, c)
#                        print "ll = {}".format(ll)
#                        return v[ind]   
#            ll +="u_it%d_0.npy" % it

#    Nx250_Nt100_CFL0_40_amp1_00_U_adv0_30_sin_harm1_00_phase0_00_u_it58_5.npy

    X = np.zeros_like(range(248)) # roe.Nx
    y = np.zeros_like(range(248)) # roe.Nx

    header_file = lambda Nt, Nx, CFL, amp, U_adv, init, harm, phase, it, real : \
        "Nt%d_Nx%d_CFL%s_amp%s_U_adv%s_%s_harm%s_phase%s_u_it%d_%d.npy" % \
        (Nt, Nx, CFL, amp, U_adv, init, harm, phase, it, real)
    Nx = 250
    Nt = 100
    init = 'sin'

    cnt = 0
    
    CFL = '0_40'
    for amp in ['0_50', '1_00', '1_50']:
        for u in ['0_30', '0_60', '0_90', '1_20', '1_50'] :
            for h in ['1_00', '2_00', '3_00'] :
                for ph in ['0_00', '0_17', '0_33', '0_50'] :
                    for it in range(199) :
                    
                        ffile = header_file(Nt, Nx, CFL, amp, u, init, h, ph, it, 0)
                        X = np.block([ [X], [np.load('../cases/data/roe/' + ffile)[1:-1]] ])

                        ffile = header_file(Nt, Nx, CFL, amp, u, init, h, ph, it+1, 0)
                        y = np.block([ [y], [np.load('../cases/data/roe/' + ffile)[1:-1]] ])
                        cnt += 1
    
    print cnt
    
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    return X, y 
    
#    for j in range(1, len(self.line_x)-1)
    
#    return X, y, (cnt, X_keys, y_keys, u_means)

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

def see_pred(roe, unet, u_means, key, indice) :
    u_n = xval[indice]
    beta_pred = unet.predict(u_n.reshape(1, roe.Nx, 1)).ravel()
    
    u_n = u_n.ravel()
    true_beta = yval[indice].ravel()

#    u_beta(self, beta, u_n, verbose=False)
    u_npred = cb.u_beta(beta_pred, u_n)
    
#    u_nexact = 
    
    

if __name__ == '__main__':
    #    run brgrs_direc1D.py -num_real 1 -itmax 180 -CFL 0.4 -Nx 250 -Nt 100 -typeJ "u" -dp "./../cases/data/2019_burger_data/"
    parser = rib.parser()
    roe = rib.Class_Roe_BFGS(parser, call=True)
    
#    unet = model(roe.Nx, 2)
#    X, y, utils = build_dataset(roe)
    
    print (" ")    
        
    input_shape = (roe.Nx-2, 1)
    input_data = Input(shape=input_shape)
    
    x = Conv1D(3, kernel_size=3, padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(input_shape, x.get_shape())) 
    
    in_pool_shape = np.shape(x)
    
    x = Conv1D(3, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 
    
    to_concatene_first = x
    
    in_pool_shape = np.shape(x)
     
    x = MaxPool1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")
    
#	Out_shape = (?, 124, 3)
    
        
    in_pool_shape = np.shape(x)
    
    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 
    
    in_pool_shape = np.shape(x)
    
    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 
    
    to_concatene_second = x
    
    in_pool_shape = np.shape(x)
     
    x = MaxPool1D(pool_size=2, padding='same')(x)
    print ("MaxPool1D pool_size = 2, padding same")
    print ("MaxPool1D(1st enc): in_shape = {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")
    
    in_pool_shape = np.shape(x)
    
    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 
    
    in_pool_shape = np.shape(x)
    
    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(3,3) = {}".format(in_pool_shape, x.get_shape())) 

    in_pool_shape = np.shape(x)
    
    x = Conv1D(12, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
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
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(6,2) = {}".format(in_pool_shape, x.get_shape())) 
    
    in_pool_shape = np.shape(x)
    
    x = Conv1D(6, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print ("in {}--> Conv1D(6,2) = {}".format(in_pool_shape, x.get_shape())) 
    
    in_pool_shape = np.shape(x)
    
    x = UpSampling1D(2)(x)
    print ("Upsampling pool_size = 2")
    print ("UpSampling(2nd dec): in_shape= {}\tOut_shape = {}".format(in_pool_shape, x.get_shape()))
    print (" ")
    
    y = concatenate([to_concatene_first, x])
    
    x = Conv1D(3, kernel_size=1, padding='same')(y)
    x = Conv1D(1, kernel_size=3, padding='same')(x)
    
    print ("Output shape = {}".format(np.shape(x)))
    
    X, y = build_dataset()
    
    from sklearn.preprocessing  import StandardScaler

    scaler = StandardScaler()
#        
#    X = np.load("X.npy")
#    y = np.load("y.npy")
    
#    X_scaled = scaler.fit_transform(X)    
    
    xtr, ytr, xte, yte, xval, yval = split_data(X, y)
    
    unet = Model(input_data, x, name='u_net_out')
        
    from keras.callbacks import TensorBoard
    
    adam = optimizers.Adam(lr=1e-2, decay=1e-5)
    unet.compile(loss='mean_squared_error', optimizer=adam)

    x_train = np.reshape(xtr, (len(xtr), 248, 1))
    y_train = np.reshape(ytr, (len(ytr), 248, 1))

    x_test = np.reshape(xte, (len(xte), 248, 1))
    y_test = np.reshape(yte, (len(yte), 248, 1))

    x_validation = np.reshape(xval, (len(xval), 248, 1))
    y_validation = np.reshape(yval, (len(yval), 248, 1))


    unet.fit(x_train, y_train,
            epochs=150,
            batch_size=256,
            shuffle=True,
            validation_data=(x_test, y_test),
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

          

def compare(u, U_adv, trained_model, roe=roe):
    f = lambda x : U_adv*x
    fprime = lambda x : U_adv
    
    u_for_ke = u
    u_for_roe = u

    u_pred_next = np.zeros_like(u) 
    u_next = np.zeros_like(u) 
    
    for i in range(200) :
        for j in range(1, 249) :
            u_next[j] = solvers.timestep_roe(u_for_roe, j, 0.4, f, fprime)
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[2]
        
        u_pred_next = trained_model.predict(u_for_ke.reshape(1, 250, 1))
        
        plt.figure("Evolution comparaison")
        plt.clf()
#        plt.plot(roe.line_x, u_next, label="true next state it = %d" %(i+1))
        plt.plot(roe.line_x, u_pred_next.ravel(), label="Prediction", fillstyle='none', linestyle='-.', marker='o' )
        plt.legend()
        plt.pause(0.1)
        
        u_for_ke = u_pred_next.ravel()
        u_for_roe = u_next 
        
        
    
#    
