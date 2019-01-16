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

# Unet 
input_shape = (82,1)

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

x = Conv1D(32, kernel_size=3, padding='valid')(x)
#Rajouter une couche BN
x = Activation('relu')(x)
print ("Upsampled --> Conv1D(32, 2, same). Out_shape = {}".format(x.get_shape()))

x = Conv1D(1, kernel_size=1, padding='same')(x)
#Rajouter une couche BN
output = Activation('relu')(x)
print ("Conv1D --> Conv1D(1, 1, same) (Output). Out_shape = {}".format(x.get_shape()))

unet = Model(input_data, output, name='u_net_out')

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# u_it139_0_Nt52_Nx82_CFL0_4_nu0_05_sin.npy
#    run brgrs1D.py -nu 5e-2 -itmax 180 -CFL 0.4 -num_real 5 -Nx 82 -Nt 32 -typeJ "u" -dp "./../cases/data/2019_burger_data/"
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
    
    key_dict = lambda it, s_nu : "u_it%d_Nt52_Nx_82_nu%s" %(it, s_nu)
    
    uu = np.zeros((Nx))
    tempu = np.zeros(Nx-2)
    
    for s in str_nu52_avail :
        for i in range(cb.itmax) :
            tempu = np.zeros(Nx-2)
            for ev in range(cb.num_real) :
                uu = np.load(warp_name(i, ev, s))
                for j in range(1, Nx-1) :
                    tempu[j-1] += uu[j] / cb.num_real
                
            u_means[key_dict(i, s)] = tempu
            
#            plt.figure("Tracage des moyennes")
#            plt.plot(cb.line_x[1:-1], u_means[key_dict(i, s)], label="it:%d" %i)
#        plt.ylim(-1,1)
#        plt.legend()
#        plt.pause(0.4)
#        
#    u_name = lambda it,  
    
    return u_means
    
