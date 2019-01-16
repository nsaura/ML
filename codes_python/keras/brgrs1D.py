#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

import time

from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, MaxPool1D, UpSampling1D, Activation, Conv1D

import tensorflow as tf

from visu_strides import glide1D

try :
    tf.reset_default_graph()
except :
    pass

# Unet 
input_shape = (82,1)

input_data = Input(shape=input_shape)
x = Conv1D(32, kernel_size=3)(input_data)
#Rajouter une couche BN
x = Activation('relu')(x)

x = Conv1D(32, kernel_size=2)(input_data)
x = Activation('relu')(x)   
print x.get_shape() # (?, 81, 32)

x = MaxPool1D(pool_size=2, padding='valid')(x)
print x.get_shape() # (?, 41, 32)
# 

x = Conv1D(64, kernel_size=2)(x)
#Rajouter une couche BN
x = Activation('relu')(x)

x = Conv1D(64, kernel_size=2)(x)
#Rajouter une couche BN
x = Activation('relu')(x)

x = MaxPool1D(pool_size=2, padding='same')(x)
print x.get_shape() # (?, 20, 64)

x = Conv1D(128, kernel_size=2, padding='same')(x)
#Rajouter une couche BN
x = Activation('relu')(x)

x = Conv1D(128, kernel_size=2, padding='same')(x)
#Rajouter une couche BN
x = Activation('relu')(x)
print x.get_shape() # (?, 20, 64)

x = UpSampling1D(2)(x)
print x.get_shape()

### Watch Dims
