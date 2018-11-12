#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import os.path as osp
from sklearn.model_selection import train_test_split

import keras 
import tensorflow as tf

import time
import argparse

from keras import backend as K

from keras.utils.vis_utils import plot_model

K.clear_session()

first_weights = 10

#use optimizer.compute_gradients or tf.gradient to get original gradients
#then do whatever you want
#finally, use optimizer.apply_gradients


# Basic NN with Keras :
n_inputs_layers = 8
n_featurs = 4
n_inputs = n_inputs_layers*n_featurs

metrics = ["mean_squared_error", "mean_absolute_error"]

inputs_dict = {}
first_layers = {}

# COnstruction du model de I a O
# Voir Keras_AC_DDPG

for in_layer_n in range(n_inputs_layers) :
    inputs_dict["layers_n%03d" %(in_layer_n)] = keras.layers.Input(shape=(n_featurs,))
    first_layers["first_layers_n%03d" %(in_layer_n)] = keras.layers.Dense(1,
                                                     activation = 'selu',
                                                     bias_initializer = 'zeros',
                                                     kernel_initializer = 'random_normal'
                                                     )(inputs_dict["layers_n%03d" %(in_layer_n)])


unsorted_keys = first_layers.keys()
unsorted_keys.sort()
sorted_keys = unsorted_keys 

input_layers  = [inputs_dict[k.split("_")[1] + "_" + k.split("_")[2]]  for k in sorted_keys]
sorted_layers = [first_layers[k] for k in sorted_keys]

scd_layer = keras.layers.concatenate(sorted_layers)

model = keras.models.Model(input_layers, scd_layer)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

#distance_from_burger = tf.placeholder(tf.float32, [None, 1])

#adam = tf.train.AdamOptimizer(lr=0.001)
#model.compile(loss='mse')





