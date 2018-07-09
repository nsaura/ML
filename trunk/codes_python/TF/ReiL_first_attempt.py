#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os

from sklearn.model_selection import train_test_split

from datetime import datetime

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

import tensorflow as tf

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    
sess = tf.InteractiveSession(config=config)

x = np.linspace(0,1,500).reshape(-1,1)
s = np.sin(2*np.pi*x).reshape(1,-1)

s_dim = x.shape

# add an extra layer just for fun
M1 = 20

w1_init = np.random.randn(s_dim[1], M1) / 28
b1_init = np.zeros(M1)
w2_init = np.random.randn(M1, s_dim[1]) / np.sqrt(M1)
b2_init = np.zeros(s_dim[1])

# define variables and expressions
inputs = tf.placeholder(tf.float32, shape=(None, s_dim[1]), name='inputs')
targets = tf.placeholder(tf.float32, shape=(None, s_dim[1]), name='target')

w1 = tf.Variable(w1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
w2 = tf.Variable(w2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))

# define the model
z1 = tf.nn.relu( tf.matmul(inputs, w1) + b1 )
out = tf.matmul(z1, w2) + b2 # remember, the cost function does the softmaxing! weird, right?

action_bound = np.ones((s_dim[1])) 

scaled_out = tf.multiply(out, action_bound)

# Pour l'update des params
network_params = tf.trainable_variables()
target_network_variables = tf.trainable_variables()[len(network_params):]

tau = 0.001 # Mixing factor commonly added

update_target_network_params =\
    [target_network_variables[i].assign(tf.mul(network_params[i], tau) +\
     tf.mul(target_network_variables[i], 1-tau))
     for i in range(len(target_network_variables))]

# En rapport avec le gradient :
action_gradients = tf.placeholder(tf.float32, [None, s_dim[1]])

# Dividing by batch size if needed (classification)
unnormalized_actor_gradient = tf.gradients(scaled_out, network_params, -action_gradient))

# On divise chaque gradient par la taille du batch
actor_gradients = list(map(labda x : tf.div(x, batch_size), unnormalized_actor_gradients))

# Optimizer avec les bons gradients

optimize = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(actor_gradients, network_params))

#Pour faire des tests ensuite
init = tf.global_variables_initializer()
sess.run(init)




