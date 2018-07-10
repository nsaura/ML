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

import tflearn

import tensorflow as tf

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    
sess = tf.InteractiveSession(config=config)


x = np.linspace(0,1,500).reshape(-1,1)
s = np.sin(2*np.pi*x).reshape(1,-1)

s_dim = x.shape

M1 = 20

##############################
##############################
######  Pour l'action  #######
##############################
##############################

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

##############################
##############################
###### Pour la critique ######
##############################
##############################

w11_init = np.random.randn(s_dim[1], M1) / 28
b11_init = np.zeros(M1)
w22_init = np.random.randn(M1, s_dim[1]) / np.sqrt(M1)
b22_init = np.zeros(s_dim[1])

# define variables and expressions
inputs = tf.placeholder(tf.float32, shape=(None, s_dim[1]), name='inputs')
actions = tf.placeholder(tf.float32, shape=(None, s_dim[1]), name='target')

# Architecture du critique particuliere : première entrée puis deuxième 
# On va utiliser tflearn.fully_connected
net = tflearn.fully_connected(inputs, 20) #crée matrices de poids et biais pour cette couche
#net = tflearn.layers.normalization.batch_normalization(net)
net = tflearn.activations.relu(net)

t1 = tflearn.fully_connected(net, 20) #On crée une deuxième hidden layer pour les inputs
t2 = tflearn.fully_connected(actions, 20) #On crée une autre branche

net_crits = tf.activation(
        tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

# Puis une couche linear representant la TD valeur cad la Qvalue ici
w_init = tflearn.initializations.uniforme(minval=-0.003, minval=+0.003) #Toujours pour bounder
out_crits = tflearn.fully_connected(net_crits, 1, weights_init=w_init)


#Fonction de cout pour le critic
predicted_q_value = tf.placeholder(tf.float, [None, 1])

loss_crits = tflearn.mean_square(predicted_q_value, out_crits)
optimize = tf.train.AdamOptimizer(learning_rate_crits).minimize(loss)

action_grads = tf.gradients(out_crits, action) #Grad Out wrt action

#Pour faire des tests ensuite
init = tf.global_variables_initializer()
sess.run(init)




