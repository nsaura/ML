#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os
import os.path as osp

import numdifftools as nd

import tensorflow as tf
import NN_class_try as NNC

#from threading import Thread, RLock
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

NNC = reload(NNC)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

import time

dx = 0.037037037037037035
dt = 0.014814814814814815
nu = 0.025
CFL = 0.4
Nx = 81
L = 3
line_x = np.arange(0,L,dx)

dict_layers = {"inputs" : [1], "N1" : [800, "selu"], "N2" : [400, "selu"], "action":[1,"tanh"]}

class Actor() :
    def __init__(self, dict_layers, lr, N_thread = 10, op_name="Adam", loss_name="OLS", reduce_type="sum", case_filename='./../cases/multinn_forwardNN/data/burger_dataset/'):
        nodes, acts = {}, {}
        
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu", "swish"]
        
        act_str_to_tf = {"elu"      :   tf.nn.elu,
                         "relu"     :   tf.nn.relu,
                         "tanh"     :   tf.nn.tanh,
                         "selu"     :   tf.nn.selu,
                         "sigmoid"  :   tf.nn.sigmoid,
                         "leakyrelu":   tf.nn.leaky_relu
                        }
        
        opt_str_to_tf = {"Adam"     :   tf.train.AdamOptimizer,
                         "RMS"      :   tf.train.RMSPropOptimizer,
                         "Momentum" :   tf.train.MomentumOptimizer,
                         "GD"       :   tf.train.GradientDescentOptimizer
                        }
        
        for item in dict_layers.iteritems() :
            if item[0] == "inputs" or item[0] == "action" :
                nodes[item[0]] = int(item[1][0])
            else :
                nodes["A_hn"+item[0][1:]] = int(item[1][0])
                acts["A_act"+item[0][1:]] = act_str_to_tf[str(item[1][1])]
        
        self.acts = acts
        self.nodes = nodes
        
        s_dim = dict_layers["inputs"][0]
        
        self.dict_layers = dict_layers
        self.N_thread = N_thread

        self.state = tf.placeholder(tf.float32, shape=(None, s_dim), name='inputs')
        self.action = tf.placeholder(tf.float32, shape=(None, s_dim), name='action')

        self.sess = tf.InteractiveSession(config=config)

        self.act_func_lst = act_func_lst
        
        self.last_key = lambda string: "A_%s_Last" % (string)
        self.last_name = lambda string: "Action_%s_Last" % (string)
        
        self.layer_key = lambda string, number : "A_%s%d" % (string, number)
        self.layer_name = lambda string, number : "Action_%s_%d" % (string, number)
        
        self.lr = lr
        self.op_name = op_name
        self.opt_str_to_tf = opt_str_to_tf
        
        self.loss_name = loss_name
        self.case_filename = case_filename
        
        self.reduce_fct = tf.reduce_sum if reduce_type == "sum" else tf.reduce_mean
        
        self.init_NN_Actor_parameter()
        self.build_NN_operating_Actor_graph()
        
        self.def_optimizer()
        
        self.network_params = tf.trainable_variables()
        
        # Op for periodically updating target network with online network
        # weights
#        self.update_target_network_params = \
#            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
#                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
#            for i in range(len(self.target_network_params))]
        
        self.action_gradient = tf.placeholder(tf.float32, (None, s_dim))
        
#        https://datascience.stackexchange.com/questions/18695/understanding-the-training-phase-of-the-tutorial-using-keras-and-deep-determini/18705#18705
        self.actor_gradients = tf.gradients(self.NN_guess, self.network_params, -self.action_gradient) # On multiplie le dernier grdient par le gradient de l'action
        
        self.optimize = self.tf_optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
###-------------------------------------------------------------------------------
        
        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
