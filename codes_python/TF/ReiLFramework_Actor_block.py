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

from sklearn.model_selection import train_test_split

import tflearn
import tensorflow as tf

import custom_swish_function as csf

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

csf = reload(csf)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

#dict_layers = {"inputs" : [1], "N1" : [20, "tanh"], "N2" : ["20", "tanh"], "action":[1]}

###-------------------------------------------------------------------------------

class actor_deepNN() :
    def __init__(self, dict_layers, lr, op_name="Adam", loss_name="OLS", reduce_type="sum"):
        nodes, acts = {}, {}
        
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu", "swish"]
        
        act_str_to_tf = {"relu"     :   tf.nn.relu,
                         "tanh"     :   tf.nn.tanh,
                         "selu"     :   tf.nn.selu,
                         "swish"    :   csf.tf_swish,
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
        
        self.reduce_fct = tf.reduce_sum if reduce_type == "sum" else tf.reduce_mean
        
###-------------------------------------------------------------------------------

    def init_NN_Actor_parameter(self) :
        
        prev_key = "inputs"
        weights = dict()
        
        for nlayer in range(1, len(self.acts.keys())+1) :
            prev_node = self.nodes[prev_key]
            curr_node_key = self.layer_key("hn", nlayer)

            curr_node = self.nodes[curr_node_key]
            
#            print ("prev_key : {}, type = {}".format(prev_key, type(prev_key)))
#            print ("prev_node : {}, type = {}".format(prev_node, type(prev_node)))
#            
#            print ("curr_node : {}, type = {}".format(curr_node, type(curr_node)))
#            print ("\n")
            
            tempWeight = np.random.randn(prev_node, curr_node) / prev_node
            
            dict_key = self.layer_key("w", nlayer)
            weights_tf_name = self.layer_name("Weight", nlayer)
            
            weights[dict_key] = tf.Variable(tempWeight.astype(np.float32), name=weights_tf_name)
            
            prev_key = curr_node_key
        
        prev_node = self.nodes[prev_key]
        curr_node = self.nodes["action"]
        
        tempWeight = np.random.randn(prev_node, curr_node) / prev_node
        
        weights[self.last_key("w")] = tf.Variable(tempWeight.astype(np.float32),\
                                         name=self.last_name("Weight"))
        
        #La boucle est bouclée
        
        for item in weights.iteritems() :
            print ("item : {}, shape = {}".format(item[0], np.shape(item[1])))
        
        self.weights = weights
        
###-------------------------------------------------------------------------------
    
    def build_NN_operating_Actor_graph(self, tau=0.0001) :
        action_layers = dict()
        incoming_layer = self.state
        
        for nlayer in range(1, len(self.acts.keys())+1) :
            w_key = self.layer_key("w", nlayer)
            a_key = self.layer_key("act", nlayer)
            
#        for kact, kw in zip(self.acts.keys(), self.weights.keys()) :        
#            print kact, kw
#            nlayer = int(kact[1:])
            action_layer = tf.matmul(incoming_layer, self.weights[w_key])
            action_layer = self.acts[a_key](action_layer, name=self.layer_name("act", nlayer))
            
            action_layers[a_key] = action_layer
            
            incoming_layer = action_layer
        
        last_action_layer = tf.matmul(incoming_layer, self.weights[self.last_key("w")])
        action_layers[self.last_key("act")] = self.acts[a_key](last_action_layer,\
                                            self.last_name("act"))
        
        print action_layers

        self.NN_guess = last_action_layer
        self.action_layers = action_layers
        
###-------------------------------------------------------------------------------

    def def_optimizer(self, decay=0.9, beta1=0.9, beta2=0.99, momentum=0.0, nest=False):
        
        tf_optimizer = self.opt_str_to_tf[self.op_name]
        
        if self.op_name == "GD" :
            tf_optimizer = tf_optimizer(self.lr, name=self.op_name)
        
        if self.op_name == "Momentum" :
            tf_optimizer = tf_optimizer(self.lr, momentum, use_nesterov=nest, name=self.op_name)
                    
        if self.op_name == "Adam" :
            tf_optimizer = tf_optimizer(self.lr, beta1, beta2, name=self.op_name)
        
        if self.op_name == "RMS" : 
            tf_optimizer = tf_optimizer(self.lr, decay, momentum, name=self.op_name)
        
        self.tf_optimizer = tf_optimizer
        
###-------------------------------------------------------------------------------

    def def_loss(self, r=1, alpha=1e-7) :
        
        classic_loss = {"OLS" = tf.square, "AVL" : tf.abs}
        regress_loss = ["Lasso", "Ridge", "Elastic"]
        
        if self.loss_name in classic_loss.keys() :
            loss_function = classic_loss[self.loss_name]
        
        else :
            if self.loss_name == "Lasso" :
                loss_function = self.Elastic_cost(r=1, alpha=alpha)
            
            if self.loss_name == "Ridge" :
                loss_function = self.Elastic_cost(r=0, alpha=alpha)
            
            if self.loss_name == "Elastic" :
                loss_function = self.Elastic_cost(r=r, alpha=alpha)
        
        self.loss_name = self.reduce_fct(loss_function)
        
###-------------------------------------------------------------------------------        
        
    def Elastic_cost(self, r=1) :
        # Pseudo code : 
        # J = MSE + r*alpha * sum(|weights|) + (1-r)*0.5*alpha * sum(tf.square(weights))
        # if r == 0 ---> Ridge
        # if r == 1 ---> Lasso 
        
        if np.abs(r) < 1e-8 :
            print ("r = %f --> Ridge function selected" % r)
        
        if np.abs(r-1) < 1e-8 :
            print ("r = %f --> Lasso function selected" % r)
        
        self.weight_sum = tf.placeholder(np.float32, (None), name="weight_sum")
        
        MSE_part = self.reduce_type_fct(tf.square(self.NN_guess - self.action))
        
        Ela_part = tf.add(tf.multiply(r*alpha, self.reduce_type_fct(tf.abs(self.weight_sum))),
               tf.multiply((1.-r)*0.5*alpha, self.reduce_type_fct(tf.square(self.weight_sum))))
        
        return tf.add(MSE_part, Ela_part) 
                
###-------------------------------------------------------------------------------                
                
    def update_network_params(self, tau=0.001) :
        self.network_params = tf.trainable_variables()
        target_network_variables = tf.trainable_variables()[len(network_params):]
        
        
        self.update_target_network_params = \
            [target_network_variables[i].assign(tf.mul(network_params[i], tau) +\
             tf.mul(target_network_variables[i], 1-tau))
             for i in range(len(target_network_variables))]
     
     def update_target_network(self):   
        
       
###-------------------------------------------------------------------------------        
    
    def train_DRL(self) :
        self.action_gradient = tf.placeholder(tf.float32, (None, 1))
        
        self.unnormalized_actor_gradient = tf.gradients(\
                    self.NN_guess, self.network_params, -self.action_gradient)
        
        self.tf_optimizer = tf_optimizer.minimize.\
                    apply_gradients(zip(self.actor_gradients, self.network_params))
        
        
        

        
if __name__=="__main__" :
#    run ReiLFramework_Actor_block.py
    dict_layers = {"inputs" : [1], "N1" : [20, "tanh"], "N2" : [40, "tanh"], "action":[1]}
    
    actor = actor_deepNN(dict_layers)
    actor.init_NN_Actor_parameter()
    

        
#        log_path = os.path.abspath("./logs")
#        now = datetime.utcnow().strftime("%Y_%m_%d_%Hh%m_%S")

#        dir_name = os.path.join(log_path, now)
#        
#        log_dir = "{}/run-{}".format(dir_name, now)
#        
#        self.log_dir = log_dir
#        self.dir_name = dir_name
#                    
#        file_writer = tflearn.summaries.get_summary(self.log_dir, tf.get_default_graph()) 
#        file_writer.close()   
                        
        
