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


#dict_layers = {"inputs" : [1], "N1" : [20, "tanh"], "N2" : ["20", "tanh"], "action":[1]}

###-------------------------------------------------------------------------------

class actor_deepNN() :
    def __init__(self, dict_layers):
        nodes, acts = {}, {}
        
        for item in dict_layers.iteritems() :
            if item[0] == "inputs" or item[0] == "action" :
                nodes[item[0]] = int(item[1][0])
            else :
                nodes[item[0]] = int(item[1][0])
                acts[item[0]] = str(item[1][1])
        
        nodes["inputs"]
        
        self.acts = acts
        self.nodes = nodes
        
        s_dim = dict_layers["inputs"][0]
        
        self.dict_layers = dict_layers

        self.state = tf.placeholder(tf.float32, shape=(None, s_dim), name='inputs')
        self.action = tf.placeholder(tf.float32, shape=(None, s_dim), name='action')
        
        self.sess = tf.InteractiveSession(config=config)

###-------------------------------------------------------------------------------

    def init_NN_parameter(self) :
        
        prev_key = "inputs"
        weights = dict()
        
        for k in self.acts.keys() :
            prev_node = self.nodes[prev_key]
            curr_node = self.nodes[k]
            
#            print ("prev_key : {}, type = {}".format(prev_key, type(prev_key)))
#            print ("prev_node : {}, type = {}".format(prev_node, type(prev_node)))
#            
#            print ("curr_node : {}, type = {}".format(curr_node, type(curr_node)))
#            print ("\n")
            
            tempWeight = np.random.randn(prev_node, curr_node) / prev_node
            
            nlayer = int(k[1:])
            dict_key = "A_w%d" %(nlayer)
            weights_tf_name = "Action_Weight_%d" %(nlayer)
            
            weights[dict_key] = tf.Variable(tempWeight.astype(np.float32), name=weights_tf_name)
            
            prev_key = k
        
        prev_node = self.nodes[prev_key]
        curr_node = self.nodes["action"]
        
        tempWeight = np.random.randn(prev_node, curr_node) / prev_node
        
        weights["A_wlast"] = tf.Variable(tempWeight.astype(np.float32), name="Action_Weight_Last")
        
        #La boucle est bouclée
        
        for item in weights.iteritems() :
            print ("item : {}, shape = {}".format(item[0], np.shape(item[1])))
        
        self.weights = weights
        
###-------------------------------------------------------------------------------
    
        
    
###-------------------------------------------------------------------------------

if __name__=="__main__" :
    dict_layers = {"inputs" : [1], "N1" : [20, "tanh"], "N2" : [40, "tanh"], "action":[1]}
    
    actor = actor_deepNN(dict_layers)
    actor.init_NN_parameter()
    

        
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
                        
        
