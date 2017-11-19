#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os, csv
from sklearn.model_selection import train_test_split

import tensorflow as tf

class Neural_Network():
###-------------------------------------------------------------------------------
    def __init__(self, lr, N_={}, max_epoch=10, pathfile="",**kwargs) :
        inputs = dict()
        for kway in kwargs :
            inputs[kway] = kwargs[kway]

        ## We check if the file indexed by "pathfile" variable exists
        ## Otherwise, we create it
        
        if os.path.isfile(pathfile) == False :
            try :
                N_hidden_layer = kwargs["N_hidden_layer"]
                print N_hidden_layer, type(N_hidden_layer)
    
            except KeyError:
                raise KeyError ("You have to create an initializer file or to define N_hidden_layer when creating Neural_Network object ")
    
            try : 
                layers_sizes = kwargs["layers_sizes"]
                print layers_sizes, type(layers_sizes)
                if len(layers_sizes) is not N_hidden_layer + 2 :  
                    raise ValueError("ValueError : layers should have the length of N_hidden_layer + 2 since it contains the number of sites of the input, hidden and output layers ")

            except :
                raise KeyError("You have to create an initializer file or to give layers as an argument when constructing the Neural_Network object.")
            
            line = 0 
            row = "N%d=%d\n" %(line, layers_sizes[line])
            f = open(pathfile, "w")
            while line < len(layers_sizes) - 1 :
                f.write(row)
                line += 1
                row = "N%d=%d\n" %(line, layers_sizes[line])
                
            f.write("K=%d\n" %(layers_sizes[line]))
            f.close()
        
        if len(N_.keys())  == 0 :
            try :
                f = csv.reader(open(pathfile, "r"), delimiter="=")
                for line in f :
                    N_[line[0]] = int(line[1])

            except IOError:
                raise IOError("N_ is empty and pathfile is {} (inexistant)".format(pathfile))
        
#        n_values_type = [type(i)==type(int) for i in N_.values()]
#        if n_values_type == False : 
#            N_ = dict((k,int(v)) for k,v in N_.interitems())
        
        self.wlastkey = "wlast"
        self.blastkey = "blast"
        
        self.inputs = inputs
        self.N_ = N_
###-------------------------------------------------------------------------------        
    def train_and_split(self, X, y, random_state=0, strat=True, scale=False):
        
        # Stratify option allows to have a loyal representation of the datasets (couples of data,target)
        # And allows a better training and can prevent from overfitting
        
#        #Stratification is the process of rearranging the data as to ensure each fold is a good     
#        representative of the whole. For example in a binary classification problem where each class 
#        comprises 50% of the data, it is best to arrange the data such that in every fold, each class 
#        comprises around half the instances.

        #Stratification is generally a better scheme, both in terms of bias and variance, when 
#        compared to regular cross-validation.
        #https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation
        
        if strat == True :
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                stratify=y, random_state=random_state)
        else :
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
        
        # Sometimes scaling datas leads to better generalization
        if scale == True :
            X_train_mean =  X_train.mean(axis=0)
            X_std        =  X_train.std(axis=0)
            
            X_train = (X_train  - X_train_mean) /   X_std
            X_test  = (X_test   - X_train_mean) /   X_std  
        
        #We finish by "selfing" Training Set and Testing Set
        self.X_train, self.y_train  =   X_train, y_train
        self.X_test, self.y_test    =   X_test , y_test
###-------------------------------------------------------------------------------
    def build_graph(self):
        ### We use X_train.shape
        w_dict, biases_d = dict(), dict()
        Nraw_Xtrain, Ncol_Xtrain = self.X_train.shape

        self.N_["N0"] = Ncol_Xtrain

        # We construct weight and biases dictionnary as we anticipate tensor dicts
        jj, err = 1, False # We skip N0 
        while jj <= len(self.N_.values()) and err == False : 
            wkey, bkey = "w%d_init" %(jj), "b%d_init" %(jj)

            prev_key = "N%d" %(jj-1)
            curr_key = "N%d" %(jj)
            try :
                w_dict[wkey] = np.random.randn(self.N_[prev_key], self.N_[curr_key]) / np.sqrt(self.N_[prev_key])
                biases_d[bkey] = np.zeros(self.N_[curr_key])
            
            except KeyError :
                w_dict[self.wlastkey] = np.random.randn(self.N_[prev_key], self.N_["K"]) / self.N_[prev_key]
                biases_d[self.blastkey] = np.zeros(self.N_["K"])
                err = True
            jj += 1

        self.w_dict = w_dict
        self.biases_d = biases_d
###-------------------------------------------------------------------------------        
    def tf_variables(self):
        ### We want to create as many tf.tensors as we have weight and biases matrices 
            ### Those ones will be tf.Variables(weight.astype(np.float32))
            
        w_tf_d, b_tf_d = dict(), dict()
        wkeys = [i.split("_")[0] for i in self.w_dict.keys()]
        bkeys = [i.split("_")[0] for i in self.biases_d.keys()]

        for (kw,w, kb,b) in zip(wkeys, self.w_dict.values(), bkeys, self.biases_d.values()) :
            w_tf_d[kw] = tf.Variable(w.astype(np.float32))
            b_tf_d[kb] = tf.Variable(b.astype(np.float32))
        
        ### We also create x and t which are training data and target to train the model
            ### Those will be tf.placeholder 
        self.x = tf.placeholder(tf.float32, (None, self.N_["N0"]))
        self.t = tf.placeholder(tf.float32, (None, self.N_["K"]))
        
        self.w_tf_d = w_tf_d
        self.b_tf_d = b_tf_d
###-------------------------------------------------------------------------------
    def feed_forward(self, activation="relu"):
        Z = dict()
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu"]
        # to have leakyrelu upgrade tensorflow sudo pip install tensorflow==1.4.0-rc0 
        ### A wide variety of models are availables to compute the neuronal outputs.
        ### We chose that way of implementation : the main work is done only if str_act
        ### belongs to a list predefined that should be improved by time
        #####   Must read and cite articles or documentations related to activation fcts
        
        for str_act, act in zip(act_func_lst, [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.leaky_relu] ) :
            if str_act == activation :
                Z["z1"] = act( tf.matmul(self.x,self.w_tf_d["w1"]) + self.b_tf_d["b1"] )    
                print ("act function chose %s \t act function wanted %s" %(str_act, activation))
                for i in xrange(2,len(self.w_tf_d)) : # i %d wi, We dont take wlast
                    curr_zkey, prev_zkey = "z%d" %(i), "z%d" %(i-1)
                    wkey = "w%d" %(i)
                    bkey = "b%d" %(i)
                    Z[curr_zkey] = act( tf.matmul(Z[prev_zkey], self.w_tf_d[wkey] )  + self.b_tf_d[bkey] ) 
                self.y_pred_model = tf.matmul(Z[curr_zkey], self.w_tf_d[self.wlastkey]) + self.b_tf_d[self.blastkey]
                
                
        if Z == {}:
            raise AttributeError ("\"{}\" activation function is unknown.\nActivation function must be one of {}".format(activation, act_func_lst))
        
        ### We constructed operations 
        print("Z's construits")
        self.Z = Z
        
    def error_computation(self, err_eval) :
        ### Now we compute error based on different available models
        ### We just begin with cross_entropy. Specified way of computation may be required 
        ### for particular error functions 
        
        error_lst_fct = ["cross_entropy", "OLS"]
        #####   To be continued.
        #####   Must read and cite articles or documentations related to error func
        for str_err, err_eval in zip(error_lst_fct, [tf.nn.softmax_cross_entropy_with_logits, tf.square]) :
            if str_err == error_lst_fct :
                self.loss = tf.reduce_sum(err_eval(logits=self.y_pred_model, label=self.t))
        print ("Fonction d\'erreur construite (uniquement pour cross_entropy)")
        
    def optimisation(self):
        ## Once again, we must look at different optimizers existing in TF
        print("Optimisation à construire") 
###-------------------------------------------------------------------------------

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
if __name__=="__main__":
    
    TF = Neural_Network(0.0004, model="", pathfile="test.csv", N_hidden_layer=3, layers_sizes = [5, 100, 100, 200, 50])
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    TF.train_and_split(cancer.data, cancer.target)
    TF.build_graph()
    TF.tf_variables()
    TF.feed_forward("relu")
    TF.error_computation("cross_entropy")
    
    TF.optimisation()
    
    
    
