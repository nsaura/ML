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
        """
        On veut construire un réseau de neurones assez rapidement pour plusieurs cas.
        Cette classe prend les arguments suivants :
        Args:
        ------
        lr : float  --  Learning Rate (voir notes pour calibration)
        N_ : dict   --  Contient les noms des différentes couches.
                    Doit les première et dernière couches "I" and "O" resp. 
                    Les autres couches sont N1,...,Nn
        max_epoch : int --  Pour le learning, on impose une fin aux aller-retours du BackPropagation Algo.
                    Les poids seront peut être optimaux.
        kwargs : dict --    Autres arguments qui peuvent être utiles dans certains cas
        """
        inputs = dict()
        for kway in kwargs :
            inputs[kway] = kwargs[kway]
        ## We check if the file indexed by "pathfile" variable exists
        ## Otherwise, we create it
        N_test = (N_ == "{}")
#        if N_test == False :
#            
#            f = open(pathfile, "w")
#            for item in N_.interitems() :
#                f.write("%s=%d\n" %(item[0], item[1]))
#            f.close()
        
        if N_test == True :
            raise Exception("Merci de fournir un dictionnaire N_ de la forme : première clé \" \"I\" : N_features\"\
                            Puis les hidden layers sous la forme (pour la j-ème ) \" \"Nj\" : 100\"\
                            Enfin il faudra rajouter l'entrée \"\"O\" : 1 \"\n")

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

        self.N_["I"] = Ncol_Xtrain

        # We construct weight and biases dictionnary as we anticipate tensor dicts
        jj, err = 1, False # We skip N0 
        while jj <= len(self.N_.values()) and err == False : 
            wkey, bkey = "w%d_init" %(jj), "b%d_init" %(jj)

            prev_key = "N%d" %(jj-1) if jj != 1 else "I" 
            curr_key = "N%d" %(jj)
            try :
                w_dict[wkey] = np.random.randn(self.N_[prev_key],\
                        self.N_[curr_key]) / np.sqrt(self.N_[prev_key])
                biases_d[bkey] = np.zeros(self.N_[curr_key])
            
            except KeyError :
                w_dict[self.wlastkey] = np.random.randn(self.N_[prev_key],\
                        self.N_["O"]) / self.N_[prev_key]
                biases_d[self.blastkey] = np.zeros(self.N_["O"])
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
        self.x = tf.placeholder(tf.float32, (None, self.N_["I"]))
        self.t = tf.placeholder(tf.float32, (None, self.N_["O"]))
        
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
        
        # cross entropy most likely to be used in classification wit sigmoid
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
        print("Z\'s construits")
        self.Z = Z
        
    def error_computation(self, err_eval) :
        ### Now we compute error based on different available models
        ### We just begin with cross_entropy. Specified way of computation may be required 
        ### for particular error functions 
        
        error_lst_fct = ["cross_entropy", "OLS"]
        #####   To be continued.
        #####   Must read and cite articles or documentations related to error func
#        cross entropy most likely to be used in classification wit sigmoid
        ## For classical regression problems, we may use L2 regularisation -> OLS
        ## We can also use L1 regression : tf.reduce_sum(tf.abs(y_pred - targets))
        
        for str_err, err_model in zip(error_lst_fct, [tf.nn.softmax_cross_entropy_with_logits, tf.square]) :
            if str_err == error_lst_fct :
                self.loss = tf.reduce_sum(err_model(logits=self.y_pred_model, label=self.t))
        print ("Fonction d\'erreur construite (uniquement pour cross_entropy)")
        
    def optimisation(self):
        ## Once again, we must look at different optimizers existing in TF
        print("Optimisation à construire") 
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
###-------------------------------------------------------------------------------

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
if __name__=="__main__":
    
    TF = Neural_Network(0.0004, model="", pathfile="test.csv", N_hidden_layer=3, layers_sizes = [2, 100, 100, 200, 1])
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    TF.train_and_split(cancer.data, cancer.target)
    TF.build_graph()
    TF.tf_variables()
    TF.feed_forward("relu")
    TF.error_computation("cross_entropy")
    
    TF.optimisation()
    

# Exemple OLS + L1 regression :
# From https://stackoverflow.com/questions/36706379/how-to-exactly-add-l1-regularisation-to-tensorflow-error-function

#import tensorflow as tf

#total_loss = meansq #or other loss calcuation
#l1_regularizer = tf.contrib.layers.l1_regularizer(
#   scale=0.005, scope=None)
#   
#weights = tf.trainable_variables() # all vars of your graph
#regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

#regularized_loss = total_loss + regularization_penalty # this loss needs to be minimized
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(regularized_loss)


