#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os, csv
from sklearn.model_selection import train_test_split

import tensorflow as tf
#import tensorlayer as tl

import NN_class_try as NNC


class Bootstraped_Neural_Network :
    def __init__(self, n_estimators, dataset) :
        
        self.n_estimators = n_estimators
        self.X = dataset["data"]
        self.y = dataset["target"]
        
        self.length_resample = len(self.X) // self.n_estimators
        
        self.bkey = lambda b : "dataset_%s" % str(b)
        
    def resample_dataset(self):
        bootstrap_dataset = dict()
        
        for b in range(self.n_estimators) :
            bootstrap_dataset[self.bkey(b)] = {}
        
        for b in range(self.n_estimators) :
            Xcp = np.copy(self.X)
            ycp = np.copy(self.y)
            
            permute = np.random.permutation(self.length_resample)
            bootstrap_dataset[self.bkey(b)]["data"] = Xcp[permute]
            bootstrap_dataset[self.bkey(b)]["target"] = ycp[permute]
        
        self.bootstrap_dataset = boostrap_dataset
        
    def build_NN(self, lr, dict_layer, scaler="Standard", max_epoch=100, reduce_type="sum", rdn=0 **kwargs) :
        NN_dict = {}
        
        testkey = lambda key : if key not in kwargs.keys()
        
        val = False if testkey["val"] else kwargs["val"]
        n_compo = "mle" if testkey["n_components"] else kwargs["n_components"]
            
        for b in range(self.n_estimators) :
            # Define an NN object
            nn_b = NNC.Neural_Network(lr, N_=dict_layer, max_epoch=max_epoch, reduce_type=reduce_type, **kwargs)
            
            # Getting and Spliting The Data
            X_NN = np.copy(bootstrap_dataset[self.bkey(b)]["data"]
            y_NN = np.copy(bootstrap_dataset[self.bkey(b)]["target"]
            
            nn_b.split_and_scale(X, y, shuffle=False, val=val, n_components=n_compo, random_state=rdn)        
                        
###----------------------------------------------------------------
###----------------------------------------------------------------        
if __name__ == "__main__" :
    from sklearn.datasets import load_boston

    boston = load_boston()
    dataset = {}
    dataset["data"] = boston.data     
    dataset["target"] = boston.target
        
    BNN = Bootstraped_Neural_Network(5, dataset) 
    BNN.resample_dataset()       
        
        
        
