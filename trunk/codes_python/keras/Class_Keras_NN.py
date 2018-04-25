#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

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

class K_Neural_Network () :
#------------------------------------------------------------------------------- 
    def __init__(self, lr, dict_layers, opti, loss, metrics=['mse','mae'], max_epoch=10, verbose=False, **kwargs):
        self.dict_layers = dict_layers
        self.lr = lr
        self.max_epoch = max_epoch
        
        self.non_uniform_act = True if "non_uniform_act" in kwargs.keys() else False
        self.batched = True if "b_sz" in kwargs.keys() or "batch_sz" in kwargs.keys() else False
        
        try :
            self.color = kwargs["color"]
        except KeyError :
            self.color= 'orchid'

        try :
            self.step = kwargs["step"]
        
        except KeyError :
            self.step = 50    
            
        self.verbose = verbose
        self.kwargs = kwargs
        
        self.model = keras.models.Sequential() #pour l'instant
        
        self.conv_str_to_acti = {"relu" : keras.layers.activations.relu,\
                                 "selu" : keras.layers.activations.selu,\
                                 "linear" : keras.layers.activations.linear\
                                 }
#                   defined as a class "leakyrelu" : keras.layers.advanced_activations.LeakyReLU\
        
        self.conv_str_to_opti = {"RMSprop" : keras.optimizers.RMSprop,\
                                 "Adamax" : keras.optimizers.Adagrad,\
                                 "Nadam" : keras.optimizers.Nadam,\
                                 "Adam" : keras.optimizers.Adam,\
                                 "SGD" : keras.optimizers.SGD\
                                }
        if opti not in self.conv_str_to_opti.keys() :
            raise KeyError("opti selected: {} is not in {}".format(opti, self.conv_str_to_opti.keys()))
                                
        self.conv_str_to_loss = {"mae" : keras.losses.MAE,\
                                 "logcosh": keras.losses.logcosh,\
                                 "mse" : keras.losses.mse\
                                }
        self.opti, self.loss = opti, loss
        
        self.keras_opti = self.conv_str_to_opti[opti]
        self.keras_loss = self.conv_str_to_loss[loss]
        
        self.metrics = metrics
#-------------------------------------------------------------------------------  
    def train_and_split(self, X, y, random_state=0, strat=False, scale=False, shuffle=True):
        if shuffle == True :
            # Inspired by : Sebastian Heinz se :
            # Medium : a simple deep-learning model for stock price prediction using tensorflow
            permute_indices = np.random.permutation(np.arange(len(y)))
            X = X[permute_indices]
            y = y[permute_indices] 

        if strat == True :
            if np.size(np.shape(y)) == 1 :
                xxyys = train_test_split(X, y, stratify=y,\
                        random_state=random_state)
            if np.size(np.shape(y)) == 2 :
                xxyys = train_test_split(X, y, stratify=y.reshape(-1),\
                        random_state=random_state)
        else :
            xxyys = train_test_split(X, y, random_state=random_state)
        
        X_train, X_test = xxyys[0], xxyys[1]
        y_train, y_test = xxyys[2], xxyys[3]
        
        X_train_std  =  X_train.std(axis=0)
        X_train_mean =  X_train.mean(axis=0)
        
        if scale == True :
            X_train_scaled = np.zeros_like(X_train)
            X_test_scaled = np.zeros_like(X_test)
            
            for i in range(X_train_mean.shape[0]) :
                X_train_scaled[:, i] = X_train[:, i] - X_train_mean[i]
                X_test_scaled[:, i] = X_test[:, i] - X_train_mean[i]
                
                if np.abs(X_train_std[i]) > 1e-12 :
                    X_train_scaled[:,i] /= X_train_std[i]
                    X_test_scaled[:,i] /= X_train_std[i]
            
            print ("Scaling done")
            print ("X_train_mean = \n{}\n X_train_std = \n{}".format(X_train_mean, X_train_std))
            
            X_train = X_train_scaled
            X_test = X_test_scaled
        
        self.scale = scale
        self.X_train_mean = X_train_mean
        self.X_train_std  = X_train_std
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
#-------------------------------------------------------------------------------
    def build(self, act=None):
        layers = {}
        
        dict_key = self.dict_layers.keys()
        dict_key.remove("I")
        dict_key.remove("O")
         
        sorted_keys = sorted(dict_key, key = lambda x: int(x[1:]))
        
        layers["I"] = keras.layers.Input(shape=(self.dict_layers["I"],))
        
        if self.non_uniform_act == True :
            
            for j, k in enumerate(sorted_keys) : # k = N1; k = N2 etc
                
                if j == 0 :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation= self.conv_str_to_acti[self.dict_layers[k][1]],\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer='zeros',\
                                           input_dim=self.dict_layers["I"]\
                                          ))
                else :                    
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation= self.conv_str_to_acti[self.dict_layers[k][1]],\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer='zeros',\
                                           name='Dense-%s' %(k)\
                                          ))
            self.model.add(keras.layers.Dense(self.dict_layers["O"][0],\
                                     activation=self.conv_str_to_acti[self.dict_layers["O"][1]],\
                                     kernel_initializer='random_uniform',\
                                     bias_initializer ='zeros',\
                                     name='Output-Layer'\
                                  ))
        else :
            for j, k in enumerate(sorted_keys) : # k = N1; k = N2 etc
                if j == 0 :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           input_dim=self.dict_layers["I"],\
                                           activation=keras.layers.activations.relu,\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer ='zeros'\
                                          ))
                                          
                else :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation=keras.layers.activations.relu,\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer ='zeros',\
                                           name='Dense-%s' %(k)\
                                          ))
               #Output line
            self.model.add(keras.layers.Dense(self.dict_layers["O"][0],\
                                    activation=keras.layers.activations.relu,\
                                    kernel_initializer='random_uniform',\
                                    bias_initializer ='zeros',\
                                    name='Output-Layer'\
                                 ))
        self.layers = layers
#-------------------------------------------------------------------------------
    def summary(self):
        print("-"*20)
        print("Input dictionnary :\n \x1b[1;37;43m{}\x1b[0m".format(self.dict_layers))
        print ("Optimizer choosen: {}".format(self.opti))
        
        print(self.model.summary())
        
        from keras.utils.vis_utils import plot_model
        
        filename, cpt = "model.png", 0
        if osp.exists("./models") == False :
            os.mkdir("./models")
        filename = osp.join("models", filename)
        
        while osp.exists(filename)==True :
            if cpt == 0:
                filename = osp.splitext(filename)[0] + "-%d" %(cpt+1) + ".png"
            else :
                filename = osp.splitext(filename)[0][:filename.index("-")+1] + "%d" %(cpt+1) + ".png"
            cpt +=1
        
        print ("Current Graphe can be seen by taping in a terminal :") 
        print ("\"eog {} \"".format(osp.abspath(filename)))
         
        plot_model(k_nn.model, to_file=filename)
#-------------------------------------------------------------------------------
    def compile_and_fit(self):
        self.summary()
        
        parameters = {}
        for item in self.kwargs.iteritems():
            parameters[item[0]] = item[1]
            
        if self.opti in ["Adam", "Adamax"] :
            optimizer = self.keras_opti(lr = parameters["lr"],\
                                        beta_1 = parameters["beta1"],\
                                        beta_2 = parameters["beta2"],\
                                        decay  = parameters["decay"]\
                                       )
        
        #### Faire de même avec RMS et SGD
        ### Faire des try except pour lancer les commandes par defaut si non precise dans le kwargs 
        
        self.model.compile(loss=self.keras_loss, optimizer=self.keras_opti, metrics=self.metrics)
        
        bsz = self.kwargs["batch_size"] if self.batched == True else len(self.X_train)
        
        fit = self.model.fit(self.X_train, self.y_train,\
                             epochs=self.max_epoch, batch_size=bsz, metrics=self.metrics)
        
        lenm = len(self.metrics)
        color = iter(cm.magma_r(lenm))
        
        for i in self.metrics :
            c = next(color)
            plt.plot(fit.history['%s' % i], label="%s" % i)
        
        plt.legend()
#-------------------------------------------------------------------------------
if __name__ == '__main__' : 
    from sklearn.datasets import load_boston
    X, y = load_boston().data, load_boston().target
    dict_layers = {"I" : X.shape[1],\
                   "N1" : [100,"relu"],\
                   "N2" : [100,"selu"],\
                   "N3" : [100,"relu"],\
                   "N4" : [100,"linear"],\
                   "N5" : [100,"selu"],\
                   "N6" : [100,"selu"],\
                   "O"  : [1, "relu"]\
                  }
                   
    k_nn = K_Neural_Network(0.02, dict_layers, opti="Nadam", loss="mae", max_epoch=10, verbose=False, non_uniform_act=False) 

    k_nn.train_and_split(X,y)
    k_nn.build()
