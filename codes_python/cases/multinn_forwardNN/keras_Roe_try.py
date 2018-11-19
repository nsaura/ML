#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import sys
import time
import keras 

import argparse
import os.path as osp
import tensorflow as tf

## Import du chemin cases ##
case_folder = osp.abspath(osp.dirname("../"))
sys.path.append(case_folder)

import solvers

from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rfr

K.clear_session()
first_weights = 40

base_path = './../data/xy_files/'
str_case = 'keras_concatenate_case'

conditions= {'L'    :   1,
             'Nt'   :   400,
             'Nx'   :   180,
             'tf'   :   0.2,
             'f'    :   lambda u : u**2,
             'fprime' : lambda u : u,
             'type_init' : "sin",
             'amp'  :   1.
             }

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

def modify_key(conditions, key, new_value):
    if key not in conditions.keys():
        print ("Wrong key. The good keys are :")
        for k in conditions.keys() : print ("%s \n" %(k))
        sys.exit("Key error")
    
    else :
        conditions[key] = new_value
        print ("Done")
        
n_cases = 5
n_phase = 3

line_x = np.linspace(0,1,conditions['Nx'])
szline = line_x[1:-1].size

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

def create_dataset(conditions, base_path=base_path, str_case=str_case, overwrite=False) :
    pathtocheckX = osp.join(base_path , str_case +"X.npy")
    pathtochecky = osp.join(base_path , str_case +"y.npy")
    
    dt = conditions['tf'] / conditions['Nt']
    dx = float(conditions['L']) / (conditions['Nx']-1)
    
    if osp.exists(pathtocheckX) and osp.exists(pathtochecky) and overwrite==False :
        print("%s and %s exist" %(pathtocheckX, pathtochecky))
        
        X = np.load(pathtocheckX)
        y = np.load(pathtochecky)
        
    else :
        if not osp.exists(base_path) :
            os.mkdir(base_path)
        
        X, y = np.zeros((1,4)), np.zeros((1))

        add_block = lambda u, j : [u[j-1], u[j], u[j+1], (u[j+1] - u[j-1])/(2*dx)]
#        plt.figure("Evolution")
        for n in range(n_cases) :
            amp = np.random.choice(np.linspace(0.4, 0.55, 10))

            for a in range(n_phase) :
                p = np.random.choice(np.linspace(-np.pi, np.pi, 200))
                u = amp*np.sin(2*np.pi*(line_x)/(conditions['L']) + p)
                u_next = amp*np.sin(2*np.pi*(line_x)/(conditions['L']) + p)
                
#                plt.clf()
                
                for t in range(conditions['Nt']) :
                    if t != 0 :
                        u = u_next
                        u_next = np.zeros_like(u)
                    for j in range(1, len(line_x)-1) :
                        u_next[j] = solvers.timestep_roe(u, j, dt/dx, conditions['f'],
                                                                      conditions['fprime'])
                    
                    u_next[0] = u_next[-2]
                    u_next[-1] = u_next[2]
                    
#                    plt.plot(line_x[1:-1], u_next[1:-1])
#                    plt.pause(0.4)
#                    if t % 20 == 0 :
#                        plt.clf()
                        
                    for j in range(1, len(line_x)-1) :
                        X = np.block( [ [X], add_block(u, j) ] )
                        y = np.block( [ [y], [u_next[j]] ] )
                    
        X = np.delete(X, 0, axis=0)
        y = np.delete(y, 0, axis=0)
        
        x = []

        for j in range(len(X)//szline) :
            x.append([X[j*szline:(j+1)*szline]])
        
        x = np.array(x)
        
        X = x.reshape(-1, szline, X.shape[1])
        
        np.save(pathtocheckX, X)
        np.save(pathtochecky, y)
        
    return X, y 

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

def scale_and_train_test_split(X, y, train_perc=0.8) :
    X = np.copy(X)
    y = np.copy(y)
    
    random_permutation = np.random.permutation(len(X))
    
    X = X[random_permutation]
    y = y[random_permutation]
    
    train_size = int(train_perc * X.shape[0])
    
    xtr, ytr  = X[:train_size], y[:train_size]
    xte, yte  = X[train_size:], y[train_size:]
    
    xtr_means = xtr.mean(axis=0) # Un moyenne par neurones
    xtr_stdds = xtr.std(axis=0)  # Une deviation standard pour un seul neurone
        
    xtr_scaled = np.zeros_like(xtr)
    xte_scaled = np.zeros_like(xte)
    
    for j in range(xtr_means.shape[0]) :
        for k in range(xtr_means.shape[1]) :
            xtr_scaled[:, j, k] = xtr[:, j, k] - xtr_means[j, k]
            xte_scaled[:, j, k] = xte[:, j, k] - xtr_means[j, k]
            
            if abs(xtr_stdds[j, k]) > 1e-12 :
                xtr_scaled[:, j, k] /= xtr_stdds[j, k]
                xte_scaled[:, j, k] /= xtr_stdds[j, k]
                
#    for i, mean in enumerate(xtr_means) :
#        xtr_scaled[:, i] = xtr[:, i] -  mean
#        xte_scaled[:, i]  = xte[:, i]  -  mean
#        
#        if np.abs(xtr_stdds[i]) > 1e-12 :
#            xtr_scaled[:,i] /= xtr_stdds[i]
#            xte_scaled[:,i] /= xtr_stdds[i]

    return xtr_scaled, ytr, xte_scaled, yte, xtr_means, xtr_stdds

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

def build_NN_conca(X, y, lr, save_plot=True, fit = True) :
    n_features = X.shape[1]
    n_input_layers = np.size(line_x[1:-1])
    
    metrics = ["mean_squared_error", "mean_absolute_error"]

    inputs_dict = {}
    first_layers = {}
    
    #-- Construction du graphe
    for in_layer_n in range(n_input_layers) :
        inputs_dict["layers_n%03d" %(in_layer_n)] = keras.layers.Input(shape=(n_features,))
        first_layers["first_layers_n%03d" %(in_layer_n)] = keras.layers.Dense(1,
                                                         activation = 'selu',
                                                         bias_initializer = 'zeros',
                                                         kernel_initializer = 'random_normal'
                                                         )(inputs_dict["layers_n%03d" %(in_layer_n)])

    unsorted_keys = first_layers.keys()
    unsorted_keys.sort()
    sorted_keys = unsorted_keys 

    # List de toutes les entrees
    input_layers  = [inputs_dict[k.split("_")[1] + "_" + k.split("_")[2]]  for k in sorted_keys]
    
    # List a concatener
    sorted_layers = [first_layers[k] for k in sorted_keys]
    
    # Concatenation    
    scd_layer = keras.layers.concatenate(sorted_layers)

    # une seule couche de sortie
    model = keras.models.Model(input_layers, scd_layer)
    
    if save_plot :
        plot_model(model, to_file='keras_conca_graph.png', show_shapes=True, show_layer_names=False)
    
    #-- 
    
    #-- Optimizer, loss
#    adam = tf.train.AdamOptimizer(learning_rate=lr)    
    sgd = keras.optimizers.nadam(lr=0.002)
    model.compile(optimizer=sgd, loss='mse', metrics=metrics)
    
#    model.summary()
    model.save_weights("./weights.h5", overwrite=True)
    return model

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#def fit_model(model, xtr, ytr, xte, yte, epochs, bsz) :
#    # On doit retravailler xtr car si on ecrit comme a l'accoutumé : mode.fit(xtr, ytr, epochs=150)
#    # On se retrouve avec l'erreur Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 50 array(s), but instead got the following list of 1 arrays:
#    # Il faut specifier a chaque batch l'entree de chacune des couches Inputs 
#        
#    xtr = np.copy(xtr) ; xte = np.copy(xte)
#    ytr = np.copy(ytr) ; yte = np.copy(yte)
#    
#    raw_xtr = xtr.shape[0] 
#    col_xtr = xtr.shape[1]
#    
#    xtr = xtr.reshape(-1, 1, col_xtr) 
#    # xtr.shape : (54000, 1, 4)

#    xtr_lst_array = [xtr[i] for i in range(len(xtr))]
#    ytr_lst_array = [ytr[i] for i in range(len(ytr))]
#    
#    print ("np.shape(xtr_lst_array) : {}".format(np.shape(xtr_lst_array)))
#    
#    final_xtr = []
#    final_ytr = []
#    
##    for i in range(len(xtr) // szline) :
##        final_xtr.append([xtr_lst_array[i*szline : (i+1)*szline]])
##        final_ytr.append([ytr_lst_array[i*szline : (i+1)*szline]])
##        
##    print ("np.shape(final_xtr) = {}".format(np.shape(final_xtr)))
##    
##    xtr_arr_final = np.array(final_xtr)
##    xtr_arr_final = xtr_arr_final.reshape(szline, raw_xtr//szline, col_xtr)
##    
##    ytr_arr_final = np.array(final_ytr)
##    ytr_arr_final = ytr_arr_final.reshape(raw_xtr//szline, szline)
##    
##    print ("np.shape(xtr_arr_final) : {}".format(np.shape(xtr_arr_final)))
##    print ("np.shape(ytr_arr_final) : {}".format(np.shape(ytr_arr_final)))
#    
#    model.load_weights("./weights.h5")
#    model.fit([i for i in xtr_arr_final], ytr_arr_final, epochs = epochs, batch_size=bsz)
#    
#    model.save_weights("weights.h5")    

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

def fit_model(model, xtr, ytr, xte, yte, epochs, bsz) :
    # On doit retravailler xtr car si on ecrit comme a l'accoutumé : mode.fit(xtr, ytr, epochs=150)
    # On se retrouve avec l'erreur Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 50 array(s), but instead got the following list of 1 arrays:
    # Il faut specifier a chaque batch l'entree de chacune des couches Inputs 
        
    xtr = np.copy(xtr) ; xte = np.copy(xte)
    ytr = np.copy(ytr) ; yte = np.copy(yte)
    
    model.fit([x for x in xtr], ytr, epochs = epochs, batch_size=bsz)
    
    model.save_weights("weights.h5")    
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def predict(model, raw_new_input, means, stdd) :
    # Les entrees doivent etre reparties sur l'ensemble des sous couches
    inp = np.copy(raw_new_input)
    
    dx = float(conditions['L']) / (conditions['Nx']-1)
    add_block = lambda u, j : [u[j-1], u[j], u[j+1], (u[j+1] - u[j-1])/(2*dx)]
    
    ninp = np.zeros((4))
    
    for i in range(1, line_x.size-1) :
        ninp = np.block([[ninp], add_block(inp, i)])
    
    ninp = np.delete(ninp, 0, axis=0)
    
    inp_scaled = np.copy(ninp) 

    for i, m in enumerate(means) :
        inp_scaled[:, i] = ninp[:, i] - m
        if np.abs(stdd[i]) > 1e-12 :
            inp_scaled[:, i] /= stdd[i]
        
    return model.predict([i.reshape(1,-1) for i in inp_scaled])

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

if __name__ == '__main__' :
    
    X, y = create_dataset(conditions, overwrite=False)
    xtr, ytr, xte, yte, means, stdds = scale_and_train_test_split(X, y)

#    m, s = scale_and_train_test_split(X, y)

#    mode = build_NN_conca(X, y, 0.0001)
#    fit_model(mode, xtr, ytr, xte, yte, epochs=1500, bsz=128)


#    for i in range(1) :
#        fit_model(mode, xtr, ytr, xte, yte, 1000, 16) 
#        print ("i = %d\n" %i)
#        print ("get_weights[-3:]\n{}".format(mode.get_weights()[-3:]))

#        mode.save_weights("./weights.h5")
#        mode.load_weights("./weights.h5")
        
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
