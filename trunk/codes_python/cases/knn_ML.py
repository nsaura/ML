#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import os
import sys
import os.path as osp
import tensorflow as tf
## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import Gaussian_Process_class as GPC
import class_functions_aux as cfa
import Class_Temp_Cst as ctc
import NN_class_try as NNC
import NN_inference_ML as NNI

import time

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)
NNI = reload(NNI)

parser = cfa.parser()
T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X,y,v,m,s = GPC.training_set(T, parser.N_sample)

def shuffle_train_split(T, X, y, scale = True): 
    permute_indices = np.random.permutation(np.arange(len(y)))
    X = X[permute_indices]
    y = y[permute_indices] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    X_train_mean =  X_train.mean(axis=0)
    X_std        =  X_train.std(axis=0)
    
    if scale == True :
        if np.abs(X_std[0]) < 1e-12:
            X_train[:,0] = (X_train[:,0]  - X_train_mean[0])
            X_test[:,0]  = (X_test[:,0]   - X_train_mean[0])
        else :    
            X_train[:,0] = (X_train[:,0]  - X_train_mean[0]) /   X_std[0]
            X_test[:,0]  = (X_test[:,0]   - X_train_mean[0]) /   X_std[0]  

        X_train[:,1] = (X_train[:,1]  - X_train_mean[1]) /   X_std[1]
        X_test[:,1]  = (X_test[:,1]   - X_train_mean[1]) /   X_std[1]

    return X_train, X_test, y_train, y_test, X_train_mean, X_std

def recentre(x_s, X_train_mean, X_train_std):
    x_s[0] -= X_train_mean[0]
    if np.abs(X_train_std[0]) > 1e-12 : 
        x_s[0] /= X_train_std[0] 
    
    x_s[1] -= X_train_mean[1]
    x_s[1] = x_s[1] / X_train_std[1]
    
    return x_s
        
def T_to_beta(T, reg, mean, std, T_inf, body, scale = True) :
    T_inf = map(T_inf, T.line_z)
    T_n = np.asarray(map(lambda x : -4*T_inf[(T.N_discr-2)/2]*x*(x-1), T.line_z) )
    beta =  []
    
    for j,t in enumerate(T_n) :
        x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
        if scale == True :
            x_s = recentre(x_s, mean, std)
        res = reg.predict(x_s.reshape(1,-1))
        print res
        beta.append(res[0][0])

    beta_n = np.asarray(beta)
    B_n = np.zeros((T.N_discr-2))
    T_n_tmp = np.zeros((T.N_discr-2))

    tol, compteur, cmax = 1e-6, 0, 6000
    err = err_beta = 1.
    
    while (np.abs(err) > tol) and (compteur <= cmax) :
        if compteur > 0 :
            beta_n = beta_nNext
            T_n = T_nNext
        compteur +=1 
            
        T_n_tmp = np.dot(T.A2, T_n)
        
        for i in range(T.N_discr-2) :
            B_n[i]      = T_n_tmp[i] + T.dt*(beta_n[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
          
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        
        beta= []
        for j,t in enumerate(T_n) :
            x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
            if scale == True :
                x_s = recentre(x_s, mean, std)
            res =  reg.predict(x_s.reshape(1,-1))
            beta.append(res[0][0])
        
        beta_nNext = np.asarray(beta)
        if compteur % 20 == 0 :
            print("État : cpt = {}, err = {}, err_beta = {}".format(compteur, err, err_beta))
            
#        print ("Iteration {}".format(compteur))
#        print ("beta.shape = {}".format(np.shape(beta)))
#        print ("beta_nNext.shape = {}".format(np.shape(beta_nNext)))        
#        
#        print("T_curr shape ={}".format(T_nNext.shape))
        
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
        err_beta = np.linalg.norm(beta_nNext - beta_n, 2)    
    
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Erreur sur beta = {} ".format(err_beta))    
    print ("Iterations = {} ".format(compteur))
    
    return T_nNext, beta_nNext

def processing(T, N, T_inf, body, scale) :
#    run knn_ML.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -N_sample 10

    from sklearn.model_selection import GridSearchCV
    
    X_train, X_test, y_train, y_test, mean, std = shuffle_train_split(T,X,y,scale=scale)
    reg = KNeighborsRegressor(n_neighbors=N).fit(X_train, y_train)

    grid_search = GridSearchCV(SVC(), param_grid, cv=5) # Objet a entrainer et evaluer

    print("Test differences entre prediction et Y_test : ")
    print("{}".format(y_test - reg.predict(X_test)))
    
    T_ML, beta_ML = T_to_beta(T, reg, mean, std, T_inf, body, scale)
    T_true = GPC.True_Temp(T ,map(T_inf, T.line_z), body)
    true_beta = GPC.True_Beta(T, T_true, map(T_inf, T.line_z))

    plt.ion()
    plt.figure("KNN vs True")
    plt.plot(T.line_z, T_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
    plt.plot(T.line_z, T_true, label='True', linestyle='--', c='k')
    plt.legend()

    plt.figure("beta KNN vs True")
    plt.plot(T.line_z, beta_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
    plt.plot(T.line_z, true_beta, label='True', linestyle='--', c='k')
    plt.legend()




