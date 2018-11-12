#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import tensorflow as tf
from sklearn.model_selection import train_test_split

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Temp_Cst as ctc
import class_functions_aux as cfa
import Gaussian_Process_class as GPC

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)

parser = cfa.parser()

#run ELM.py -T_inf_lst 10 15 20 -N_sample 2

plt.ion()

#def tf_pinv(input_) :
#    return tf.py_func(np.linalg.pinv, [input_], tf.float32)

def shuffle_train_split(T, X, y, scale = False): 
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

def mean_std_new_input(x_s, train_mean, train_std):
    for i in range(len(x_s)) :
        x_s[i] -= train_mean[i]
        if np.abs(train_std[i]) > 1e-12:
            x_s[i] /= train_std[i]
    return x_s     

T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X,y,v,m,s = GPC.training_set(T, parser.N_sample)
X_train, X_test, y_train, y_test, m,s = shuffle_train_split(T, X, y, False)

from hpelm import ELM

elm = ELM(X_train.shape[1], y_train.shape[1])
elm.add_neurons(1000, "rbf_l2")
elm.train(X_train, y_train, "LOO")

def T_to_beta_elm(T, elm, T_inf, body, m, s, scale=True):
    T_n = np.asarray(map(lambda x : -4*T_inf[(T.N_discr-2)/2]*x*(x-1), T.line_z) )
    beta= []    
        
    for j,t in enumerate(T_n) :
        x_s = np.array([[T_inf[j], t]]) ## T_inf, T(z)
#        x_s = mean_std_new_input(x_s, m, s)
        beta.append(elm.predict(x_s)[0,0])
    print ("Premier beta = {}".format(beta))
        
    beta_n = np.asarray(beta)
    
    B_n = np.zeros((T.N_discr-2))
    T_n_tmp = np.zeros((T.N_discr-2))

    tol, compteur, cmax = 1e-6, 0, 10000 
    err = err_beta = 1.
    
    while (np.abs(err) > tol) and (compteur <= cmax) :
        if compteur > 0 :
            beta_n = beta_nNext
            T_n = T_nNext
        compteur +=1 
            
        T_n_tmp = np.dot(T.A2, T_n)
        
        for i in range(T.N_discr-2) :
            B_n[i] = T_n_tmp[i] + T.dt*(beta_n[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
          
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        
        beta, sigma = [], []
        for j,t in enumerate(T_n) :
            x_s = np.array([[T_inf[j], t]]) ## T_inf, T(z)
            if scale == True :
#                print("Rescale")
                x_s = mean_std_new_input(x_s, m, s ) # recentre par rapport à la moyenne en cours
            beta.append(elm.predict(x_s)[0,0])

        beta_nNext = np.asarray(beta)
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
    
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Iterations = {} ".format(compteur))
    
    return T_nNext, beta_nNext

def solver_elm(T, elm, N_sample, T_inf, body, m, s, verbose = False) :
    T_inf_lambda = T_inf
    T_inf = map(T_inf, T.line_z) 

    T_ELM, beta_ELM = T_to_beta_elm(T, elm, T_inf, body, m,s, True)
    T_true = GPC.True_Temp(T, T_inf, body)

    n = T.N_discr-2
    T_true = T_true.reshape(n)
    T_ELM = T_ELM.reshape(n)
    T_base = GPC.beta_to_T(T, T.beta_prior, T_inf, body+"_base")
    
    true_beta = GPC.True_Beta(T, T_true, T_inf)
    
    if verbose == True :
        plt.figure("T_True vs T_ELM; N_sample = {}; T_inf = {}".format(N_sample, body)) 
        plt.plot(T.line_z, T_true, label="True T_field for T_inf={}".format(body), c='k', linestyle='--')
        plt.plot(T.line_z, T_ELM, label="ELM T_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
        plt.plot(T.line_z, T_base, label="Base solution", c='green')
        plt.legend(loc='best')

        title = osp.join(osp.abspath("./res_all_T_inf"),"T_True_vs_T_ELM_N_sample_{}_T_inf_{}".format(N_sample, body))

        plt.savefig(title)

        plt.figure("Beta_True vs Beta_ELM; N_sample = {}; T_inf = {}".format(N_sample, body)) 
        plt.plot(T.line_z, true_beta, label="True Beta_field for T_inf={}".format(body), c='k', linestyle='--')
        plt.plot(T.line_z, beta_ELM, label="ELM Beta_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
        plt.plot(T.line_z, T.beta_prior, label="Base solution", c='green')
        plt.legend(loc='best')

        title = osp.join(osp.abspath("./res_all_T_inf"),"Beta_True_vs_Beta_ELM_N_sample_{}_T_inf_{}".format(N_sample, body))

        plt.savefig(title)
        
    ELM_out = dict()
    ELM_out["ELM_T"]  = T_ELM
    ELM_out["ELM_beta"] = beta_ELM.reshape(n)
    
    return ELM_out
