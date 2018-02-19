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

import Gaussian_Process_class as GPC
import class_functions_aux as cfa
import Class_Temp_Cst as ctc
import NN_class_try as NNC

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)

parser = cfa.parser()

plt.ion()

# On déclare un objet de la classe T pour pouvoir avoir accès à des grandeurs propres à la méthode.
# On n'a cependant pas besoin de relancer l'inférence
# Puisque les distributions importantes ont été écrites
T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X,y,v = GPC.training_set(T, parser.N_sample)

dict_layers = {"I" : 2,\
               "N1" : 1000,\
               "N2" : 500,\
               "N3" : 100,\
               "N4" : 100,\
               "N5" : 100,\
               "N6" : 100,\
               "N7" : 100,\
               "N8" : 100,\
               "N9" : 100,\
               "N10": 100,\
               "O"  : 1}
               
#dict_layers = {"I" : 2,\
#               "N1" : 1000,\
#               "N2" : 100,\
#               "N3" : 1000,\
#               "N4" : 100,\
#               "N5" : 1000,\
#               "N6" : 100,\
#               "N7" : 1000,\
#               "O"  : 1}               
N_hidden_layer = len(dict_layers.keys()) - 1
#nn = NNC.Neural_Network(parser.lr, N_=dict_layers, max_epoch=parser.N_epoch)
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
def build_case(lr, X, y, act, opti, loss, N_=dict_layers, max_epoch=parser.N_epoch, **kwargs) :
    # build_case(1e-3, X, y , act="relu", opti="RMS", loss="OLS", decay=0.7, momentum=0.8, max_epoch=1000) marche très bien avec [10, 15, 20, 25, 30, 35, 40, 45, 50]

    nn_obj = NNC.Neural_Network(lr, N_=dict_layers, max_epoch=max_epoch)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True)
#       nn.X_train, nn.X_test
#       nn.y_train, nn.y_test
    nn_obj.tf_variables()
#       nn.w_tf_d 
#       nn.w_tf_d
    nn_obj.feed_forward(activation=act)
#       nn.Z
#nn.def_training("RMS", decay=0.99, momentum=0.9)
#    dico = dict()
#    for k in kwargs :
#        dico[item[0]] = item[1]
    nn_obj.def_training(opti, **kwargs)
#       nn.train_op
    nn_obj.cost_computation(loss)
    nn_obj.def_optimization()
#       nn.minimize_loss
    nn_obj.training_session(batched=False)

    plt.figure("Evolution de l\'erreur %s" %(loss))
    plt.plot(range(len(nn_obj.costs)), nn_obj.costs, c='r', marker='o', alpha=0.3,\
            linestyle="none")
    
    return nn_obj
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
def grid_search(dict_layers):
    lr = 1e-5
    last_costs = dict()
    cost, c_max, cpt, cpt_max = 100000, 100, 0, 5
    while (cost > c_max or np.isnan(cost)==True) and cpt < cpt_max:
        print("Cas {}, lr = {}".format(cpt+1, lr))
        nn = NNC.Neural_Network(lr, N_=dict_layers,\
            max_epoch=parser.N_epoch)
        try :
            build_case(nn,X,y,act="relu", opti="GD", loss="OLS")
            cost = nn.costs[-1]
        except IOError :
            cost = np.nan

        last_costs["lr_{}_cpt_{}".format(lr, cpt)] = cost
        lr /= 10
        cpt +=1
    return last_costs
#-------------------------------------------#
#-------------------------------------------#
def T_to_beta_NN(T, nn_obj, T_inf, body):
    T_n = np.asarray(map(lambda x : -4*T_inf[(T.N_discr-2)/2]*x*(x-1), T.line_z) )
    beta= []    
        
    for j,t in enumerate(T_n) :
        x_s = np.array([[T_inf[j], t]]) ## T_inf, T(z)
        beta.append(nn_obj.sess.run(nn_obj.y_pred_model, feed_dict={nn_obj.x: x_s})[0,0])
        
    beta_n = np.asarray(beta)
    
    B_n = np.zeros((T.N_discr-2))
    T_n_tmp = np.zeros((T.N_discr-2))

    tol, compteur, cmax = 1e-4, 0, 1000 
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
            beta.append(nn_obj.sess.run(nn_obj.y_pred_model, feed_dict={nn_obj.x: x_s}))
        
        beta_nNext = np.asarray(beta)
#        print ("Iteration {}".format(compteur))
#        print ("beta.shape = {}".format(np.shape(beta)))
#        print ("beta_nNext.shape = {}".format(np.shape(beta_nNext)))        
#        
#        print("T_curr shape ={}".format(T_nNext.shape))
        
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
    
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Iterations = {} ".format(compteur))
    
    return T_nNext, beta_nNext
#-------------------------------------------#
#-------------------------------------------#
def solver_NN(T, nn_obj, N_sample, T_inf, body) :
    T_inf_lambda = T_inf
    T_inf = map(T_inf, T.line_z) 

    T_ML, beta_ML = T_to_beta_NN(T, nn_obj, T_inf, body)

    T_true = GPC.True_Temp(T, T_inf, body)

    n = T.N_discr-2
    T_true = T_true.reshape(n)
    T_ML = T_ML.reshape(n)
    T_base = GPC.beta_to_T(T, T.beta_prior, T_inf, body+"_base")
    #    T_nmNext= T_nmNext.reshape(n)
    #    T_nMNext= T_nMNext.reshape(n)

    plt.figure("Beta_True vs Beta_ML; N_sample = {}; T_inf = {}".format(N_sample, body)) 
    plt.plot(T.line_z, T_true, label="True T_field for T_inf={}".format(body), c='k', linestyle='--')
    plt.plot(T.line_z, T_ML, label="ML T_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
    plt.plot(T.line_z, T_base, label="Base solution", c='green')
    plt.legend(loc='best')
    
    return T_true, T_ML

## Les différents attibuts que les prochaines méthodes vont créer seront utiliser après ouverture d'une session

#print("L\'arbre est construit. Selon le dictionnaire N_: \n{}\n\
#Les matrices sont a priori bien dimensionnées, on vérifie avec les différents poids pour chaque couches: \n{}".format(nn.N_, nn.w_tf_d))
lambda_list = [lambda z: 28, lambda z: 55, lambda z: 15+5*np.cos(np.pi*z)]
body_list = ["28", "55", "15+5cos(piz)"]
    
def repeat(T, nn_obj, N_sample, lambda_lst, body_lst) :
    for T_inf, body in zip(lambda_lst, body_lst) :
        solver_NN(T, nn_obj, N_sample, T_inf, body)
    
    plt.show()
#nn.error_computation(err_

     


