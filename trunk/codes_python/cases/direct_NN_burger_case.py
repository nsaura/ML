#!/usr/bin/python2.7
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

import numdifftools as nd

import time

from tensorflow import reset_default_graph

import Class_write_case as cwc
## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Vit_Choc as cvc

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

NNC = reload(NNC)
cvc = reload(cvc)

#run Class_Vit_Choc.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10 -typeJ "u"

parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser) 

u_name = cb.u_name

X, y = np.zeros((1)), np.zeros((1))

for it in range(cb.itmax) :
    u_curr = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))
    u_next = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it+1))
    for j in range(1, cb.Nx-1) : #[1, Nx-1]
        
        X = np.block([[X], [u_curr[j]]])
        y = np.block([[y], [u_next[j]]])

X = np.delete(X, 0, axis=0)
y = np.delete(y, 0, axis=0)

x_to_randomize = np.copy(X)
y_to_randomize = np.copy(y)

for i in range(2) :
    XX, yy  = [], []
    
    permute_indices = np.random.permutation(np.arange(len(y_to_randomize)))
    x_random = x_to_randomize[permute_indices]
    y_random = y_to_randomize[permute_indices]
    
    for j in range(len(x_random)) :
        XX.append(x_random[j]*(1 + np.random.rand()))
        yy.append(y_random[j])
    
    XX = np.array(XX)#.reshape(-1,1)
    yy = np.array(yy)
    
    X = np.block([[X], [XX]])
    y = np.block([[y], [yy]])

dict_layers = {"I": 1, "O" :1, "N1":80, "N2":80, "N3":80, "N4":80, "N5":80, "N6":80, "N7":80, "N8":80, "N9":80, "N10":80}#, "N11":40}

def build_direct_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, scaler, N_=dict_layers, step=50, **kwargs) :
    plt.ion()
    print (kwargs)
    
    # Define an NN object
    nn_obj = NNC.Neural_Network(lr, scaler = scaler, N_=N_, max_epoch=max_epoch, reduce_type=reduce_type, **kwargs)
    
    # Spliting The Data
    nn_obj.split_and_scale(X, y, shuffle=True)
    
    #Preparing the Tensorflow Graph
    nn_obj.tf_variables()
    nn_obj.layer_stacking_and_act(act)
    
    #Setting Optimizer and Loss for the graph    
    nn_obj.def_optimizer(opti)
    nn_obj.cost_computation(loss)
    
    #Display a Recap
    nn_obj.case_specification_recap()
    
    kwargs = nn_obj.kwargs
    
#    return nn_obj
    print (nn_obj.X_train.shape)
    try :
        nn_obj.training_phase(tol=1e-3)

    except KeyboardInterrupt :
        print ("Session closed")
        del nn_obj.sess

    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    
    score = nn_obj.score(beta_test_preds, nn_obj.y_test)
    print ("score of this session is : {}".format(score))
    
    test_line = range(len(nn_obj.X_test))
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in test_line])
    error_estimation = sum(deviation)

    dev_lab = "Pred_lr_{}_{}_{}_Maxepoch_{}".format(lr, opti, act, scaler, max_epoch)
    
    if plt.fignum_exists("Comparaison sur le test set") :
        plt.figure("Comparaison sur le test set")
        plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                    fillstyle='none', linestyle='none', c=nn_obj.kwargs["color"])

    else :
        plt.figure("Comparaison sur le test set")
        plt.plot(test_line, nn_obj.y_test, label="Expected value", marker='o', fillstyle='none',\
                    linestyle='none', c='k')   
        plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                    fillstyle='none', linestyle='none', c=nn_obj.kwargs["color"])
 
    plt.legend(loc="best", prop={'size': 7})
    
    if plt.fignum_exists("Deviation of the prediction") :
            plt.figure("Deviation of the prediction")
            plt.plot(nn_obj.y_test, beta_test_preds, c=nn_obj.kwargs["color"], marker='o',\
                     linestyle='none', label=dev_lab, ms=3)
        
    else :
        plt.figure("Deviation of the prediction")
        plt.plot(nn_obj.y_test, nn_obj.y_test, c='k', label="reference line")
        plt.plot(nn_obj.y_test, nn_obj.y_test, c='navy', marker='+', label="wanted value",linestyle='none')
        plt.plot(nn_obj.y_test, beta_test_preds, c=nn_obj.kwargs["color"], marker='o',\
                      linestyle='none', label=dev_lab, ms=3)

    plt.legend(loc="best", prop={'size': 7}) 

#    print("Modèle utilisant N_dict_layer = {}".format(N_))\\
    print("Modèle pour H_NL = {}, H_NN = {} \n".format(len(N_.keys())-2, N_["N1"]))
    print("Fonction d'activation : {}\n Fonction de cout : {}\n\
    Méthode d'optimisation : {}".format(act, loss, opti))
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()

# see burger_case_u_NN for writing something into ols fashion
    
    return nn_obj

def dNN_solver(nn_obj, cb=cb):
    plt.figure()
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, 0))
    
    u_nNext = []
    for it in range(1, cb.itmax) :
        beta = []
        if it > 1 :
            u = np.copy(u_nNext)
            u_nNext = [] 
        
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j]])
            
            xs = nn_obj.scale_inputs(xs)
            xs = xs.reshape(1, -1)

            u_nNext.append(nn_obj.predict(xs)[0,0])
        # u_nNext.shape = 30 
        # use of list type to insert in a second time boundary condition
        
        u_nNext.insert(0, u[-2])
        u_nNext.insert(len(u), u[1])
        
        u_nNext = np.array(u_nNext)
        
        plt.clf()        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it+1))[1:cb.Nx-1], label="True it = %d" %(it+1), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted at it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c=nn_obj.kwargs["color"])
        plt.legend()
        plt.pause(5)
