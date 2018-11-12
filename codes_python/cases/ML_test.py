#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from itertools import cycle
from matplotlib.pyplot import cm

from scipy import optimize as op

import numdifftools as nd

import glob
import time

import Class_write_case as cwc

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

NNC = reload(NNC)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#X_line = np.linspace(-np.pi,np.pi,1000)
#f = np.array(map(lambda x : x**2, X_line))
#g = np.gradient(f, X_line)


## Construction de la dataset
#X = np.zeros((4))
#y = np.zeros((1))

#dict_layers = {"I"  : 4,\
#               "N1" : 500,\
#               "N2" : 10,\
###               "N6" : 10,\
##               "N7" : 10,\
##               "N8" : 10,\
#               "O"  : 1}


#for j in range(0, len(X_line)-1) :
#    X = np.block([[X], [X_line[j], X_line[j+1], f[j], f[j+1]]])
#    y = np.block([[y], [g[j]]])
#    
#X, y = np.delete(X, 0, axis=0), np.delete(y, 0, axis=0)

#for i in range (2) :
#    rand_index = np.random.permutation(np.arange(len(y)))
#    XX = X[rand_index]
#    yy = y[rand_index]

#    X = np.block([[X], [XX]])
#    y = np.block([[y], [yy]])
    

def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_, scale=True, step=50, **kwargs) :
    plt.ion()
    print (kwargs)
    print ("X.shape = {}".format(X.shape))
    nn_obj = NNC.Neural_Network(lr, N_, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(X, y, strat=False, shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.def_optimizer(opti)
    nn_obj.layer_stacking_and_act(activation=act)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.case_specification_recap()
    
    kwargs = nn_obj.kwargs
    
#    return nn_obj
    print ("nn_obj.X_train.shape = {}".format(nn_obj.X_train.shape))
    try :
        nn_obj.training_session(tol=1e-7)

    except KeyboardInterrupt :
        print ("Session closed")
        nn_obj.sess.close()

    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    test_line = range(len(nn_obj.X_test))
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in test_line])
    error_estimation = sum(deviation)
    
    fig, ax = plt.subplots(1,2, figsize=(7,7), num="Comparaison sur le test set")
    
    ax[0].set_title("Comparaison Pred(xtest)/ytest")
    ax[0].plot(test_line, beta_test_preds, label="Prediction sur Test set", marker='+',\
    fillstyle='none', linestyle='none', c='r')
    ax[0].plot(test_line, nn_obj.y_test, label="Expected value", marker='o', fillstyle='none', linestyle='none', c='k')   
    ax[0].legend()
#    plt.figure("Evolution de l\'erreur %s" %(loss))
#    plt.plot(range(len(nn_obj.costs)), nn_obj.costs, c='r', marker='o', alpha=0.3,\
#            linestyle="none")
#    error_estimation /= (len(nn_obj.X_test) -1)
    
    ax[1].set_title("Deviation of the prediction")
    ax[1].plot(nn_obj.y_test, nn_obj.y_test, c='k', label="reference line")
    ax[1].plot(nn_obj.y_test, nn_obj.y_test, c='b', marker='+', label="wanted     value",linestyle='none')
    ax[1].plot(nn_obj.y_test, beta_test_preds, c='r', marker='o', linestyle='none',   label="predicted value", ms=3)
    ax[1].legend(loc="best") 

#    print("Modèle utilisant N_dict_layer = {}".format(N_))\\
    print("Modèle pour H_NL = {}, H_NN = {} \n".format(len(N_.keys())-2, N_["N1"]))
    print("Fonction d'activation : {}\n Fonction de cout : {}\n\
    Méthode d'optimisation : {}".format(act, loss, opti))
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()
    plt.pause(0.001)
    return nn_obj
#knn = KNeighborsRegressor(n_neighbors=3)
#y1 = y.reshape(-1, 1)

#X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=0)

#knn.fit(X_train, y_train)

#plt.figure()
#plt.plot(y_test,y_test,label="Expected")
#plt.plot(y_test, knn.predict(X_test), label="Predicted KNN", marker="o", markersize=3, fillstyle="none", linestyle="none")
#plt.legend()

def knnproc(knn, X_line, f, g):
    beta = []
    
    if knn.__dict__['_fit_X'].shape[1] == 4 :
        start = 0
        end = len(f)-1
        inp = lambda j :np.array([[X_line[j], X_line[j+1], f[j], f[j+1]]])
    
    if knn.__dict__['_fit_X'].shape[1] == 6 :
        start = 1
        end = len(f)-1
        inp = lambda j :np.array([[X_line[j-1], X_line[j], X_line[j+1], f[j-1], f[j], f[j+1]]])
    
    for j in range(start, end) :
        xs = inp(j)
        beta.append(knn.predict(xs))
    
    return np.array(beta)[0,0]
#    plt.figure()
#    plt.plot(X_line[0:-1], g[0:-1], label="True der", c='r')
#    plt.plot(X_line[0:-1], beta, marker='o', linestyle='none', fillstyle='none', c='purple', label="Pred")
#    plt.legend()
#    
#    gg = g[1:-1]
#    ecart = []
#    for j in range(0, len(gg)) :    
#        if np.abs(gg[j]) > 1e-8 : 
#            ecart.append((gg[j] - beta[j])/(gg[j]) * 100)
#        else :
#            ecart.append((gg[j] - beta[j])/(1.) * 100)    
#    fig, axes = plt.subplots(1,2,figsize=(10,5))
#    axes[0].hist(ecart) 
#    axes[1].plot(X_line[1:-1], ecart)
#    axes[1].set_xlabel("X")
#    axes[1].set_ylabel("Erreur en pourcentage")
#    plt.show()
    
#########
    
def recentre(xs, X_train_mean, X_train_std):
    """
    This function refocuses the input before prediction. It needs three arguments :
    
    xs              :   The input to refocuses
    X_train_mean    :   The mean vector calculated according to the training set
    X_train_std     :   The standard deviation vector calculated according to the training set   
    
    xs is a vector of shape [N] 
    X_train_mean and X_train_std are vectors of whose shape is [N_features]
    """
    for i in range(np.size(X_train_mean)-1) :
        xs[i] -= X_train_mean[i]
        if np.abs(X_train_std[i]) > 1e-12 :
            xs[i] /= X_train_std[i]

    return np.array([xs])

def proc(nn_obj, X_line, f, g):
    beta = []
    if nn_obj.N_["I"] == 4 :
        start = 0
        end = len(f)-1
        inp = lambda j :np.array([X_line[j], X_line[j+1], f[j], f[j+1]])
    
    if nn_obj.N_["I"] == 6 :
        start = 1
        end = len(f)-1
        inp = lambda j : np.array([X_line[j-1], X_line[j], X_line[j+1], f[j-1], f[j], f[j+1]])
    
    for j in range(start, end) :
        try : xs = inp(j)
        except IndexError : print ("j = %d" % j); print("X_line shape = {}\t f shape = {}".format(X_line.shape, np.shape(f)))
        if nn_obj.scale == True :
            recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
        
        xs = xs.reshape(-1, nn_obj.X.shape[1])
        beta.append(nn_obj.predict(xs)[0,0])
        
#    plt.figure()
#    plt.plot(X_line[0:-1], g[0:-1], label="True der", c='r')
#    plt.plot(X_line[0:-1], beta, marker='o', linestyle='none', fillstyle='none', c='purple', label="Pred")
#    plt.legend()
    
    return beta

def evol_loglog(nn_ml, n_dx, color, label):
    cpt = 0
    Nxlst, err = [], []
    while cpt < n_dx :
        x_line = np.linspace(1.,2, 10*(1 + cpt))
        dx = abs(x_line[5] - x_line[4])
        
        fx = np.array(map(lambda x: x**2, x_line))
        gx = np.array(map(lambda x: 2*x, x_line))
        
        beta = np.array(proc(nn_ml, f=fx, g=gx, X_line=x_line))
        
        if nn_ml.N_["I"] == 4 :
            (s, e) = (0,-1)  
            title = "Evolution error NN 2 points"
        
        if nn_ml.N_["I"] == 6 :
            (s, e) = (1,-1)  
            title = "Evolution error NN 3 points"
            
        Nxlst.append( 1./dx)
        err.append(np.linalg.norm(beta-gx[s:e], np.inf))
        
        cpt += 5

    plt.figure("%s" % title)
    plt.loglog(Nxlst, err, marker='o', color=color, label= label)
        
    return title
    
def knnevol_loglog(knn, n_dx, color, label):
    cpt = 0
    Nxlst, err = [], []
    while cpt < n_dx :
        x_line = np.linspace(1.,2, 10*(1 + cpt))
        dx = abs(x_line[5] - x_line[4])
        
        fx = np.array(map(lambda x: x**2, x_line))
        gx = np.array(map(lambda x: 2*x, x_line))
        
        beta = np.array(knnproc(knn, f=fx, g=gx, X_line=x_line))
        
        spe = knn.__dict__['_fit_X'].shape[1]
        
        if spe == 4 :
            (s, e) = (0,-1)  
            title = "Evolution error KNN 2 points"
        
        if spe == 6 :
            (s, e) = (1,-1)  
            title = "Evolution error KNN 3 points"        
        
        Nxlst.append(1./dx)
        err.append(np.linalg.norm(beta-gx[0:-1], np.inf))
        
        cpt += 5

    plt.figure(title)
    plt.loglog(Nxlst, err, marker='o', color=color, label= label)
        
    return title
#def stacking(nn_obj, X_line=X_line) : +
#    input_knn = nn_obj.predict(nn_obj.X)



############ Notes ############

# Sur le rand_index :

# s3 = np.array([["pol", "lol", "kol"], ["ref", "tef", "yef"], ["frt", "grt","hrt"], ["bvc", "nvc", ";vc"]])
# s3
# array([['pol', 'lol', 'kol'],
#       ['ref', 'tef', 'yef'],
#       ['frt', 'grt', 'hrt'],
#       ['bvc', 'nvc', ';vc']], dtype='|S3')
# rand_index = np.random.permutation(np.arange(len(s3)))

# s4 = s3[rand_index]
# s4 = s4
# array([['ref', 'tef', 'yef'],
#       ['frt', 'grt', 'hrt'],
#       ['pol', 'lol', 'kol'],
#       ['bvc', 'nvc', ';vc']], dtype='|S3')


