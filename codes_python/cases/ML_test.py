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
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
NNC = reload(NNC)

X_line = np.linspace(0,np.pi,1000)
f = map(lambda x : np.sin(x), X_line)

g = np.gradient(f, X_line)

# Construction de la dataset
E = np.zeros((3))
B = np.zeros((1))

for j in range(1, len(X_line)-1) :
    E = np.block([[E], [f[j-1], f[j], f[j+1]]])
    B = np.block([[B], [g[j]]])

E, B = np.delete(E, 0, axis=0), np.delete(B, 0, axis=0)

for i in range (5) :
    rand_index = np.random.choice(np.shape(E)[0], size=998)

    EE = E[rand_index]
    BB = B[rand_index]

    E = np.block([[E], [EE]])
    B = np.block([[B], [BB]])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

knn = KNeighborsRegressor(n_neighbors=3)
y1 = B.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(E,y1, random_state=0)

knn.fit(X_train, y_train)

plt.figure()
plt.plot(y_test,y_test,label="Expected")
plt.plot(y_test, knn.predict(X_test), label="Predicted KNN", marker="o", markersize=3, fillstyle="none", linestyle="none")

def knnproc(knn, X_line=X_line, f=f, g=g):
    beta = []
    for j in range(1, len(X_line)-1) :
        xs = np.array([[f[j-1], f[j], f[j+1]]])

        beta.append(knn.predict(xs)[0])
        
    plt.figure()
    plt.plot(X_line[1:-1], g[1:-1], label="True der", c='r')
    plt.plot(X_line[1:-1], beta, marker='o', linestyle='none', fillstyle='none', c='purple', label="Pred")
    plt.legend()
    
    gg = g[1:-1]
    ecart = []
    for j in range(0, len(gg)) :    
        if np.abs(gg[j]) > 1e-8 : 
            ecart.append((gg[j] - beta[j])/(gg[j]) * 100)
        else :
            ecart.append((gg[j] - beta[j])/(1.) * 100)    
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].hist(ecart) 
    axes[1].plot(X_line[1:-1], ecart)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Erreur en pourcentage")
    plt.show()
    
dict_layers = {"I"  : 3,\
               "N1" : 50,\
#               "N2" : 100,\
#               "N3" : 100,\
#               "N4" : 50,\
#               "N5" : 50,\
##               "N6" : 10,\
#               "N7" : 10,\
#               "N8" : 10,\
               "O"  : 1}
               
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

def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=True, step=50, **kwargs) :
    plt.ion()
    print kwargs
    nn_obj = NNC.Neural_Network(lr, N_=dict_layers, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.def_training(opti)
    nn_obj.feed_forward(activation=act)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.def_optimization()
    
    kwargs = nn_obj.kwargs
    
#    return nn_obj
    print nn_obj.X_train.shape
    try :
        nn_obj.training_session(tol=1e-7, verbose=True)

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

    plt.figure("Comaparaison sur le test set")
    plt.plot(test_line, beta_test_preds, label="Prediction sur Test set", marker='+',\
    fillstyle='none', linestyle='none', c='r')
    plt.plot(test_line, nn_obj.y_test, label="Expected value", marker='o', fillstyle='none', linestyle='none', c='k')   
    plt.legend()
#    plt.figure("Evolution de l\'erreur %s" %(loss))
#    plt.plot(range(len(nn_obj.costs)), nn_obj.costs, c='r', marker='o', alpha=0.3,\
#            linestyle="none")
#    error_estimation /= (len(nn_obj.X_test) -1)
    
    plt.figure("Deviation of the prediction")
    plt.plot(nn_obj.y_test, nn_obj.y_test, c='k', label="reference line")
    plt.plot(nn_obj.y_test, nn_obj.y_test, c='b', marker='+', label="wanted     value",linestyle='none')
    plt.plot(nn_obj.y_test, beta_test_preds, c='r', marker='o', linestyle='none',   label="predicted value", ms=3)
    plt.legend(loc="best") 

#    print("Modèle utilisant N_dict_layer = {}".format(N_))\\
    print("Modèle pour H_NL = {}, H_NN = {} \n".format(len(N_.keys())-2, N_["N1"]))
    print("Fonction d'activation : {}\n Fonction de cout : {}\n\
    Méthode d'optimisation : {}".format(act, loss, opti))
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()
    
#    lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=True, step=50, **kwargs
    
    return nn_obj

def proc(nn_obj, X_line=X_line, f=f, g=g):
    beta = []
    for j in range(1, len(X_line)-1) :
        xs = np.array([f[j-1], f[j], f[j+1]])
        if nn_obj.scale == True :
            recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
        
        xs = xs.reshape(-1, nn_obj.X.shape[1])
        beta.append(nn_obj.predict(xs)[0,0])
        
    plt.figure()
    plt.plot(X_line[1:-1], g[1:-1], label="True der", c='r')
    plt.plot(X_line[1:-1], beta, marker='o', linestyle='none', fillstyle='none', c='purple', label="Pred")
    plt.legend()
