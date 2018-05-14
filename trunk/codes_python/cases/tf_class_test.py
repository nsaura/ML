#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

import sklearn.datasets as sdata

import tensorflow as tf

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)
import NN_class_try as NNC
NNC = reload(NNC)
## -----
## -----
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
## -----
def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_, scale=True, step=50, **kwargs) :
    plt.ion()
    print kwargs
    nn_obj = NNC.Neural_Network(lr, N_, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.def_training(opti)
    nn_obj.feed_forward(activation=act)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.def_optimization()
    
    kwargs = nn_obj.kwargs
    
    return nn_obj
    print nn_obj.X_train.shape
    try :
        nn_obj.training_session(tol=1e-8, verbose=True)

    except KeyboardInterrupt :
        print ("Session closed")
        nn_obj.sess.close()

    error_estimation /= (len(nn_obj.X_test) -1)
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    return nn_obj
## -----
def knnproc(n, X_train, X_test, X_val, y_train, y_test, y_val, name):
    knn = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train.reshape(-1))
    
    print ("Score pour %s = %f" %(name, knn.score(X_test, y_test)))
    
    y_pred_test =   knn.predict(X_test)
    y_pred_val  =   knn.predict(X_val)
    
    plt.figure("Comparaison prediction/Vraie valeure %s" % name)
#    plt.plot(y_test, y_test, label="True der", c='purple', marker="o", fillstyle='none', linestyle='none')
    plt.plot(y_test, y_pred_test, marker='o', c='darkgreen', label="Pred", linestyle='none')
    plt.legend()
    
    plt.figure("Validation %s" % name)
    plt.plot(y_val, y_val, label="True der", c='k')
    plt.plot(y_val, y_pred_val, marker='o', c='darkgreen', label="Pred KNN", linestyle="none")
    plt.legend()
    
    ecart_test = []
    ecart_val  = []
    for j in range(X_test.shape[0]):
        ecart_test.append((y_test[j] - y_pred_test[j])/(y_test[j] + 1e-8) * 100)
    
    for j in range(X_val.shape[0]):
        ecart_val.append((y_val[j] - y_pred_val[j])/(y_val[j] + 1e-8) * 100)
    
    fig, axes = plt.subplots(2,2,figsize=(10,5), num="KNN Hist Pred (gauche) Val (droite)")
    
    for ax, ecart in zip(axes, [ecart_test, ecart_val]) :
        ax[0].hist(ecart) 
        ax[1].plot(range(len(ecart)), ecart)
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Erreur en pourcentage")
        ax[1].yaxis.set_label_position("right")
    plt.show()
## -----
def rfproc(n_estimators, max_depth, X_train, X_test, X_val, y_train, y_test, y_val, name) :
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth).fit(X_train, y_train.reshape(-1))
    
    print ("Score pour %s = %f" %(name, rf.score(X_test, y_test)))
    
    y_pred_test =   rf.predict(X_test)
    y_pred_val  =   rf.predict(X_val)
    
    plt.figure("Comparaison prediction/Vraie valeure %s" % name)
#    plt.plot(y_test, y_test, label="True der", c='purple', marker="o", fillstyle='none', linestyle='none')
    plt.plot(y_test, y_pred_test, marker='o', c='dodgerblue', label="Pred RF", linestyle='none')
    plt.legend()
    
    
    plt.figure("Validation %s" % name)
    plt.plot(y_val, y_val, label="True der", c='k')
    plt.plot(y_val, y_pred_val, marker='o', c='dodgerblue', label="Pred", linestyle="none")
    plt.legend()
    
    ecart_test = []
    ecart_val  = []
    for j in range(X_test.shape[0]):
        ecart_test.append((y_test[j] - y_pred_test[j])/(y_test[j] + 1e-8) * 100)
    
    for j in range(X_val.shape[0]):
        ecart_val.append((y_val[j] - y_pred_val[j])/(y_val[j] + 1e-8) * 100)
    
    fig, axes = plt.subplots(2,2,figsize=(10,5), num="RF Hist Pred (gauche) Val (droite)")
    
    for ax, ecart in zip(axes, [ecart_test, ecart_val]) :
        ax[0].hist(ecart) 
        ax[1].plot(range(len(ecart)), ecart)
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Erreur en pourcentage")
        ax[1].yaxis.set_label_position("right")
    plt.show()
## -----
## -----
boston = sdata.load_boston()
diabetes = sdata.load_diabetes()
mf1 = sdata.make_friedman1(n_samples=2500)
mf2 = sdata.make_friedman2(n_samples=2500)

datas =\
[   #[boston.data, boston.target, "boston"],
    [diabetes.data, diabetes.target, "diabetes"],
#    [mf1[0], mf1[1], "friedman1"],
#    [mf2[0], mf2[1], "friedman2"],
]

gen_dict = lambda inputsize : \
               {"I"  : inputsize,\
               "N1" : 50,\
               "N2" : 50,\
               "N3" : 50,\
               "N4" : 50,\
               "N5" : 50,\
               "N6" : 50,\
               "N7" : 50,\
               "N8" : 50,\
               "N9" : 50,\
               "N10" : 50,\
               "N11" : 50,\
               "N12" : 50,\
               "N13" : 50,\
               "N14" : 50,\
               "N15" : 50,\
               "N16" : 50,\
               "N17" : 50,\
               "N18" : 50,\
               "N19" : 50,\
               "N20" : 50,\
               "O"  : 1\
               }

for j, (X, y, name) in enumerate(datas) :
    dict_layers = gen_dict(X.shape[1])
    col = NNC.cm.magma(np.linspace(0,1,15))
    color = col[int(np.random.choice(range(len(col)-1)))]
    
    nn_obj = build_case(1e-3, X, y , act="selu", opti="Adam", loss="OLS", max_epoch=3000, reduce_type="sum", verbose=True, N_=dict_layers, color=color, scale=True, BN=True, bsz=30)
    print nn_obj.X_train.shape
    
    try :
        nn_obj.training_session(tol=1e-7, verbose=True)

    except KeyboardInterrupt :
        print ("Session closed")
        nn_obj.sess.close()
    
    nn = nn_obj
    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in range(len(nn_obj.y_test))])
    
    error_estimation = sum(deviation)
    error_estimation /= (len(nn_obj.X_test) -1)
    
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.figure("Comparaison prediction/Vraie valeure %s" % name)
    plt.plot(nn.y_test, nn.y_test, label="Wanted %s" % name, color='black')
    plt.plot(nn.y_test, nn.predict(nn.X_test), label="Predicted %s" % name, linestyle="none", marker='o', c='red')
    plt.legend(loc='best')
    
    ecart = []
    for j in range(len(nn.X_val)) :
        ecart.append(np.abs(nn.y_val[j] - nn.predict(nn.X_val[j].reshape(1,-1))[0,0])/nn.y_val[j]*100)
    
    fig, axes = plt.subplots(1,2,figsize=(10,5), num="NN Hist Pred (gauche) Val (droite)")
    axes[0].hist(ecart) 
    axes[1].plot(range(len(nn.X_val)), ecart)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Erreur (%)")
    axes[1].yaxis.set_label_position("right")
    
    plt.show()    
    nlst = range(1,70)
    knnscore = {}
    for n in nlst :
        knn = KNeighborsRegressor(n_neighbors=n).fit(nn.X_train, nn.y_train.reshape(-1))
#        print ("Score pour %s = %f" %(name, knn.score(nn.X_test, nn.y_test)))
        
        knnscore["score_%d"%n] = knn.score(nn.X_test, nn.y_test)
    knnproc(9, nn.X_train, nn.X_test, nn.X_val, nn.y_train, nn.y_test, nn.y_val, name)
    
    rff = [[j,i] for j in range(50, 500, 50) for i in range (5,50,5)] 
    rffscore = {}

    for N,n in rff :
        rf = RandomForestRegressor(n_estimators=N, max_depth=n, random_state=50).fit(nn.X_train, nn.y_train.ravel())
        rffscore["score_N%d_n%d"% (N,n)] = rf.score(nn.X_test, nn.y_test.ravel())

#        rfproc(N, n, nn.X_train, nn.X_test, nn.X_val, nn.y_train, nn.y_test, nn.y_val, name)
    
