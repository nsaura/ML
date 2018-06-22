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
tf_folder = osp.abspath(osp.dirname("../TF/"))
sys.path.append(tf_folder)
import NN_class_try as NNC

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

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
    print (kwargs)
    nn_obj = NNC.Neural_Network(lr, N_, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.layer_stacking_and_act(activation=act)
    nn_obj.def_optimizer(opti)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.case_specification_recap()
    
    kwargs = nn_obj.kwargs
    
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
        ecart_test.append((y_test[j] - y_pred_test[j])/(y_test[j] + 1e-5) * 100)
    
    for j in range(X_val.shape[0]):
        ecart_val.append((y_val[j] - y_pred_val[j])/(y_val[j] + 1e-5) * 100)
    
    fig, axes = plt.subplots(2,2,figsize=(10,5), num="RF Hist Pred (gauche) Val (droite)")
    
    for ax, ecart, name in zip(axes, [ecart_test, ecart_val], ["test", "val"]) :
        ax[0].hist(ecart) 
        ax[1].plot(range(len(ecart)), ecart)
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Erreur en pourcentage")
        ax[1].yaxis.set_label_position("right")
        ax[1].xaxis.set_label_coords(0.01, -0.025)
        ax[1].set_title("%s" %name)
    
    fig.tight_layout()
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
               "N1" : 100,\
               "N2" : 100,\
               "N3" : 100,\
               "N4" : 100,\
               "N5" : 100,\
               "N6" : 100,\
               "N7" : 100,\
               "N8" : 100,\
               "N9" : 100,\
               "N10" : 100,\
#               "N11" : 50,\
#               "N12" : 50,\
#               "N13" : 50,\
#               "N14" : 50,\
#               "N15" : 50,\
               "O"  : 1\
               }

X, y, name = datas[0]

X = np.block([[X], [X*np.random.random()*1e-2]])
y = np.block([[y], [y*np.random.random()*1e-3]])

y = y.reshape(-1,1) 

dict_layers = gen_dict(X.shape[1])
color = "blue"

nn_ML = build_case(5e-4, X, y , act="selu", opti="Adam", loss="OLS", max_epoch=1800, reduce_type="sum", verbose=True, N_=dict_layers, color=color, scale=False, BN=True, bsz=50)

try :
    nn_ML.training_session(tol=1e-7, verbose=True)

except KeyboardInterrupt :
    print ("Session closed")
    nn_ML.sess.close()

plt.figure("True vs Stack ensemble")
plt.plot(nn_ML.y_test, nn_ML.y_test, label="Expected")
plt.plot(nn_ML.y_test, nn_ML.predict(nn_ML.X_test), label="Stack Results ytest NN",\
        marker='o', fillstyle='none', linestyle='none', color="dodgerblue")
plt.legend()

XX = np.zeros((1))
yy = np.zeros((1))

for j in range(len(y)) :
    val = nn_ML.predict( [nn_ML.X[j]] )[0,0]
    XX = np.block([[XX], [val] ]) #y_NN
    yy = np.block([[yy], [nn_ML.y[j]] ]) # y_true

XX = np.delete(XX, 0, axis=0)
yy = np.delete(yy, 0, axis=0)

permute_indices = np.random.permutation(np.arange(len(y)))
XX = XX[permute_indices]
yy = yy[permute_indices]

xtrain, xtest, ytrain, ytest = train_test_split(XX, yy, random_state=0)

xtest_length = len(xtest)
xval = xtest[:int(xtest_length*0.2)]
yval = ytest[:int(xtest_length*0.2)]
    
xtest = xtest[int(xtest_length*0.2):]
ytest = ytest[int(xtest_length*0.2):]

nlst = range(1,70)
knnscore = {}
for n in nlst :
    knn = KNeighborsRegressor(n_neighbors=n).fit(xtrain, ytrain.reshape(-1))
#        print ("Score pour %s = %f" %(name, knn.score(nn.X_test, nn.y_test)))
    knnscore["score_%d"%n] = knn.score(xtest, ytest)
    print ("Score pour %s = %f" %(name, knn.score(xtest, ytest)))
    
index_max = knnscore.values().index(max(knnscore.values()))    
n_neigh_best = int(knnscore.keys()[index_max].split("_")[1])

print (n_neigh_best)

knn = KNeighborsRegressor(n_neighbors=n_neigh_best).fit(xtrain, ytrain.reshape(-1))

plt.figure("True vs Stack ensemble")
plt.plot(ytest, ytest, label="Expected")
plt.plot(ytest, knn.predict(xtest), label="Stack Results ytest KNN",\
        marker='o', fillstyle='none', linestyle='none')
plt.legend()

ecart_test = []
ecart_val  = []

for j in range(xtest.shape[0]):
    y_pred_test =   knn.predict(xtest)
    y_pred_val  =   knn.predict(xval)
    
    ecart_test.append(np.abs(ytest[j] - y_pred_test[j])/(ytest[j] + 1e-5) * 100)

for j in range(xval.shape[0]):
    ecart_val.append(np.abs(yval[j] - y_pred_val[j])/(yval[j] + 1e-5) * 100)

fig, axes = plt.subplots(2,2,figsize=(10,5), num="NN \& KNN Hist Pred (gauche) Val (droite)")

for ax, ecart, name in zip(axes, [ecart_test, ecart_val], ["test", "val"]) :
    ax[0].hist(ecart) 
    ax[1].plot(range(len(ecart)), ecart)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Erreur en pourcentage")
    ax[1].yaxis.set_label_position("right")
    ax[1].xaxis.set_label_coords(0.01, -0.025)
    ax[1].set_title("%s" %name)
    
    fig.tight_layout()
plt.show()


rff = [[j,i, imp] for j in range(30, 420, 30) for i in range (3,15,3) for imp in [0.1,0.15,0.2,0.25,0.3]] 
rffscore = {}
for N,n,imp in rff :
    rf = RandomForestRegressor(n_estimators=N, max_depth=n, random_state=50, bootstrap=True, min_impurity_decrease=imp).fit(xtrain, ytrain.ravel())
    rffscore["score_N%d_n%d_imp%s"% (N,n,str(imp))] = rf.score(xtest, ytest.ravel())
    
    print ("Score pour %s = %f" %(name, rf.score(xtest, ytest)))
    
index_max = rffscore.values().index(max(rffscore.values()))    
N_est_best, md_best, imp_best = int(rffscore.keys()[index_max].split("_")[1][1:]), int(rffscore.keys()[index_max].split("_")[2][1:]), float(rffscore.keys()[index_max].split("_")[3][3:])

print ("Best number of estimators : %d\t Best max_depth : %d" % (N_est_best, md_best))

rff = RandomForestRegressor(n_estimators=N_est_best, max_depth = md_best).fit(xtrain, ytrain.reshape(-1))

plt.figure("True vs Stack ensemble")
plt.plot(ytest, ytest, label="Expected")
plt.plot(ytest, rff.predict(xtest), label="Stack Results ytest RFF",\
        marker='o', fillstyle='none', linestyle='none')
plt.legend()

ecart_test = []
ecart_val  = []

for j in range(xtest.shape[0]):
    y_pred_test =   rff.predict(xtest)
    y_pred_val  =   rff.predict(xval)
    
    ecart_test.append(np.abs(ytest[j] - y_pred_test[j])/(ytest[j] + 1e-5) * 100)

for j in range(xval.shape[0]):
    ecart_val.append(np.abs(yval[j] - y_pred_val[j])/(yval[j] + 1e-5) * 100)

fig, axes = plt.subplots(2,2,figsize=(10,5), num="NN \& RFF Hist Pred (gauche) Val (droite)")

for ax, ecart, name in zip(axes, [ecart_test, ecart_val], ["test", "val"]) :
    ax[0].hist(ecart) 
    ax[1].plot(range(len(ecart)), ecart)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Erreur en pourcentage")
    ax[1].yaxis.set_label_position("right")
    ax[1].xaxis.set_label_coords(0.01, -0.025)
    ax[1].set_title("%s" %name)
    
    fig.tight_layout()
    
plt.show()


