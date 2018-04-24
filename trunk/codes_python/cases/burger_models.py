#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

# Author : NS
# Use of Various Helps From SO and CV; stack community
# Ridge https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb
# Vanilla Gradient descent https://towardsdatascience.com/improving-vanilla-gradient-descent-f9d91031ab1d

# To run
# run burger_models.py -nu 2.5e-2 -itmax 100 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10    
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from itertools import cycle
from matplotlib.pyplot import cm
from sklearn.ensemble import RandomForestRegressor

from scipy import optimize as op

import numdifftools as nd

import glob
import time

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Vit_Choc as cvc

from burger_NN import xy_burger, recentre

NNC = reload(NNC)
cvc = reload(cvc)

parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

cb.obs_res(True, True)
X, y = xy_burger(input("num_real : " ), cb)

nn_obj = NNC.Neural_Network(1e-3, max_epoch=10)
nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=False)

X_train, y_train, X_test, y_test = nn_obj.X_train, nn_obj.y_train, nn_obj.X_test, nn_obj.y_test
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

forest = RandomForestRegressor(n_estimators=100, random_state=1)
forest.fit(X_train, y_train)

print("Train Accuracy: {:.3f}".format(forest.score(X_train, y_train)))
print("Test Accuracy: {:.3f}".format(forest.score(X_test, y_test)))

#---------------------------------------------------------------------------------------------------------------------------
def solver_forest(nn_obj, cb=cb, forest=forest) :
#    run knn_ML.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -N_sample 10

    beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
    
    # Initialisation it = 1
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=0))
    for it in range(1, cb.itmax) :
        beta = []
        u_mean = np.mean(u)
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1], u_mean])
            if nn_obj.scale == True :
                xs = recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
            xs = xs.reshape(-1,4)
            beta.append(forest.predict(xs)[0])
            print(beta)

        print(beta, type(beta), np.shape(beta))
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
#        u[0] = u[-2]
#        u[-1] = u[1]
        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted it = %d"%(it), marker='o', fillstyle = 'none', linestyle= 'none', c='steelblue') 
        plt.legend()
        plt.pause(5)
        plt.clf()
#    grid_search = GridSearchCV(SVC(), param_grid, cv=5) # Objet a entrainer et evaluer

    print("Test differences entre prediction et Y_test : ")
    print("{}".format(y_test - reg.predict(X_test)))
#    
#    

#def solver_RF(forest=forest):
    


