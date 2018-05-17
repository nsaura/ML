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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import ML_test as mlt
mlt = reload(mlt)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def build_train(Nx) :
    X_line = np.linspace(-np.pi,np.pi,Nx)
    f = np.array(map(lambda x : x**2, X_line))
    g = np.gradient(f, X_line)

# Construction de la dataset
    X = np.zeros((4))
    y = np.zeros((1))

    for j in range(1, len(X_line)-1) :
        X = np.block([[X], [X_line[j], X_line[j+1], f[j], f[j+1]]])
        y = np.block([[y], [g[j]]])
        
    X, y = np.delete(X, 0, axis=0), np.delete(y, 0, axis=0)

    for i in range (2) :
        rand_index = np.random.permutation(np.arange(len(y)))
        XX = X[rand_index]
        yy = y[rand_index]

        X = np.block([[X], [XX]])
        y = np.block([[y], [yy]])
    
    return X,y
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------

dict_layers = {"I"  : 4,\
               "N1" : 500,\
               "N2" : 10,\
##               "N6" : 10,\
#               "N7" : 10,\
#               "N8" : 10,\
               "O"  : 1}

Nxlst = [500, 600, 700, 800 ,900, 1000]

d = {}
Color=iter(cm.Dark2(np.linspace(0,1,len(Nxlst)+1)))
c = next(Color)

plt_lines = []
leg = []

for Nx in Nxlst :
    X,y = build_train(Nx)
    c = next(Color)
    kwargs = {"bsz" : 50,\
              "BN"  : True,\
              "color" : c,\
              "label" : "Nx = %d" % Nx\
             }    
    
    nn = mlt.build_case(8e-4, X, y, act= "selu", opti="Adam", loss="OLS", max_epoch = 100,\
                        reduce_type="sum", N_=dict_layers, scale=True, step = 50, **kwargs)
    
    title = mlt.evol_loglog(nn, 100, c, kwargs["label"])
    plt.pause(0.001)
    
if plt.fignum_exists(title) :
    plt.figure(title)
plt.legend()        

Color=iter(cm.Dark2(np.linspace(0,1,len(Nxlst)+1)))
c = next(Color)

for Nx in Nxlst :
    X,y = build_train(Nx)
    c = next(Color)
    
    label = "Nx = %d" % Nx
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
    
    title = mlt.knnevol_loglog(knn, 100, c, label)
    plt.pause(0.001)
    
if plt.fignum_exists(title) :
    plt.figure(title)
plt.legend()            
    
