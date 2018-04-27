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

import keras 
import tensorflow as tf

import argparse

plt.ion()

#run keras_burger.py -nu 2.5e-2 -itmax 100 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10 -dp "./../cases/data/burger_dataset/" -p "./../cases/logbooks/"

model = keras.models.load_model("non-trained-model-1.h5")

## Import de la classe cvc ##
cvc_folder = osp.abspath(osp.dirname("../cases/Class_Vit_Choc.py"))
sys.path.append(cvc_folder)
import Class_Vit_Choc as cvc

## Import de la classe cwc ##
cwc_folder = osp.abspath(osp.dirname("../cases/Class_write_case.py"))
sys.path.append(cwc_folder)
import Class_write_case as cwc

parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

def train_and_split(X, y, random_state=0, strat=False, scale=False, shuffle=True):
    if shuffle == True :
        # Inspired by : Sebastian Heinz se :
        # Medium : a simple deep-learning model for stock price prediction using tensorflow
        permute_indices = np.random.permutation(np.arange(len(y)))
        X = X[permute_indices]
        y = y[permute_indices] 

    if strat == True :
        if np.size(np.shape(y)) == 1 :
            xxyys = train_test_split(X, y, stratify=y,\
                    random_state=random_state)
        if np.size(np.shape(y)) == 2 :
            xxyys = train_test_split(X, y, stratify=y.reshape(-1),\
                    random_state=random_state)
    else :
        xxyys = train_test_split(X, y, random_state=random_state)
    
    X_train, X_test = xxyys[0], xxyys[1]
    y_train, y_test = xxyys[2], xxyys[3]
    
    X_train_std  =  X_train.std(axis=0)
    X_train_mean =  X_train.mean(axis=0)
    
    if scale == True :
        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        
        for i in range(X_train_mean.shape[0]) :
            X_train_scaled[:, i] = X_train[:, i] - X_train_mean[i]
            X_test_scaled[:, i] = X_test[:, i] - X_train_mean[i]
            
            if np.abs(X_train_std[i]) > 1e-12 :
                X_train_scaled[:,i] /= X_train_std[i]
                X_test_scaled[:,i] /= X_train_std[i]
        
        print ("Scaling done")
        print ("X_train_mean = \n{}\n X_train_std = \n{}".format(X_train_mean, X_train_std))
        
        X_train = X_train_scaled
        X_test = X_test_scaled
    
    scale = scale
    X_train_mean = X_train_mean
    X_train_std  = X_train_std
    X_train, X_test = X_train, X_test
    y_train, y_test = y_train, y_test
    
    return X_train, X_test, y_train, y_test, X_train_mean, X_train_std
    
def xy_burger (num_real, cb=cb) :
    #cb.minimization(maxiter=50, step=5)

    u_name = cb.u_name
    b_name = cb.beta_name

    root = osp.split(cb.beta_path)[0]

    uloc = cb.inferred_U
    betaloc = cb.beta_path
    cholloc = cb.chol_path

    # On va se servir de ça
    b_files = glob.glob(betaloc+'/*')
    u_files = glob.glob(uloc+'/*')
    c_files = glob.glob(cholloc+'/*')

    b_u = dict()
    b_c = dict()
    lst_pairs_bu = []
    lst_pairs_bc = []

    l = osp.split(b_files[0])[1].split("_")
    ll = [i.split(":") for i in l[1:-1]]

    # To be able to modify u,b and c_files we have first to copy them 
    u_sorted = sorted(u_files, key= lambda x: int(osp.splitext(x)[0][-3:]))
    b_sorted = sorted(b_files, key= lambda x: int(osp.splitext(x)[0][-3:]))
    c_sorted = sorted(c_files, key= lambda x: int(osp.splitext(x)[0][-3:]))


    cpt_rm = 0
    for u, b, c in zip(u_sorted, b_sorted, c_sorted) :
        cond = lambda f : int(osp.splitext(f)[0][-3:]) > cb.itmax-1
        if cond(u) and cond(b) and cond(c) :
            u_files.remove(u)
            b_files.remove(b)
            c_files.remove(c)
#            print "RM"
            
            cpt_rm += 1
    print ("%d files removed from u.b and c_files lists" %(cpt_rm))

    #print ("u_files = \n{}b_files = \n{}c_files = {}\n".format(u_files, b_files, c_files))

    # Init
    for elt in ll :
        b_u[elt[0]] = []
        b_c[elt[0]] = []
        
    for b in b_files :
        to_find = osp.split(b)[1][4:]
        
        u_to_find = uloc + "/U" + to_find
        
        if osp.exists(u_to_find) :
            lst_pairs_bu.append((b, u_files[u_files.index(u_to_find)]))
        else :
            print ("%s does not exist" %(u_to_find))

        c_to_find = cholloc + "/chol" + to_find
        
        if osp.exists(c_to_find)  :
            lst_pairs_bc.append((b, c_files[c_files.index(c_to_find)]))
        else :
            print ("%s does not exist" %(c_to_find))
    # For checking    
    #    for i in range (10) :
    #        print lst_pairs[np.random.randint(len(lst_pairs))]

    X = np.zeros((4))
    y = np.zeros((1))

    print (lst_pairs_bu[0][0])
    print (X[0])
    #return X, y 
    # pairs :   p[0] -> beta
    #           p[1] -> u
    beta_chol = dict()
    u_beta_chol = dict()

    lst_pairs_bu = sorted(lst_pairs_bu)
    lst_pairs_bc = sorted(lst_pairs_bc)

    #color=cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0]))
    color=iter(cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0])))

    for it, (pbu, pbc) in enumerate(zip(lst_pairs_bu, lst_pairs_bc)) :
        beta = np.load(pbu[0])
        chol = np.load(pbc[1])
        u = np.load(pbu[1])
        
        u_mean = np.zeros_like(u[1:cb.Nx-1])
        c = next(color)
        
        if osp.splitext(pbu[1])[0][-3:] != "000" :
            for i in range(num_real) :
                beta_chol[str(i)] = beta_last + chol_last.dot(np.random.rand(len(beta_last)))            
    #            print np.shape(beta_chol[str(i)])
                u_beta_chol[str(i)] = cb.u_beta(beta_chol[str(i)], u_last)
                
    #            for k in range(0, len(u)-2) : #Pour aller de 1 à Nx-1 vav de u-beta_chol mais commencer aux bons endroits
    #                u_mean[k] += u_beta_chol[str(i)][k+1] / float(num_real)
                
        for j in range(1, len(u)-1) :
            if osp.splitext(pbu[1])[0][-3:] != "000" :
                X = np.block([[X], [u[j-1], u[j], u[j+1], np.mean(u)]])
                y = np.block([[y], [beta[j]]])
                
                for i in range(num_real) :
                    bb = beta_chol[str(i)]
                    uu = u_beta_chol[str(i)]
                    
                    X = np.block([[X], [uu[j-1], uu[j], uu[j+1], np.mean(uu)]])
                    y = np.block([[y], [bb[j]]])
            
        u_last = u
        beta_last = beta  
        chol_last = chol
        
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    
    return X, y

X, y = xy_burger(5, cb)

keras_model = keras.models.load_model("non-trained-model-1.h5")

X_train, X_test, y_train, y_test, mean, std = train_and_split(X, y, random_state=10, scale=True, shuffle=True)

history = keras_model.fit(X_train, y_train, batch_size=256, epochs=500)

lenm = 3
color = iter(cm.magma_r(lenm))

for i in ['mean_squared_error', 'mean_absolute_error'] :
    c = next(color)
    plt.plot(history.history['%s' % i], label="%s" % i)

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

    return xs


def NN_solver(model, X_train_mean, X_train_std, cb=cb, scale=True):
    beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
    
    # Initialisation it = 1
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=1))
    plt.figure()
    for it in range(1, cb.itmax) :
        beta = []
        u_mean = np.mean(u)
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1], u_mean])
            print xs
            if scale == True :
                xs = recentre(xs, X_train_mean, X_train_std)
            print xs
            xs = xs.reshape(-1, 4)
            beta.append(model.predict(xs)[0,0])

        print(beta, type(beta), np.shape(beta))
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
        u[0] = u[-2]
        u[-1] = u[1]
        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c='steelblue') 
        plt.legend()
        plt.pause(5)
        plt.clf()

plt.figure("Comparaison prediction/Vraie valeure")
plt.plot(y_test, y_test, label="Wanted", color='black')
plt.plot(y_test, keras_model.predict(X_test), label="Predicted", linestyle="none", marker='o', c='green')
plt.legend(loc='best')

#import time
#time.sleep(5)
#NN_solver(keras_model, mean, std, scale=True)
