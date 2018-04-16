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
import glob

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Vit_Choc as cvc

cvc = reload(cvc)

#def x_y_burger () :
parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

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

X = np.zeros((3))
y = np.zeros((1))

print lst_pairs_bu[0][0]
print X[0]
#return X, y 
# pairs :   p[0] -> beta
#           p[1] -> u
for p in lst_pairs_bu :
    u = np.load(p[1])
    beta = np.load(p[0])
    for j in range(1, len(u)-1) :
        X = np.block([[X], [u[j-1], u[j], u[j+1]]])
        y = np.block([[y], [beta[j]]])
        
X = np.delete(X, 0, axis=0)
y = np.delete(y, 0, axis=0)

dict_layers = {"I" : 3,\
               "N1" : 100,\
               "N2" : 50,\
               "N3" : 50,\
               "N4" : 50,\
               "N5" : 50,\
               "N6" : 50,\
               "N7" : 50,\
               "N8" : 50,\
               "N9" : 50,\
               "N10": 50,\
               "O"  : 1}

def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=False, **kwargs) :
    plt.ion()
    nn_obj = NNC.Neural_Network(lr, N_=dict_layers, max_epoch=max_epoch)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.feed_forward(activation=act)
    nn_obj.def_training(opti, **kwargs)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.def_optimization()
    try :
        nn_obj.training_session(tol=1e-3, batched=False, step=50)
    
    except KeyboardInterrupt :
        print "Session closed"
        nn_obj.sess.close()
    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    test_line = range(len(nn_obj.X_test))
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in test_line])
    error_estimation = sum(deviation)

    if verbose == True :
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
    print ("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()
    return nn_obj
    
#def NN_solver(nn_obj):
