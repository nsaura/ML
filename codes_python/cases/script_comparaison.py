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
import NN_inference_ML as NNI

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)
NNI = reload(NNI)

parser = cfa.parser()

plt.ion()

# On déclare un objet de la classe T pour pouvoir avoir accès à des grandeurs propres à la méthode.
# On n'a cependant pas besoin de relancer l'inférence
# Puisque les distributions importantes ont été écrites
def calcul_erreurs(tab1, tab2, tbase):
    err1, err2 = [], []
    for c1, c2, cb in zip(tab1, tab2, tbase) :
        if cb == 0:
            print("cb = 0")
            break
        err1.append((c1 - cb)/cb)
        err2.append((c2 - cb)/cb)
    return np.asarray(err1), np.asarray(err2)
    
T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X_NN,y_NN,v_NN = GPC.training_set(T, parser.N_sample)
X_GP,y_GP,v_GP = GPC.training_set(T, 5)

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
NN = NNI.build_case(1e-3, X_NN, y_NN, act="relu", opti="RMS", loss="OLS", decay=0.7, momentum=0.8, max_epoch = parser.N_epoch)

# On prépare les différent cas
lambda_list = [lambda z: 28,\
               lambda z: 55,\
               lambda z: 15+5*np.cos(np.pi*z)\
              ]
body_list = ["28", "55", "15+5cos(piz)"]

T_ML_dict = dict()
beta_ML_dict = dict()
GP_T_max_dict = dict()
GP_T_min_dict = dict()

T_true = dict()
T_base = dict()

key_GP = ["GP_T_ML", "GP_T_ML_max", "GP_T_ML_min"]

h_op, phi_var_inv = GPC.maximize_LML(T, X_GP, y_GP, v_GP, 5)

for l,b in zip(lambda_list, body_list):
    GP_res = GPC.solver_ML(T, X_GP, y_GP, v_GP, h_op, phi_var_inv, 5, l, b, False)
    NN_res = NNI.solver_NN(T, NN, parser.N_sample, l, b)

    key = lambda s : s + "_%s" %(b)

    T_ML_dict[key("GP_T_ML")]    =   GP_res["GP_T_ML"]
    T_ML_dict[key("NN_T_ML")]   =   NN_res["NN_T_ML"]

    beta_ML_dict[key("GP_beta_ML")] = GP_res["GP_beta_ML"]
    beta_ML_dict[key("NN_beta_ML")] = NN_res["NN_beta_ML"]

    GP_T_min_dict[key("GP_T_ML_min")] = GP_res["GP_T_ML_min"]
    GP_T_max_dict[key("GP_T_ML_max")] = GP_res["GP_T_ML_max"]

    T_true[key("T_true")] = GP_res["T_true"]
    T_base[key("T_base")] = GP_res["T_base"]

    # Comparaison des champs de température    
    plt.figure("T_ML_NN_vs_T_ML_GP_%s" %(b))

    plt.plot(T.line_z, T_true[key("T_true")], label=key("True T"), c='k', linestyle='--')
    plt.plot(T.line_z, T_base[key("T_base")], label=key("T_base"), c='green')

    plt.plot(T.line_z, T_ML_dict[key("NN_T_ML")] , marker='o',\
            label=key("NN_T_ML"), c='magenta',linestyle='none', fillstyle='none')
    plt.plot(T.line_z, T_ML_dict[key("GP_T_ML")], marker='+',\
            label=key("GP_T_ML"), c='yellow', linestyle='none')

    plt.fill_between(T.line_z, GP_T_min_dict[key("GP_T_ML_min")], GP_T_max_dict[key("GP_T_ML_max")],\
    facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="grey", label="Span")

    plt.legend(loc="best")

    # Comparaison des beta NN vs ML
    plt.figure(key("Beta GP vs NN vs Inference"))

    plt.plot(T.line_z, beta_ML_dict[key("GP_beta_ML")], marker = 'o',\
            label=key("GP_beta_ML"), c='purple', linestyle="none", fillstyle='none')
    plt.plot(T.line_z, beta_ML_dict[key("NN_beta_ML")], marker = 'o',\
            label=key("NN_beta_ML"), c='k', linestyle="none", fillstyle='none')

    plt.legend(loc="best")
    # Tracés des erreurs relatives :
    errGP, errNN = calcul_erreurs(T_ML_dict[key("GP_T_ML")],\
                                  T_ML_dict[key("NN_T_ML")],\
                                  T_true[key("T_true")]\
                                 )
    plt.figure("ML_-_true_/_true")
    plt.plot(T.line_z, errGP, marker="s", label=key("Erreur GP vav True"), linestyle='none',\
                fillstyle='none', c='green')
    plt.plot(T.line_z, errNN, marker='o', label=key("Erreur NN vav True"), linestyle='none',\
                fillstyle='none', c='purple')
    plt.legend()

