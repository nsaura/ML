#!/usr/bin/python2.7
# -*- coding: utf-8-*-

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import os.path as osp

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import argparse
import NN_class_try as NNC

import NN_inference_ML as NNI

from itertools import cycle

import Class_Temp_Cst as ctc  #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

#run NN_Temp.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -cptmax 150 -N 71 -g_sup 1e-4 -cov_mod "full"

plt.ion()

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

#run script_data_generation.py -T_inf_lst 30 -cptmax 150 -N 71 -g_sup 1e-4 -cov_mod "full"
#run script_data_generation.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -cptmax 150 -N 71 -g_sup 1e-4 -cov_mod "full" 
ctc = reload(ctc)
cfa = reload(cfa)
NNC = reload(NNC)
NNI = reload(NNI)
############
#- - - - - -
# - - - - -
#   Code   #
# - - - - - -
#- - - - - -
############

parser = cfa.parser()

temp = ctc.Temperature_cst(parser)

# X = [tinf, h, xi, eps0]
# X = [xi, tinf]
# y = [ti]
#X = np.zeros((2))
#y = np.zeros((1))

#wheretoload = osp.split(temp.path_fields)[0]
#pathfile = lambda tinf, cpt : osp.join(wheretoload, "full_obs_T_inf_%d_N69_%d.npy" %(tinf, cpt))
#for t in temp.T_inf_lst :
#    for i in range(temp.num_real) :
#        file = np.load(pathfile(t, i))
#        for j in range(len(temp.line_z)) : 
#            X = np.block([[X], [temp.line_z[j], t]])
#            y = np.block([[y], [file[j]]])
#       
#X = np.delete(X, 0, axis=0)
#y = np.delete(y, 0, axis=0)

N_ = {"I" : 2,\
               "N1" : 300,\
               "N2" : 80,\
#               "N3" : 5,\
#               "N4" : 10,\
#               "N5" : 10,\
#               "N6" : 10,\
#               "N7" : 10,\
#               "N8" : 10,\
#               "N9" : 10,\
#               "N10": 10,\
               "O"  : 1}

#lr, X, y, act, opti, loss, reduce_type, N_=dict_layers, max_epoch=parser.N_epoch, scale=True, verbose=True, **kwargs) :
nn = NNI.build_case(1e-3, X, y, "selu", "Adam", "lasso", "sum", N_=N_, max_epoch=150, scale=True, verbose=True, color="purple", **kwargs)
# full_obs

def True_Temp(T, T_inf, body) :
    """
    T_inf doit être une liste
    """
    T_n_obs =  list(map(lambda x : 0., T.line_z) )
    T_nNext_obs = T_n_obs

    B_n_obs     =   np.zeros((T.N_discr-2, 1))
    T_n_obs_tmp =   np.zeros((T.N_discr-2, 1))

    tol ,err_obs, compteur = 1e-8, 1.0, 0 
    
    while (np.abs(err_obs) > tol) and (compteur < 15000) :
        if compteur > 0 :
            T_n_obs = T_nNext_obs
        compteur += 1
        T_n_obs_tmp = np.dot(T.A2, T_n_obs)

        for i in range(T.N_discr-2) :
            B_n_obs[i] = T_n_obs_tmp[i] + T.dt*  (
            ( 10**(-4) * ( 1.+5.*np.sin(3.*T_n_obs[i]*np.pi/200.) + 
            np.exp(0.02*T_n_obs[i]) + T.lst_gauss[0][i] ) ) *(T_inf[i]**4 - T_n_obs[i]**4)
             + T.h * (T_inf[i]-T_n_obs[i])      ) 

        T_nNext_obs = np.dot(np.linalg.inv(T.A1), B_n_obs)
        err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2)
    
    print ("Calculus with {} completed. Convergence status :".format(body))
    print ("Err_obs = {} ".format(err_obs))    
    print ("Iterations = {} ".format(compteur))
    
#    plt.figure("T_inf : {}".format(body)) 
#    plt.plot(T.line_z, T_nNext_obs, label="T_inf={}".format(body))
#    plt.legend()
    
    return T_nNext_obs
    
def solve_and_compare(temp, nn, T_inf, body) :
    """ T_inf doit etre une lambda """
    T_inf_list = map(T_inf, temp.line_z)
    true = True_Temp(temp, T_inf_list, body)
    
    T_solved = []
    for j, zj in enumerate(temp.line_z) :
        inputs = np.array([zj, T_inf_list[j]])
        scaled_inputs = nn.scale_inputs(inputs)
        inputs = scaled_inputs.reshape(1,-1)
        T_solved.append(nn.predict(inputs)[0,0])
    
    T_solved = np.asarray(T_solved)
    true = np.asarray(true)
    
#    print T_solved.shape
#    print [T_solved[i] - true[i] for i in range(len(temp.line_z))]
    
    plt.figure("La comp %s" % body)
    plt.plot(temp.line_z, true, label="True", color='black')
    plt.plot(temp.line_z, T_solved, label="pred", linestyle="none", marker='o', color=nn.kwargs["color"], fillstyle="none")
    plt.legend()
