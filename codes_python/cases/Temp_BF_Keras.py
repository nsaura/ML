#!/usr/bin/python2.7
# -*- coding: utf-8-*-

import sys
import time
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from itertools import cycle

import os
import os.path as osp

## Import de la classe TF ##
ker_folder = osp.abspath(osp.dirname("../keras/"))
sys.path.append(ker_folder)

import argparse
import Class_Keras_NN as CKN
import NN_inference_ML as NNI

# Pour utiliser les fonctions de classs_temp
import Class_Temp_Cst as ctc  

# Pour les tracés post-process
import class_functions_aux as cfa 

# run Temp_BF_Keras -T_inf_lst 3 5 8 10 13 15 18 20 23 25 28 30 35 40 45 50 -N 71 -g_sup 1e-4 -cov_mod "full" -cptmax 150
plt.ion()

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

ctc = reload(ctc)
cfa = reload(cfa)
CKN = reload(CKN)
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

# X = [xi, tinf]
# y = [ti]

X = np.zeros((2))
y = np.zeros((1))

temp = ctc.Temperature_cst(parser) 
temp.obs_pri_model()
temp.get_prior_statistics()

if osp.exists("./data/xy_files") == False :
    os.mkdir("./data/xy_files")

case = "complet"
while case not in {"complet", "normal"} :
    case = str(input("Quel cas ? complet ou normal (boucle infinie sinon) : "))
    print ("case choisi : %s" % case)
    
Xcasetoloadifexists = "./data/xy_files/X_T" + case
ycasetoloadifexists = "./data/xy_files/y_T" + case

if osp.exists(Xcasetoloadifexists) or osp.exists(ycasetoloadifexists) :
    X = np.load(Xcasetoloadifexists)
    y = np.load(ycasetoloadifexists) 

else :
    wheretoload = osp.split(temp.path_fields)[0]
    pathfile = lambda tinf, cpt : osp.join(wheretoload, "full_obs_T_inf_%d_N69_%d.npy" %(tinf, cpt))
    for t in temp.T_inf_lst :
        for i in range(temp.num_real) :
            file = np.load(pathfile(t, i))
            for j in range(len(temp.line_z)) : 
                X = np.block([[X], [temp.line_z[j], t]])
                y = np.block([[y], [file[j]]])
           
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0, axis=0)

    np.save("./data/xy_files/X_T%s.npy" %(case), X)
    np.save("./data/xy_files/y_T%s.npy" %(case), y)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

dict_layers = {"I" : 2,\
               "N1" : [300,'selu'],\
               "N2" : [10,'selu'],\
               "N3" : [5, 'selu'],\
#               "N4" : 10,\
#               "N5" : 10,\
#               "N6" : 10,\
#               "N7" : 10,\
#               "N8" : 10,\
#               "N9" : 10,\
#               "N10": 10,\
               "O"  : [1]}

kwargs = {}
kwargs["batch_size"] = 128

dico = {'opti': 'Adam', 'metrics':['mean_squared_error', 'mean_absolute_error'], 'loss':'mse', 'max_epoch':100,
        'batch_size': 128, 'non_uniform_act' : True, 'save': True, 'scale':True, 'shuffle': True, 'color': 'navy'}



default_value ={"SGD"   : {"lr" : 0.01,\
                                  "momentum" : 0.0,\
                                  "decay" :   0.0,\
                                  "nesterov" : False},\
                         
                "RMSprop":{"lr" : 0.001,\
                                  "rho"   :   0.9,\
                                  "decay" :   0.0},\
                         
                "Adam"  : {"lr" : 0.001,\
                                 "decay" :   0.0,\
                                  "beta1" :   0.9,\
                                  "beta2" :   0.999},\
                         
                "Adamax": {"lr" : 0.002,\
                                   "beta1" :   0.9,\
                                   "beta2" :   0.999,\
                                   "decay" :   0.0},\
                         
                "Nadam" : {"lr" : 0.002,\
                                  "beta1" :   0.9,\
                                  "beta2" :   0.999,\
                                  "schedule_decay" : 0.004}\
                }

kwargs = {"batch_size" : dico['batch_size'], 'color':"navy"}

for k, v in default_value.iteritems():
    if k == dico["opti"] :
        for vk, vv in v.iteritems():
            kwargs[vk] = vv

def modif_dico(dico, key, new_value) :
    dico[key] = new_value

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def build_nn(dict_layers, dico, X, y, kwargs) :
    k_nn = CKN.K_Neural_Network(dict_layers, dico["opti"], dico['loss'], dico['metrics'], dico['max_epoch'], dico['non_uniform_act'],\
                                **kwargs)
    
    k_nn.train_and_split(X, y, shuffle=dico['shuffle'], scale=dico['scale'])
    k_nn.build()

    temp = time.strftime("%m_%d_%Hh%M", time.localtime())
    model_name = "Temperature_K_NN.h5"
    
    k_nn.compile(save=dico['save'], name=model_name)
    k_nn.fit_model()
    
    return k_nn
# Call this function k_nn =  build_nn(dict_layers, dico, X, y, kwargs)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def True_Temp(T, T_inf, body) :
    """
    T_inf doit être une liste
    """
    T_n_obs =  list(map(lambda x : 0., T.line_z) )
    T_nNext_obs = T_n_obs

    B_n_obs     =   np.zeros((T.N_discr-2, 1))
    T_n_obs_tmp =   np.zeros((T.N_discr-2, 1))

    tol ,err_obs, compteur = 1e-8, 1.0, 0 
    
    while (np.abs(err_obs) > tol) and (compteur < 10000) :
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
    
    return T_nNext_obs
#-------------------------------------------------------------------------------
def solve_and_compare(temp, k_nn, T_inf, body) :
    """ T_inf doit etre une lambda """
    
    T_inf_list = map(T_inf, temp.line_z)
    true = True_Temp(temp, T_inf_list, body)
    
    T_solved = []
    for j, zj in enumerate(temp.line_z) :
        inputs = np.array([zj, T_inf_list[j]])
        
        if k_nn.scale == True :
            inputs -= k_nn.X_train_mean
            inputs /= k_nn.X_train_stdd
            
        T_solved.append(k_nn.model.predict(inputs.reshape(1,-1))[0,0])
    
    T_solved = np.asarray(T_solved)
    true = np.asarray(true)
    
#    print T_solved.shape
#    print [T_solved[i] - true[i] for i in range(len(temp.line_z))]
    
    plt.figure("La comp %s" % body)
    plt.plot(temp.line_z, true, label="True", color='black')
    plt.plot(temp.line_z, T_solved, label="pred", linestyle="none", marker='o', color=k_nn.kwargs["color"], fillstyle="none")
    plt.legend()
