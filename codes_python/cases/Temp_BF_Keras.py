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
               "N2" : [100,'selu'],\
               "N3" : [10, 'selu'],\
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
    if key not in dico.keys() :
        raise KeyError
        
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
    
    while (np.abs(err_obs) > tol) and (compteur < 12000) :
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
#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

path = osp.split(Xcasetoloadifexists)[0]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def write_train_and_test_sets(k_nn, action='save', path=path):
    
    paths = {'X_train' : [osp.join(path, 'X_train.npy'), k_nn.X_train],
             'y_train' : [osp.join(path, 'y_train.npy'), k_nn.y_train],
             'X_test'  : [osp.join(path, 'X_test.npy') , k_nn.X_test] ,
             'y_test'  : [osp.join(path, 'y_test.npy') , k_nn.y_test] }
    
        
    if action == 'save' :
        for lsts in paths.values() : 
            np.save(lsts[0], lsts[1])
            
    xtr, ytr = np.load(paths['X_train'][0]), np.load(paths['y_train'][0])
    xte, yte = np.load(paths['X_test'][0]), np.load(paths['y_test'][0])
    
    return xtr, ytr, xte, yte

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#def scikit_prediction(model, model_name, T_inf)

def processing(T, model, model_name, T_inf, body, scale) :
    T_ML, beta_ML = T_to_beta(T, model, mean, std, T_inf, body, scale)
    T_true = GPC.True_Temp(T ,map(T_inf, T.line_z), body)
    true_beta = GPC.True_Beta(T, T_true, map(T_inf, T.line_z))

    plt.ion()
    plt.figure("T_%s_vs_True%s" %(model_name, body))
    plt.plot(T.line_z, T_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML_%s_%s" %(model_name, body))
    plt.plot(T.line_z, T_true, label="True_%s_%s" %(model_name, body), linestyle='--', c='k')
    plt.legend()

    plt.figure("beta_%s_vs_True_%s" %(model_name, body))
    plt.plot(T.line_z, beta_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML_%s_%s" %(model_name, body))
    plt.plot(T.line_z, true_beta, label="True_%s_%s" %(model_name, body), linestyle='--', c='k')
    plt.legend()
    
    return T_ML, beta_ML, T_true.ravel(), true_beta.ravel()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def several_models(X_train, y_train, X_test, y_test) :

    ## Entraînement des modèles
    ridge = Ridge(alpha=0.0001).fit(X_train,y_train)
    print("Ridge Score train set : {}".format(ridge.score(X_train, y_train.ravel())))
    print("Ridge Score test set :  {}\n ".format(ridge.score(X_test, y_test)))

    lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train,y_train)
    print("Lasso 0.01 Score train set : {}".format(lasso.score(X_train, y_train)))
    print("Lasso 0.01 Score test set :  {}\n ".format(lasso.score(X_test, y_test)))

    lasso_ = Lasso(alpha=0.00001, max_iter=100000).fit(X_train,y_train)
    print("Lasso 0.00001 Score train set : {}".format(lasso_.score(X_train, y_train)))
    print("Lasso 0.00001 Score test set :  {}\n ".format(lasso_.score(X_test, y_test)))

    LinReg = LR().fit(X_train,y_train)
    print("Linear Regression Train set : {}".format(LinReg.score(X_train, y_train)))
    print("Linear Regression Test set :  {}\n ".format(LinReg.score(X_test, y_test)))

    n_esti = 100
    forest = RandomForestRegressor(n_estimators=n_esti, random_state=0)
    forest.fit(X_train, y_train)
    print("Forest n_esti {} Score train set : {}".format(n_esti ,forest.score(X_train, y_train)))
    print("Forest n_esti {} Score test set :  {}\n ".format(n_esti, forest.score(X_test, y_test)))

    lr, max_depth = 0.1, 5
    gbrt_mdlow = GradientBoostingRegressor(random_state=0, max_depth=max_depth, learning_rate=lr).fit(X_train, y_train)
    print("GBRT lr={}, max_dpeth={} score train set: {}".format(lr, max_depth, gbrt_mdlow.score(X_train,y_train)))
    print("GBRT lr={}, max_depth={} score test set:  {}\n ".format(lr, max_depth, gbrt_mdlow.score(X_test,y_test)))

    max_depth = 4 
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X_train, y_train)
    print("DT max_depth {} Score train set : {}".format(max_depth, tree.score(X_train, y_train)))
    print("DT max_depth {} Score test set :  {}\n ".format(max_depth, tree.score(X_test, y_test)))

    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    print("reg Score train set : {}".format(knn.score(X_train, y_train)))
    print("reg Score test set :  {}\n ".format(knn.score(X_test, y_test)))

    model = [ridge, lasso, LinReg, forest, gbrt_mdlow, tree, knn]
    names = ["Ridge", "lasso", "LinReg", "Forest", "GrdBoost", "DT", "knn" ]

    llist = [lambda z : 35-15*z, lambda z: 35+20*np.sin(np.pi*2*z), lambda z: 28]
    blist = ["35-15z", "35+20sin_2piz", "28"]

    for m, n in zip(model, names): 
        for T_inf, body in zip(llist, blist):  
            processing(T, m, n, mean, std, T_inf, body, True)

