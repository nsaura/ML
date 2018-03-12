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

from random import randint, random
from operator import add

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Temp_Cst as ctc
import class_functions_aux as cfa
import Gaussian_Process_class as GPC
import NN_inference_ML as NNI

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)
NNI = reload(NNI)

parser = cfa.parser()
#run NN_genetic.py -T_inf_lst 5 10 15 20 -N_sample 1

nn_params = dict()
global_keys = ["N_HL", "N_HN", "Act", "Opt"]

nn_params["N_HL"] = [1000, 500, 100, 500, 1000, 100, 1000, 500, 100]
nn_params["N_HN"] = [2**j for j in range(6,10)]
nn_params["Act"]  = ["leakyrelu", "relu", "selu", "sigmoid"]
nn_params["Opt"]  = ["RMS", "Adam"]

lr = 1e-3
pop = 10
reduce_type = "mean"

T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X,y,v,m,s = GPC.training_set(T, parser.N_sample)

def first_shuffle(len_pop, params = nn_params) :
    """
    Première génération de réseaux construite à partir du dictionnaire de listes de possibilités
    """
    new_params  = []
    curr_params = dict()
    
    for i in range(len_pop) :
        for j, item in enumerate(params.items()) :
            curr_params[item[0]] = np.random.choice(item[1])
            
            if i > 0 :
                if curr_params[item[0]] == new_params[-1][item[0]] :
                    ind = params[item[0]].index(curr_params[item[0]])

                    try :
                        ind += 1
                        curr_params[item[0]] = params[item[0]][ind]
                    except IndexError :
                        ind -= 1
                        curr_params[item[0]] = params[item[0]][ind]
        new_params.append(curr_params)
        curr_params = dict()
    
    return new_params
    
def mutate(c_params) :
    """
    On va effecuter une seule mutation d'un paramètre aléatoire des nouvelles souches
        params      -  dictionnaire des hyperparamètres sélectionnés
        pop_missing -  nombre de network à fournir pour compléter la population
    """
    
    plague = np.random.choice(global_keys)
    
    c_params[plague] = np.random.choice(nn_params[plague])
    
    return c_params
    
def breed(m_params, f_params) :
    """
    On crée un nouvel individu à partir d'un mix des paramètres d'un réseau mère et d'un réseau père
    """
    c_params = dict()
    for m, f in zip(m_params.items(), f_params.items()) :
        c_params[m[0]] = np.random.choice([m[1], f[1]])
    
    return c_params
    
def individual(params):
    """
    nn_params doit être un dictionnaire de liste prenant plusieurs valeurs des 4 paramètres :
        1 - Nombre de HL 
        2 - Nombre de nœuds par HL
        3 - Fonction d'activation
        4 - Méthode d'optimisation
    """
    if params["Opt"] is tf.train.RMSPropOptimizer:
        momentum = 0.8
        decay = 0.7
    
    dict_layers = {"I" : 2,\
                   "O" : 1}
    for hl in range(params["N_HL"]) :
        dict_layers["N%d" %(hl)] = params["N_HN"]
    
    loss =   "OLS"
    act  =   params["Act"]
    opti =   params["Opt"]
    
    nn = NNI.build_case(lr, X, y, act, opti, loss, reduce_type, N_ = dict_layers, max_epoch = 10, scale = True)
    
    return nn
    
def evolve(lst_network_parameters) :
    """
    Dans cette fonction on va définir quels sont les réseaux pertinents pour la prochaine génération. Ce choix se fait évidemment sur le score qu'ils auront obtenus.
    On veillera à ne pas prendre exclusivement les meilleurs. Prendre les pires ajoutera de l'aléatoire et nous permettra de converger vers la  combinaison de paramètres la plus efficace  
    """
    costs = [(individual(kset).costs[-1], kset) for kset in lst_network_parameters]
    
    costs = [x[1] for x in sorted(costs, key=lambda x: x[0], reverse=True)]
    
    return costs
