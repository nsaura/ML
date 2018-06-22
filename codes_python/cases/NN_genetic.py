#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

import time

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import tensorflow as tf
from sklearn.model_selection import train_test_split

from random import randint, random
from operator import add

## Import de la classe TF ##
tf_folder = osp.abspath(osp.dirname("../TF/"))
sys.path.append(tf_folder)

import NN_class_try as NNC
import Class_Temp_Cst as ctc
import class_functions_aux as cfa
import Gaussian_Process_class as GPC
import NN_inference_ML as NNI

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)
NNI = reload(NNI)

##
#On suit le tutoriel de https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
##

parser = cfa.parser()
# run NN_genetic.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -N_sample 5

nn_params = dict()
global_keys = ["N_HL", "N_HN", "Act", "Opt"]

nn_params["N_HL"] = [10, 30, 50, 75, 100, 120, 150, 200, 300, 400, 500]
nn_params["N_HN"] = [100*i for i in range(1,8)]
nn_params["Act"]  = ["leakyrelu", "relu", "selu"]
nn_params["Opt"]  = ["RMS", "Adam", "RMS"]

lr = 1e-3
pop = 10
reduce_type = "sum"

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
#    print "entree de mutate : \n", c_params
    plague_str = str(np.random.choice(global_keys))
    
#    print "plague_str = ", plague_str
#    print "c_params[plague_str] = ", c_params[plague_str]
#    print "nn_params[plague_str]", nn_params[plague_str]
    
    c_params[plague_str] = np.random.choice(nn_params[plague_str])
    return c_params
    
def breed(m_params, f_params) :
    """
    On crée un nouvel individu à partir d'un mix des paramètres d'un réseau mère et d'un réseau père
    """
    c_params = dict()
    
    for k in global_keys :
        m = m_params[k]
        f = f_params[k]
        c_params[k] = np.random.choice([m,f])
    
    return c_params
    
def individual(params, I=2, O=1, max_epoch = 1000, close=True):
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
    
    dict_layers = {"I" : I,\
                   "O" : O}
    for hl in range(params["N_HL"]) :
        dict_layers["N%d" %(hl)] = params["N_HN"]
    
    loss =   "OLS"
    act  =   params["Act"]
    opti =   params["Opt"]
    
    nn = NNI.build_case(lr, X, y, act, opti, loss, reduce_type, N_ = dict_layers, max_epoch = max_epoch, scale = True)
    
    if nn.err_type == "OLS" :
        test_cost = sum([(nn.predict(nn.X_test)[i] - nn.y_test[i])**2 for i in range(len(nn.y_test))])
    
    if nn.err_type == "AVL" :
        test_cost = sum([np.abs(nn.predict(nn.X_test)[i] - nn.y_test[i]) for i in range(len(y_test))])
    
    train_cost = nn.costs[-1]
    
    if close == True :
        nn.sess.close()
    
    return nn, train_cost, test_cost
    
def evolve(network_configurations, len_pop, random_select, mutate_chance,  retain_length = 3, max_epoch = 1000) :
    """
    Dans cette fonction on va définir quels sont les réseaux pertinents pour la prochaine génération. Ce choix se fait évidemment sur le score qu'ils auront obtenus.
    
    On veillera à ne pas prendre exclusivement les meilleurs. Prendre les pires ajoutera de l'aléatoire et nous permettra de converger vers la  combinaison de paramètres la plus efficace  
    """
    # On va runner et entrainer les réseaux puis les classifier selon leur score final
    costs = []
    parents = []
    tests, trains = [], [] 
    for kset in network_configurations :
        nn, L_cost, T_cost = individual(kset, max_epoch = max_epoch)
        costs.append((T_cost, kset))
        tests.append(T_cost)
        trains.append(L_cost)
    
    alpha_c = min(tests)
    
    print ("sorted list : {}".format(sorted(costs, key=lambda x: x[0][0])))
    
    sorted_costs = [x[1] for x in sorted(costs, key=lambda x: x[0][0])]
    
    # Puis on construit la prochaine génération à partir du trie précédent
    # On garde les premiers qu'on a trié juste au dessus
    parents = sorted_costs[:retain_length]
    alpha = (costs[0], alpha_c)
    
    etendue = max(tests) - min(tests)
    
    # On va garder de manière aléatoire quelques autres
    for kset in sorted_costs[retain_length:] :
        if random_select > np.random.random():
            parents.append(kset)
            print ("kset {} kept".format(kset))
    
#    print "\n Parents = \n", parents
    # On va incorporer des mutations dans les éléments des parents, toujours de aléatoirement
    for elt in parents :
        if mutate_chance > np.random.random() :
#            print "\n elt = \n", elt
            mutate(elt)
            
    # On utilise la fonction breed pour compléter la nouvelle génération
    N_children = len(network_configurations) - len(parents)
    children = []
    
    while len(children) < N_children :
        # Père et mère aléatoire
        male_index = np.random.randint(0, len(parents) -1)
        female_index = np.random.randint(0, len(parents) -1)
        
        if male_index != female_index :
            children.append(breed(parents[male_index], parents[female_index]))
    
    parents.extend(children)
    
#    final_key = sorted_costs[-1]
##    print "final_key", final_key
#    
#    for kset in costs :
#        if kset[1] == final_key :
#            print kset
#            parents.append(kset[1])
    
#    print "parents augmented ", parents
#    print ("post added the worst case\n len parents = {}, len_pop = {}".format(len(parents), len_pop))
    
    return parents, alpha, etendue

def T_to_beta_NN(T, nn_obj, T_inf, body):
    T_n = np.asarray(map(lambda x : 0., T.line_z) )
    beta= []    
        
    for j,t in enumerate(T_n) :
        x_s = np.array([[T_inf[j], t]]) ## T_inf, T(z)
        x_s = recentre(x_s[0], nn_obj.train_mean, nn_obj.train_std)
        beta.append(nn_obj.sess.run(nn_obj.y_pred_model, feed_dict={nn_obj.x: x_s})[0,0])
    
    beta_n = np.asarray(beta)
    
    B_n = np.zeros((T.N_discr-2))
    T_n_tmp = np.zeros((T.N_discr-2))

    tol, compteur, cmax = 1e-7, 0, 15000 
    err = err_beta = 1.
    
    while (np.abs(err) > tol) and (compteur <= cmax) :
        if compteur > 0 :
            beta_n = beta_nNext
            T_n = T_nNext
        compteur +=1 
            
        T_n_tmp = np.dot(T.A2, T_n)
        
        for i in range(T.N_discr-2) :
            B_n[i] = T_n_tmp[i] + T.dt*(beta_n[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
          
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        
        beta = []
        for j,t in enumerate(T_n) :
            x_s = np.array([[T_inf[j], t]]) ## T_inf, T(z)
            x_s = recentre(x_s[0], nn_obj.train_mean, nn_obj.train_std)
            beta.append(nn_obj.sess.run(nn_obj.y_pred_model, feed_dict={nn_obj.x: x_s})[0,0])
        
#        print ("beta premiere iteration :\n {}".format(beta))
        
        beta_nNext = np.asarray(beta)
        if compteur % 20 == 0 :
            print("État : cpt = {}, err = {}, err_beta = {}".format(compteur, err, err_beta))
#        print ("Iteration {}".format(compteur))
#        print ("beta.shape = {}".format(np.shape(beta)))
#        print ("beta_nNext.shape = {}".format(np.shape(beta_nNext)))        
#        
#        print("T_curr shape ={}".format(T_nNext.shape))
        
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
        err_beta = np.linalg.norm(beta_nNext - beta_n, 2)
        
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Iterations = {} ".format(compteur))
    
    return T_nNext, beta_nNext
#-------------------------------------------#
#-------------------------------------------#
def solver_NN(T, nn_obj, T_inf, body,  N_sample= parser.N_sample, verbose = False) :
    T_inf_lambda = T_inf
    T_inf = map(T_inf, T.line_z) 

    T_ML, beta_ML = T_to_beta_NN(T, nn_obj, T_inf, body)

    T_true = GPC.True_Temp(T, T_inf, body)

    n = T.N_discr-2
    T_true = T_true.reshape(n)
    T_ML = T_ML.reshape(n)
    T_base = GPC.beta_to_T(T, T.beta_prior, T_inf, body+"_base")
    
    true_beta = GPC.True_Beta(T, T_true, map(T_inf_lambda, T.line_z))
    #    T_nmNext= T_nmNext.reshape(n)
    #    T_nMNext= T_nMNext.reshape(n)
    
    if verbose == True :
        plt.figure("T_True vs T_ML; N_sample = {}; T_inf = {}".format(N_sample, body)) 
        plt.plot(T.line_z, T_true, label="True T_field for T_inf={}".format(body), c='k', linestyle='--')
        plt.plot(T.line_z, T_ML, label="ML T_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
        plt.plot(T.line_z, T_base, label="Base solution", c='green')
        plt.legend(loc='best')
        
        title = osp.join(osp.abspath("./res_all_T_inf"),"T_True_vs_T_ML_N_sample_{}_T_inf_{}".format(N_sample, body))
        
        plt.savefig(title)
        
        plt.figure("Beta_NN_vs_True_%s" %(body))
        plt.plot(T.line_z, beta_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML_NN_%s" %(body))
        plt.plot(T.line_z, T.beta_prior, label="beta prior", c='yellow')
        plt.plot(T.line_z, true_beta, label="True_NN_%s" %(body), linestyle='--', c='k')
        plt.legend(loc = "best")
        
        title = osp.join(osp.abspath("./res_all_T_inf"),"beta_True_vs_beta_NN_N_sample_{}_T_inf_{}".format(N_sample, body))
        
        T_rel_error = np.array([np.abs(T_true[i] - T_ML[i])/T_true[i] for i in range(T.N_discr-2)])
        beta_rel_error = np.array([np.abs(true_beta[i] - beta_ML[i])/true_beta[i] for i in range(T.N_discr-2)])
        
        fig, axes = plt.subplots(1,2,figsize=(15,5))
        axes[0].plot(T.line_z, T_rel_error*100)
        axes[1].plot(T.line_z, beta_rel_error*100)
        
        axes[0].set_ylabel("Pourcentage d'erreur")
        axes[1].set_ylabel("Pourcentage d'erreur")
        axes[0].set_title("Erreur relative entre T_ML et T_true (pourcentage)")
        axes[1].set_title("Erreur relative entre beta_ML et beta_true (pourcentage)")
        
    NN_out = dict()
    NN_out["NN_T_ML"]  = T_ML   
    NN_out["NN_beta_ML"] = beta_ML.reshape(n)
    
    return NN_out

def main(len_pop, gen_max, nn_params=nn_params, max_epoch=1000) :
    
    def check_params(params) :
        elt_mere = params[0]
        check = []
        for elt in params :
            check.append(elt == elt_mere)
        
        if False in check :
            go_on = True
        
        else :
            go_on = False
        
        return go_on
        
    gen_tree = dict()
    
    params_ = first_shuffle(len_pop, nn_params)
    print ("\x1b[1;37;44mParams de la premiere génération : \n {}\x1b[0m".format(params_))
    
    gen_tree["Gen_000"] = params_
    
    bests = []
    
    gen = 0
    tol = 0.1
    alpha_Prev = []
    
    while True and gen < gen_max  :

        params_, alpha, etendue = evolve(params_, len_pop=len_pop, random_select=0.22, mutate_chance=0.42, max_epoch=max_epoch)
        print ("\x1b[1;37;44mParams de la future génération : \n {}\x1b[0m".format(params_))
        
        gen_tree["Gen_%03d"%(gen)] = params_
        
        print ("Étendue génération {} = {}".format(gen, etendue))
        if etendue <= tol :
            print("Fin d'optimisation, gen = {}".format(gen))
            break

        print ("\x1b[1;37;43mNext generation\x1b[0m")
        
        if gen > 0 :
            print("alpha_nPrev = {}".format(alpha_Prev[0]))
            print("alpha_curr = {}".format(alpha[0]))
            
            print("Cout n_prev = ".format(alpha_Prev[1]))
            print("Cout n_curr = ".format(alpha[1]))
            
        bests.append(params_[0])
            
        time.sleep(3)
        
        alpha_Prev = alpha    
        
        go_on = check(params_)
        if go_on == False :
            print("Cases are all the same. The winner is : \n{}".format(params_[0])
            break
        gen += 1
        
    return (params_, bests, gen_tree)
    
    
# Il semblerait que la combi {'Opt': 'Adam', 'Act': 'selu', 'N_HL': 120, 'N_HN': 400} soit la meilleure
