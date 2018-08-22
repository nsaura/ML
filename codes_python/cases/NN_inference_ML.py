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

import NN_class_try as NNC
import Class_Temp_Cst as ctc
import class_functions_aux as cfa
import Gaussian_Process_class as GPC

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

parser = cfa.parser()

plt.ion()

# On déclare un objet de la classe T pour pouvoir avoir accès à des grandeurs propres à lClassa méthode.
# On n'a cependant pas besoin de relancer l'inférence
# Puisque les distributions importantes ont été écrites
if  __name__ == "__main__" :
#   run NN_inference_ML.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -N_sample 5 
    T = ctc.Temperature_cst(parser) 
    T.obs_pri_model()
    T.get_prior_statistics()

    X,y,v,m,s = GPC.training_set(T, parser.N_sample)

dict_layers = {"I" : 2,\
               "N1" : 5,\
               "N2" : 5,\
               "N3" : 5,\
               "N4" : 10,\
               "N5" : 10,\
               "N6" : 10,\
#               "N7" : 10,\
#               "N8" : 10,\
#               "N9" : 10,\
#               "N10": 10,\
               "O"  : 1}

#dict_layers = {"I" : 2,\
#               "N1" : 1000,\
#               "N2" : 500,\
#               "N3" : 100,\
#               "N4" : 100,\
#               "N5" : 100,\
#               "N6" : 100,\
#               "N7" : 100,\
#               "N8" : 100,\
#               "N9" : 100,\
#               "N10": 100,\
#               "O"  : 1}

N_hidden_layer = len(dict_layers.keys()) - 2
#nn = NNC.Neural_Network(parser.lr, N_=dict_layers, max_epoch=parser.N_epoch)
###-------------------------------------------------------------------------------
def recentre(xs, X_train_mean, X_train_std):
    """
    This function refocuses the input before prediction. It needs three arguments :
    
    xs              :   The input to refocuses
    X_train_mean    :   The mean vector calculated according to the training set
    X_train_std     :   The standard deviation vector calculated according to the training set   
    
    xs is a vector of shape [N] 
    X_train_mean and X_train_std are vectors of whose shape is [N_features]
    """
    for i in range(np.size(X_train_mean)) :
        xs[i] -= X_train_mean[i]
        if np.abs(X_train_std[i]) > 1e-12 :
            xs[i] /= X_train_std[i]

    return xs
###-------------------------------------------------------------------------------
def build_case(lr, X, y, act, opti, loss, reduce_type, N_=dict_layers, max_epoch=parser.N_epoch, scale=True, verbose=True, **kwargs) :
    # build_case(1e-3, X, y , act="relu", opti="RMS", loss="OLS", decay=0.7, momentum=0.8, max_epoch=1000) marche très bien avec [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # nn_adam_mean = build_case(1e-3, X, y , act="relu", opti="Adam", loss="OLS", decay=0.7, momentum=0.8, max_epoch=2000, reduce_type="mean")
#    nn_adam_mean = build_case(1e-3, X, y , act="relu", opti="Adam", loss="OLS", decay=0.7, momentum=0.8, max_epoch=4000, reduce_type="mean")

#   Ce cas nous donne une erreur totale moyennée  0.00099802657]
    
    nn_obj = NNC.Neural_Network(lr, N_=N_, scaler = "Standard", reduce_type=reduce_type, color=kwargs["color"], verbose=verbose, max_epoch=max_epoch, clip=False, r_parameter=0.5)

    nn_obj.split_and_scale(X, y, shuffle=True, val=False)
    nn_obj.tf_variables()
    nn_obj.layer_stacking_and_act(act)
    nn_obj.def_optimizer("Adam")
    nn_obj.cost_computation(loss)
    nn_obj.case_specification_recap()    
    
    nn_obj.training_phase(1e-3)
    
    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    test_line = range(len(nn_obj.X_test))
    
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
    
    return nn_obj
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
def grid_search(dict_layers):
    lr = 1e-5
    last_costs = dict()
    cost, c_max, cpt, cpt_max = 100000, 100, 0, 5
    while (cost > c_max or np.isnan(cost)==True) and cpt < cpt_max:
        print("Cas {}, lr = {}".format(cpt+1, lr))
        nn = NNC.Neural_Network(lr, N_=dict_layers,\
            max_epoch=parser.N_epoch)
        try :
            build_case(nn,X,y,act="relu", opti="GD", loss="OLS")
            cost = nn.costs[-1]
        except IOError :
            cost = np.nan

        last_costs["lr_{}_cpt_{}".format(lr, cpt)] = cost
        lr /= 10
        cpt +=1
    return last_costs
#-------------------------------------------#
#-------------------------------------------#
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
            x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
            x_s = recentre(x_s, nn_obj.train_mean, nn_obj.train_std)
            x_s = x_s.reshape(-1,1)
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

## Les différents attibuts que les prochaines méthodes vont créer seront utiliser après ouverture d'une session

#print("L\'arbre est construit. Selon le dictionnaire N_: \n{}\n\
#Les matrices sont a priori bien dimensionnées, on vérifie avec les différents poids pour chaque couches: \n{}".format(nn.N_, nn.w_tf_d))
lambda_list = [lambda z: 28, lambda z: 55, lambda z: 15+5*np.cos(np.pi*z)]
body_list = ["28", "55", "15+5cos(piz)"]
    
def repeat(T, nn_obj, N_sample, lambda_lst, body_lst, verbose=False) :
    for T_inf, body in zip(lambda_lst, body_lst) :
        print("lambda = {}, body = {}".format(T_inf, body))
        solver_NN(T, nn_obj, N_sample, T_inf, body, verbose=True)
    
    plt.show()


