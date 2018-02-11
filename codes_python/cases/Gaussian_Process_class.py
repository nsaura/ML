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

#class GaussianProcess(ctc.Temperature_cst) : ## Héritage
#    def __init__(self, parser):
#        ctc.Temperature_cst.__init__(self, parser)
#        print self.parser

#if __name__ == "__main__" :
#    import class_functions_aux as cfa
#    parser = cfa.parser()
#    g = GaussianProcess(parser)
    
#def get_variances(T):
#    variances = dict()
#    for t in T.T_inf_lst :
#        sT_inf = "T_inf_" + str(t)  # Clé pour les dictionnaires de l'objet T
#        filename = "adj_post_cov_full_%s.csv" %(sT_inf)
#        filepath = osp.join(osp.join(T.parser.datapath, "post_cov"), filename)
#        print filepath
#        
#        # On prend la variance à partir de la covariance
#        variances[sT_inf] = np.diag(T.pd_read_csv(filepath)) 
#    
#    return variances

def cov_function(x1, x2, h) :
    phi = np.exp(-np.linalg.norm(x1-x2, 2)**2/h**2)
    
    return phi

def training_set(T, N_sample):
    """
    Dans cette fonction on construit le tableau X_train en construisant un tableau dont la première colonne comprendra
    la température T_inf et les colonnes successives seront composées des valeurs des températures finales pour chaque
    point de discrétisation.
    La taille de ce tableau sera donc (N_train_points, 1 + (N_discr-2)).
    N_discr est définie lors de l'inférence. 
    N_train_points dépendra du nombre de tirages de température voulu pour une distribution de béta donnée
    
    Arguments :
    -----------
    T : l'objet de la classe Classe_Temp_Cst dans laquelle l'inférence Bayesienne est faite
    N_sample : Nombre de tirages que l'on veut faire dans les distributions de beta_map 
    
    Returns :
    ----------
    X_train 
    Y_train
    """
    # Initialisation des tableaux
    X_train = np.zeros((1, 2))  # Pour l'instant on essaye ça
    Y_train = np.zeros((1, 1))      # Pour l'instant on essaye ça
    var = np.zeros((N_sample*T.N_discr-2, N_sample*T.N_discr-2))
    
    variances = []
    
    for t in T.T_inf_lst :
        sT_inf = "T_inf_" + str(t)  # Clé pour les dictionnaires de l'objet T
                
        # On construit la distribution de beta autout de betamap
        # On n'a pas encore construit la BONNE COVARIANCE, juste pour le test
        distrib_bmap = lambda s : T.bfgs_adj_bmap[sT_inf] + np.dot(T.bfgs_adj_cholesky[sT_inf], s)
        
        # Calcule de la variance autour de beta map 
        for i in range(N_sample) :   
            # liste de valeurs aléatoires issues d'une distribution gaussienne centrée réduite 
            s_curr = T.tab_normal(0,1,T.N_discr-2)[0] 
            
            T_finale = T.h_beta(distrib_bmap(s_curr), t) # Calcule de T_finale à partir du tirage en béta
            # On empile la ligne "line" sous la dernière ligne du X_train
            for j,d in zip(T_finale,distrib_bmap(s_curr)) :
                line = np.append(t, j) ## Ligne comprenant T_inf et le champ final de température
                X_train = np.block([[X_train], [line]]) 
                
                Y_train = np.block([[Y_train], [d]]) 
                
                variances.append(np.std(distrib_bmap(s_curr))**2)
            # On empile la ligne "line" sous la dernière ligne du Y_train
    
    X_train =   np.delete(X_train, 0, axis=0)        # On enlève les lignes de zéros
    Y_train =   np.delete(Y_train, 0, axis=0)        # On enlève les lignes de zéros
    
    return X_train, Y_train, variances
            
def maximize_LML(T, N_sample): #Rajouter variances

    N = N_sample*(T.N_discr-2)*len(T.T_inf_lst)
    X_train, Y_train, var = training_set(T, N_sample)
    var_mat = var*np.eye(N)
    
    h_curr = 1.5

    ### On définit les lambda
    phi = lambda h : np.asarray([[np.exp(- np.linalg.norm(X_train[i] - X_train[j], 2)**2/h**2)\
                                  for i in range(N)] for j in range(N)])
    # On inversera L plutot que phi (Voir Ramussen and Williams)
#    L = lambda h : np.linalg.cholesky(phi(h) + var_mat) 
    
    # max(f) = -min(-f)
    m_LML = lambda h : (-1)*(-0.5)*(np.log(np.linalg.det(phi(h) + var_mat)) +\
                            np.dot(np.dot(Y_train.T, np.linalg.inv(phi(h) + var_mat)), Y_train)[0,0]+\
                            N_sample*np.log(2*np.pi) )

    ### On minimise -LML
    # On calcule -LML  
    val = op.minimize_scalar(m_LML)
    h_op = val.x # On garde l'antécédent ou argument
    print h_op, val.success
    
    phi_op = phi(h_op)
    
    phi_var_inv = np.linalg.inv(phi_op + var_mat)
    
    return h_op, phi_var_inv 

def ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s) :
    """
    h ici est supposée optimizée pour maximiser la correspondance entre les prédictions sur Y_train et la fonction de covariance phi
    x_s l'entrée dont on veut prédire la sortie
    """
    N = N_sample*(T.N_discr-2)*len(T.T_inf_lst)
    if len(np.shape(var)) == 1 :
        var *= np.eye(N)
    
    var_mat = var * np.eye(N)
    phi_vec_s = np.asarray([np.exp(-np.linalg.norm(x_s - X, 2)**2 / h_op**2) for X in X_train]) ## Vecteur de taille N
    
    ## Vecteur de taille N 
    alpha = np.dot(phi_var_inv, Y_train).reshape(N)
    
    beta_ML = alpha.dot(phi_vec_s) 
    sigma_ML = 1 - phi_vec_s.dot(phi_var_inv.dot(phi_vec_s))

    return (beta_ML, sigma_ML)
    

def test(T, N_sample, T_inf) :
    X_train, Y_train, var = training_set(T, N_sample)
    h_op, phi_var_inv = maximize_LML(T, N_sample)
    
    sT_inf = "T_inf_%d" %(T_inf)
    Tdistrib = T.h_beta(T.bfgs_adj_bmap[sT_inf] + T.bfgs_adj_cholesky[sT_inf].dot(T.tab_normal(0,1,T.N_discr-2)[0]), T_inf)
    var_moy = np.mean(var)
    
    X_test = np.zeros((1,2))
    for t in Tdistrib :
        X_test = np.block([[X_test], [T_inf, t]])    
    X_test = np.delete(X_test, 0, axis=0)    
    beta, sigma = [],[] 
    
    for x in X_test :
        res = ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x)
        beta.append(res[0])
        sigma.append(res[1] + var_moy)
    
    beta_span = []
    beta_span.append([beta[i] + sigma[i] for i in range(len(beta))])
    beta_span.append([beta[i] - sigma[i] for i in range(len(beta))])

    T_min = T.h_beta(beta_span[0], T_inf) 
    T_max = T.h_beta(beta_span[1], T_inf)   
        
    plt.figure("Temperature beta learning")
    plt.plot(T.line_z, T.h_beta(beta, T_inf), label="Learning N_sample %d" %(N_sample), marker='o', c='r', fillstyle = 'none', linestyle='None')
    plt.plot(T.line_z, T.h_beta(T.bfgs_adj_bmap[sT_inf], T_inf), label="Inference" , marker='+', color='green', fillstyle = 'none', linestyle='None')
    plt.plot(T.line_z, Tdistrib, label="True", color='k', linestyle='--')
    plt.fill_between(T.line_z, T_min, T_max, facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="grey")
    plt.legend(loc="best")
    plt.show()
    
    plt.figure("Comparaison Beta learning VS inference")
    plt.plot(T.line_z, beta, label= "learning")
    plt.plot(T.line_z, T.bfgs_adj_bmap[sT_inf], label="inference")
    plt.legend()
    plt.show()
    
    return beta, sigma

def True_Temp(T, T_inf, body) :
    """
    T_inf doit avoir être un type lambda. Boucle conditionnelle qui check ça
    """
    T_inf_map = np.asarray(map(T_inf, T.line_z))
    T_n_obs =  list(map(lambda x : -4*T_inf_map[len(T.line_z)/2]*x*(x-1), T.line_z) )
    T_nNext_obs = T_n_obs

    B_n_obs     =   np.zeros((T.N_discr-2, 1))
    T_n_obs_tmp =   np.zeros((T.N_discr-2, 1))
    tol ,err_obs, compteur = 1e-4, 1.0, 0 
    
    while (np.abs(err_obs) > tol) and (compteur < 800) :
        if compteur > 0 :
            T_n_obs = T_nNext_obs
        compteur += 1
        T_n_obs_tmp = np.dot(T.A2, T_n_obs)

        for i in range(T.N_discr-2) :
            B_n_obs[i] = T_n_obs_tmp[i] + T.dt*  (
            ( 10**(-4) * ( 1.+5.*np.sin(3.*T_n_obs[i]*np.pi/200.) + 
            np.exp(0.02*T_n_obs[i]) + T.lst_gauss[0][i] ) ) *(T_inf_map[i]**4 - T_n_obs[i]**4)
             + T.h * (T_inf_map[i]-T_n_obs[i])      ) 

        T_nNext_obs = np.dot(np.linalg.inv(T.A1), B_n_obs)
        err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2)
    
    print ("Calculus with {} completed. Convergence status :".format(body))
    print ("Err_obs = {} ".format(err_obs))    
    print ("Iterations = {} ".format(compteur))
    
    plt.figure("T_inf : {}".format(body)) 
    plt.plot(T.line_z, T_nNext_obs, label="T_inf={}".format(body))
    plt.legend()
    
    return T_nNext_obs
    
def True_Beta(T, T_inf) :
    """
    Pour comparer beta_true et beta_ML
    """
    t1 = np.asarray([ 1./T.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i])) *10**(-4) for i in range(T.N_discr-2)])
    t2 = np.asarray([T.h / T.eps_0*(T_inf[i] - T[i])/(T_inf[i]**4 - T[i]**4)  for i in range(N_discr-2)]) 

#*************#
#*************#
#*************#

def solver_ML(T, N_sample, T_inf, body):
    X_train, Y_train, var = training_set(T, N_sample)
    h_op, phi_var_inv = maximize_LML(T, N_sample)
    
    T_inf_old = T_inf
    T_inf = map(T_inf, T.line_z) 
    
    var_moy = np.mean(var)
    
    T_n = T_nmspan = T_nMspan = np.asarray(map(lambda x : -4*T_inf[(T.N_discr-2)/2]*x*(x-1), T.line_z) )
    beta_span, beta = [[0], [0]], []
    sigma = []

    for j,t in enumerate(T_n) :
        x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
        res = ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s)
        beta.append(res[0])
        sigma.append(res[1] + var_moy )
    beta_span[0] = [beta[i] - sigma[i] for i in range(len(beta))]
    beta_span[1] = [beta[i] + sigma[i] for i in range(len(beta))]
    
    B_n = np.zeros((T.N_discr-2, 1))
    B_nmspan   =   np.zeros((T.N_discr-2, 1)) # Calcul des incertitudes (m pour min)
    B_nMspan   =   np.zeros((T.N_discr-2, 1)) # Calcul des incertitudes (M pour max)
    
    beta_n = np.asarray(beta)
    
    T_n_tmp = T_mn_tmp = T_Mn_tmp = np.zeros((T.N_discr-2, 1))
    tol ,err, compteur, cmax = 1e-4, 1.0, 0, 800 
    
    while (np.abs(err) > tol) and (compteur <= cmax) :
        if compteur > 0 :
            T_n = T_nNext
            T_nmspan = T_nmNext
            T_nMspan = T_nMNext

            beta_n = beta_nNext
            beta_span[0] = [beta[i] - sigma[i] for i in range(len(beta))]
            beta_span[1] = [beta[i] + sigma[i] for i in range(len(beta))]
 
            beta, sigma = [], []
        compteur +=1 
            
        T_n_tmp = np.dot(T.A2, T_n)
        T_mn_tmp = np.dot(T.A2, T_nmspan)
        T_Mn_tmp = np.dot(T.A2, T_nMspan)
        
        for i in range(T.N_discr-2) :
#            try :
            B_n[i]      = T_n_tmp[i] + T.dt*(beta_n[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
            B_nmspan[i] = T_mn_tmp[i] + T.dt*(beta_span[0][i])*T.eps_0*(T_inf[i]**4 - T_nmspan[i]**4)
            B_nMspan[i] = T_Mn_tmp[i] + T.dt*(beta_span[1][i])*T.eps_0*(T_inf[i]**4 - T_nMspan[i]**4)
          
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        T_nmNext = np.dot(np.linalg.inv(T.A1), B_nmspan)
        T_nMNext = np.dot(np.linalg.inv(T.A1), B_nMspan)
        
        for j,t in enumerate(T_n) :
            x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
            res =  ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s)
            beta.append(res[0])
            sigma.append(res[1] + var_moy)
        beta_nNext = np.asarray(beta)

        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
    
    print ("Calculus with {} completed. Convergence status :".format(body))
    print ("Err_obs = {} ".format(err))    
    print ("Iterations = {} ".format(compteur))

    T_true = True_Temp(T, T_inf_old, body)
    
    n = T.N_discr-2
    T_true = T_true.reshape(n)
    T_nNext = T_nNext.reshape(n)
    T_nmNext= T_nmNext.reshape(n)
    T_nMNext= T_nMNext.reshape(n)
    
    plt.figure("Beta_True vs Beta_ML; N_sample = {}; T_inf = {}".format(N_sample, body)) 
    plt.plot(T.line_z, T_true, label="True T_field for T_inf={}".format(body), c='k', linestyle='--')
    plt.plot(T.line_z, T_nNext, label="ML T_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
    plt.fill_between(T.line_z, T_nmNext, T_nMNext, facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="grey")
    plt.legend()
    
    return T_true, T_nNext
        
    


