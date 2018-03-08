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

def training_set(T, N_sample, scale=False, shuffle=False):
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
    X = np.zeros((1, 2))      # Pour l'instant on essaye ça
    y = np.zeros((1,))      # Pour l'instant on essaye ça
    var = np.zeros((N_sample*T.N_discr-2, N_sample*T.N_discr-2))
    
    mean, std = 0, 0
    
    variances = []
    bmap_fields = dict()
    chol_fields = dict()
    
    for t in T.T_inf_lst :
        sT_inf = "T_inf_" + str(t)  # Clé pour les dictionnaires de l'objet T
        
        bmap_ = "adj_bfgs_beta_%s_N%d_cov%s.csv" %(sT_inf, T.N_discr-2, T.cov_mod)
        bmap_ = osp.join("./data/matrices",bmap_)
        
        chol_ = "adj_bfgs_cholesky_%s_N%d_cov%s.csv" %(sT_inf, T.N_discr-2, T.cov_mod)
        chol_ = osp.join("./data/matrices",chol_)
            
        if osp.exists(bmap_) == False or osp.exists(chol_) == False :
            sys.exit("{} or {} or both don't exist. Check".format(bmap_, chol_))
        
        bmap_fields[sT_inf] = T.pd_read_csv(bmap_)
        chol_fields[sT_inf] = T.pd_read_csv(chol_)

        # On construit la distribution de beta autout de betamap
        distrib_bmap = lambda s : bmap_fields[sT_inf] + np.dot(chol_fields[sT_inf], s)
        
        # Calcule de la variance autour de beta map 
        for i in range(N_sample) :   
            # liste de valeurs aléatoires issues d'une distribution gaussienne centrée réduite 
            s_curr = T.tab_normal(0,1,T.N_discr-2)[0] 
            
            T_finale = T.h_beta(distrib_bmap(s_curr), t) # Calcule de T_finale à partir du tirage en béta
            # On empile la ligne "line" sous la dernière ligne du X_train
            for j,d in zip(T_finale,distrib_bmap(s_curr)) :
                line = np.append(t, j) ## Ligne comprenant T_inf et le champ final de température
                X = np.block([[X], [line]]) 
                y = np.block([[y], [d]]) 
                
                variances.append(np.std(distrib_bmap(s_curr))**2)
            # On empile la ligne "line" sous la dernière ligne du Y_train
    
    X =   np.delete(X, 0, axis=0)        # On enlève les lignes de zéros
    y =   np.delete(y, 0, axis=0)        # On enlève les lignes de zéros
    variances = np.asarray(variances)
    
    if shuffle == True :
        permute_indices = np.random.permutation(np.arange(len(y)))
        X = X[permute_indices]
        y = y[permute_indices]
        variances = variances[permute_indices]
    
    if scale == True :
        std = X.std(axis=0)
        mean = X.mean(axis=0)
    
        X[:,0] = X[:,0] - mean[0]
        
        if np.abs(std[0]) > 1e-12 : X[:,0] /= std[0]
        
        X[:,1]  = (X[:,1] - mean[1]) / std[1]
        X[:,1]  = (X[:,1] - mean[1]) / std[1]
        
    return X, y, variances, mean, std
#-------------------------------------------#            
def maximize_LML(T, X_train, Y_train, var, N_sample, h_curr = 2.): #Rajouter variances
    N = N_sample*(T.N_discr-2)*len(T.T_inf_lst)
    var_mat = var*np.eye(N)
    
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
#-------------------------------------------#
def ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s) :
    """
    h ici est supposée optimizée pour maximiser la correspondance entre les prédictions sur Y_train et la fonction de covariance phi
    x_s l'entrée dont on veut prédire la sortie
    """
    N = N_sample*(T.N_discr-2)*len(T.T_inf_lst)
    
    var_mat = var * np.eye(N)
    phi_vec_s = np.asarray([np.exp(-np.linalg.norm(x_s - X, 2)**2 / h_op**2) for X in X_train]) ## Vecteur de taille N
    
    ## Vecteur de taille N 
    alpha = np.dot(phi_var_inv, Y_train).reshape(N)
    
    beta_ML = alpha.dot(phi_vec_s) 
    sigma_ML = 1 - phi_vec_s.dot(phi_var_inv.dot(phi_vec_s))

    return beta_ML, sigma_ML
#-------------------------------------------#
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
    
#    plt.figure("T_inf : {}".format(body)) 
#    plt.plot(T.line_z, T_nNext_obs, label="T_inf={}".format(body))
#    plt.legend()
    
    return T_nNext_obs
#-------------------------------------------#    
def True_Beta(T, temp, T_inf) :
    """
    Pour comparer beta_true et beta_ML
    T_inf est une liste
    """
    t1 = np.asarray([ 1./T.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * temp[i]) + np.exp(0.02*temp[i])) *10**(-4) for i in range(T.N_discr-2)])
    t2 = np.asarray([T.h / T.eps_0*(T_inf[i] - temp[i])/(T_inf[i]**4 - temp[i]**4)  for i in range(T.N_discr-2)]) 
    
    return t1 + t2
#-------------------------------------------#
def beta_to_T(T, beta, T_inf, body) :
    T_n = np.asarray(map(lambda x : 0., T.line_z) )
    B_n = np.zeros((T.N_discr-2))
    T_nNext = T_n
        
    err, tol, compteur, compteur_max = 1., 1e-6, 0, 10000
        
    while (np.abs(err) > tol) and (compteur <= compteur_max) :
        if compteur > 0 :
            T_n = T_nNext
        compteur +=1 
        T_n_tmp = np.dot(T.A2, T_n)
        for i in range(T.N_discr-2) :
            B_n[i] = T_n_tmp[i] + T.dt*(beta[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
                            
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
        
        if body.split("_")[-1] == "min" :
            print("compteur = {}".format(compteur))
            print("T_nNext = \n{}".format(T_nNext))
            print("erreur = {}".format(err))
    
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Iterations = {} ".format(compteur))
    return T_nNext 
#------------------------------------#
def T_to_beta(T, X_train, Y_train, var, mean, std, phi_var_inv, h_op, N_sample, T_inf, body, scale=True):
    T_n = np.asarray(map(lambda x : 0, T.line_z) )
    beta, sigma =  [], []
    
    var_moy = np.mean(var)
    
    for j,t in enumerate(T_n) :
        x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
        if scale == True :
            x_s[0] -= mean[0]
            x_s[1] -= mean[1]
            x_s[1] /= std[1]
            
            if np.abs(std[0]) > 1e-12 :
                x_s[0] /= std[0]
                
        res = ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s)
        beta.append(res[0])
        sigma.append(res[1] + var_moy )

    beta_n = np.asarray(beta)
    B_n = np.zeros((T.N_discr-2))
    T_n_tmp = np.zeros((T.N_discr-2))

    tol, compteur, cmax = 1e-4, 0, 6000
    err = err_beta = 1.
    
    while (np.abs(err) > tol) and (compteur <= cmax) :
        if compteur > 0 :
            beta_n = beta_nNext
            T_n = T_nNext
        compteur +=1 
            
        T_n_tmp = np.dot(T.A2, T_n)
        
        for i in range(T.N_discr-2) :
            B_n[i]      = T_n_tmp[i] + T.dt*(beta_n[i])*T.eps_0*(T_inf[i]**4 - T_n[i]**4)
          
        T_nNext = np.dot(np.linalg.inv(T.A1), B_n)
        
        beta, sigma = [], []
        for j,t in enumerate(T_n) :
            x_s = np.array([T_inf[j], t]) ## T_inf, T(z)
            if scale == True :
                x_s[0] -= mean[0]
                x_s[1] -= mean[1]
                x_s[1] /= std[1]

                if np.abs(std[0]) > 1e-12 :
                    x_s[0] /= std[0]
                    
            res =  ML(T, X_train, Y_train, var, phi_var_inv, h_op, N_sample, x_s)
            beta.append(res[0])
            sigma.append(res[1] + var_moy)
        
        beta_nNext = np.asarray(beta)
        if compteur % 20 == 0 :
            print("État : cpt = %d, err = %f" %(compteur, err))
#        print ("Iteration {}".format(compteur))
#        print ("beta.shape = {}".format(np.shape(beta)))
#        print ("beta_nNext.shape = {}".format(np.shape(beta_nNext)))        
#        
#        print("T_curr shape ={}".format(T_nNext.shape))
        
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
        err_beta = np.linalg.norm(beta_nNext - beta_n, 2)    
    
    print ("Calculs complétés pour {}. Statut de la convergence :".format(body))
    print ("Erreur sur la température = {} ".format(err))    
    print ("Erreur sur beta = {} ".format(err_beta))    
    print ("Iterations = {} ".format(compteur))
    
    return T_nNext, beta_nNext, sigma
#-------------------------------------------#
def solver_ML(T, X, y, var, mean, std, h_op, phi_var_inv, N_sample, T_inf, body, verbose = False):
#    X_train, Y_train, var = training_set(T, N_sample)
    T_inf_lambda = T_inf
    T_inf = map(T_inf, T.line_z) 

    T_ML, beta_ML, sigma_ML = T_to_beta(T, X, y, var, mean, std, phi_var_inv, h_op, N_sample, T_inf, body)

    T_true = True_Temp(T, T_inf, body)

    n = T.N_discr-2
    T_true = T_true.reshape(n)
    T_ML = T_ML.reshape(n)
    T_max = beta_to_T(T, beta_ML + sigma_ML, T_inf, body+"_max")
    T_min = beta_to_T(T, beta_ML - sigma_ML, T_inf, body+"_min")
    T_base = beta_to_T(T, T.beta_prior, T_inf, body+"_base")
    #    T_nmNext= T_nmNext.reshape(n)
    #    T_nMNext= T_nMNext.reshape(n)
    
    GP_out = dict()
    GP_out["GP_T_ML"]       =   T_ML
    GP_out["GP_T_ML_max"]   =   T_max
    GP_out["GP_T_ML_min"]   =   T_min
    GP_out["GP_beta_ML"]    =   beta_ML    
    
    GP_out["T_true"]    =   T_true
    GP_out["T_base"]    =   T_base

    if verbose == True :
        plt.figure("Beta_True vs Beta_ML; N_sample = {}; T_inf = {}".format(N_sample, body)) 
        plt.plot(T.line_z, T_true, label="True T_field for T_inf={}".format(body), c='k', linestyle='--')
        plt.plot(T.line_z, T_ML, label="ML T_field".format(body), marker='o', fillstyle='none', linestyle='none', c='r')
        plt.plot(T.line_z, T_base, label="Base solution", c='green')
        
        plt.fill_between(T.line_z, T_min, T_max, facecolor= "1", alpha=0.7,\
                        interpolate=True, hatch='/', color="grey", label="Span")
        plt.legend()
    
    return GP_out
        
