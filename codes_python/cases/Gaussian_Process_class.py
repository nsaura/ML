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
    X_train = np.zeros((1, 1 + T.N_discr-2))  # Pour l'instant on essaye ça
    Y_train = np.zeros((1, T.N_discr-2))      # Pour l'instant on essaye ça
    var = np.zeros((1, T.N_discr-2))
    
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
        
            line = np.append(t, T_finale) ## Ligne comprenant T_inf et le champ final de température
            # On empile la ligne "line" sous la dernière ligne du X_train
            X_train = np.block([[X_train], [line]]) 
            
            line = distrib_bmap(s_curr)
            # On empile la ligne "line" sous la dernière ligne du Y_train
            Y_train = np.block([[Y_train], [line]]) 
            
            variances.append(np.std(distrib_bmap(s_curr))**2)
            
    X_train =   np.delete(X_train, 0, axis=0)        # On enlève les lignes de zéros
    Y_train =   np.delete(Y_train, 0, axis=0)        # On enlève les lignes de zéros
    
    
    return X_train, Y_train, variances
            
def maximize_LML(T, N_sample): #Rajouter variances
    X_train, Y_train, var = training_set(T, N_sample)
    
    h_curr = 1.5
    var_mat = np.zeros((T.N_discr-2, T.N_discr-2))

    phi = lambda h : np.asarray([[np.exp(- np.linalg.norm(X_train[i] - X_train[j], 2)**2/h**2)\
                                  for i in range(N_sample)] for j in range(N_sample)])
    
    L = lambda h : np.linalg.cholesky(phi(h) + variances*np.eye(N_samples*len(T.T_inf_lst))) # On inversera L plutot que phi
    ## On minimise -LML
    # max(f) = -min(-f)
    # On calcule -LML  
    m_LML = lambda h : 0.5*( np.log(np.linalg.det(phi(h) + var_mat)) +\
                            np.dot(np.dot(Y_train[np.newaxis, :], np.linalg.inv(L(h))), Y_train[:, np.newaxis])[0,0] +\
                            N*np.log(2*np.pi) )
    
    jac_m_LML = nd.Jacobian(m_LML)
    val = op.minimize(m_LML, h_curr, jac=jac_m_LML, method="Newton-CG")
    val_h = val.x
    print val_h
    
