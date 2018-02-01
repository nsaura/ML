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

def gaussian(T, phi, N_sample):
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
    
    for t in T.T_inf_lst :
        sT_inf = "T_inf_" + str(t)  # Clé pour les dictionnaires de l'objet T
        
        # On construit la distribution de beta autout de betamap
        # On n'a pas encore construit la BONNE COVARIANCE, juste pour le test
        distrib_bmap = lambda s : T.bfgs_adj_bmap[sT_inf] + np.dot(T.bfgs_adj_cholesky[sT_inf], s)
        
        for i in range(N_sample) :   
            s_curr = T.tab_normal(0,1,T.N_discr-2)[0] # liste de valeurs aléatoires issues d'une distribution gaussienne centrée réduite 
            T_finale = T.h_beta(distrib_bmap(s_curr), t) # Calcule de T_finale à partir du tirage en béta
        
            line = np.append(t, T_finale) ## Ligne comprenant T_inf et le champ final de température
#            print "X_line = {}".format(line)
            X_train = np.block([[X_train], [line]]) # On empile la ligne "line" sous la dernière ligne du X_train
        
            line = distrib_bmap(s_curr)
#            print "Y_line = {}".format(line)
            Y_train = np.block([[Y_train], [line]]) # On empile la ligne "line" sous la dernière ligne du Y_train
    
    X_train = np.delete(X_train, 0, axis=0)        # On enlève les lignes de zéros
    Y_train = np.delete(Y_train, 0, axis=0)        # On enlève les lignes de zéros
    
    return X_train, Y_train
    
    
def maximize_LML(X_train, Y_train):
    h_curr = 1.5
    N = X_train.shape[0]
    phi = lambda h : np.asarray([[np.exp(- np.linalg.norm(X_train[i] - X_train[j], 2)/h**2) for i in range(N)] for j in range(N)])
    phi_curr = phi(h_curr)
        
    return phi_curr
    
