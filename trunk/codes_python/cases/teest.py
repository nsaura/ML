# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

N_discr =50
kappa = 1.
dt = 0.0001

z_init, z_final =   0.0, 1.0
dz = np.abs(z_final - z_init) / float(N_discr)
        
M1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) 
P1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)  
A_diag1 = np.diag(np.transpose([(1 + dt/dz**2*kappa) for i in range(N_discr-2)])) 

A1 = A_diag1 + M1 + P1 
        
M2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) 
P2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)             
A_diag2 = np.diag(np.transpose([(1 - dt/dz**2*kappa) for i in range(N_discr-2)]))

A2 = A_diag2 + M2 + P2 #Construction de la matrice des coefficients

eps_0 = 5.*10**(-4)

def h_beta(beta, T_inf=50, verbose=False) :
#   T_n = list(map(lambda x : -4*T_inf*x*(x-1), self.line_z))
#   Initial condition
    
    sT_inf  =   "T_inf_" + str(T_inf)
    T_n= curr_d

    B_n = np.zeros((N_discr-2))
    T_nNext = T_n
    
    err, tol, compteur, compteur_max = 1., 1e-4, 0, 1000
    if verbose == True :
        plt.figure()
        
    while (np.abs(err) > tol) and (compteur <= compteur_max) :
        if compteur > 0 :
            T_n = T_nNext
        compteur +=1 
        
        T_n_tmp = np.dot(A2, T_n)
        
        for i in range(N_discr-2) :
            try :
                B_n[i] = T_n_tmp[i] + dt*(beta[i])*eps_0*(T_inf**4 - T_n[i]**4)
            except IndexError :
                print ("i = ", i)
                print ("B_n = ", B_n)
                print ("T_n = ", T_n)
                print ("T_N_tmp = ", T_n_tmp)
                raise Exception ("Check")    
                            
        T_nNext = np.dot(np.linalg.inv(A1), B_n)
        
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
        
        if verbose == True and compteur % 5 == 0 :
            print (err)
            plt.plot(line_z, T_nNext, label='tracer cpt %d' %(compteur))
        
        if compteur == compteur_max :
            warnings.warn("\x1b[7;1;255mH_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {} \t \x1b[0m".format(err, tol))
#                time.sleep(2.5)
#        if verbose == True :
#        plt.plot(self.line_z, T_nNext, marker="o", linestyle='none')
#        plt.legend(loc="best", ncol=4)
        if verbose==True :
            print ("Err = {} ".format(err))
            print ("Compteur = ", compteur)
    
    if verbose == True :
        print("H_beta ok")
#        time.sleep(1)
    return T_nNext 
    
def DR_DT(beta, T_inf=50) :
    M1 = np.diag([(N_discr-1)**2 for i in range(N_discr-3)], -1) # Extra inférieure
    P1 = np.diag([(N_discr-1)**2 for i in range(N_discr-3)], +1)  # Extra supérieure
    A_diag1 = -4* np.diag((h_beta(beta, T_inf)**3 * beta * eps_0))-np.diag([2*(N_discr-1)**2 for i in range(N_discr-2) ]) # Diagonale
    result = A_diag1 + M1 + P1
    return  result
    
def DJ_DT(beta) :
    result  =   np.dot(np.linalg.inv(cov_obs), h_beta(beta) - curr_d) 
    return result

def DJ_DBETA(beta):
        return np.dot( np.linalg.inv(cov_pri), beta - beta_prior )

def DR_DBETA(beta):
    return (50.**4 - h_beta(beta)**4) * eps_0

def PSI(beta) :
    return -np.dot(DJ_DT(beta), np.linalg.inv(DR_DT(beta)).T)

def Next_hess(prev_hess_inv, y_nN, s_nN, dim ) :
    
    rho_nN  =   1./np.dot(y_nN.T, s_nN) if np.dot(y_nN.T, s_nN) is not 0 else 1./1e-5
    print  ("rho_nN = {}".format(rho_nN))
        
    Id      =   np.eye(dim)
        
    A1 = Id - rho_nN * s_nN[:, np.newaxis] * y_nN[np.newaxis, :]
    A2 = Id - rho_nN * y_nN[:, np.newaxis] * s_nN[np.newaxis, :]
        
    return np.dot(A1, np.dot(prev_hess_inv, A2)) + (rho_nN* s_nN[:, np.newaxis] * s_nN[np.newaxis, :])
    
p_obs   =  "./a_lire_obs.csv"
p_pri   =  "./a_lire_pri.csv"

cov_obs =   pd.read_csv(p_obs).get_values()
cov_pri =   pd.read_csv(p_pri).get_values()
curr_d  =   pd.read_csv("a_lire_T.csv").get_values()
curr_d  =   curr_d.reshape(N_discr-2)

beta_prior  =   beta_n  =   np.asarray([1 for i in range(N_discr - 2)])
J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - h_beta(beta)),
                    np.linalg.inv(cov_obs)) , (curr_d - h_beta(beta) )  )  
                + np.dot( np.dot(np.transpose(beta - beta_prior), 
                    np.linalg.inv(cov_pri) ) , (beta - beta_prior) )   
                        ) ## Fonction de coût
grad_J = lambda beta : np.dot(PSI(beta), np.diag(DR_DBETA(beta))) + DJ_DBETA(beta)

J_lst, alpha_lst = [], []
sup_g, cpt, cptmax = np.linalg.norm(grad_J(beta_prior), np.inf), 0, 100
H_n = np.eye(48)
g_n = grad_J(beta_prior)

errr = 1.
while np.abs(errr) > 1e-5 :
    if cpt > 0 :
        g_n =   g_nN
        H_n =   H_nN
        beta_n  =   beta_nN

    cpt += 1
    d_n =  np.dot(H_n, g_n)
    print ("descent of {}-eme iteration: \n{}".format(cpt, d_n))
    J_new = lambda alpha : J(beta_n + alpha * d_n)
    GJ_new= lambda alpha : grad_J(beta_n + alpha * d_n)
    
    alpha = op.linesearch.line_search_wolfe2(J_new, GJ_new, beta_n, d_n)[0]

    print ("alpha = {}".format(alpha))

    
    beta_nN = beta_n + alpha*d_n
    print ("beta {}-eme iteration: \n{}".format(cpt, beta_nN))
    
    g_nN    =   grad_J(beta_nN)
    s_nN    =   beta_nN - beta_n
    y_nN    =   g_nN  - g_n
    H_nN    =   Next_hess(H_n, y_nN, s_nN, 48)
    errr    =   np.linalg.norm(grad_J(beta_nN), np.inf)

