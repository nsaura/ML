#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys


plt.ion()

T_recup, T_mod = [], [] 
cov_m_lst, cov_prior_lst = [], [] 

T_inf_lst = [i*5 for i in xrange(1, 11)]

N_discr = 33
dt = 0.001

np.random.seed(500) # To keep the same random generator
z_init, z_final = 0., 1.
N_discr = 33
kappa=0.1

dz = np.abs(z_init-z_final)/float(N_discr)

line_z = np.linspace(0.,1.,N_discr)[1:N_discr-1]

prior_sigma = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]

M1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Extra inférieure
P1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Extra supérieure
A_diag = np.diag(np.transpose([(1+( 2.0)*dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

A = A_diag + M1 + P1 #Construction de la matrice des


for i, (T_inf, prior_s) in enumerate(zip(T_inf_lst, prior_sigma)) : 
    #i : xrange(len(T_inf_lst), T_inf = T_inf_lst[i]
    
    # Initialize appropriate pathfile for exact solution and modelled (which has discrepancy)
    path_abs = os.path.abspath(os.path.curdir)
    pathfile_ex = os.path.join(path_abs, 'data', 'T_inf_{}.csv'.format(T_inf))
    pathfile_mod = os.path.join(path_abs, 'data', 'prior_T_inf_{}.csv'.format(T_inf))
    
    # Compute covariance from data 
    T_temp = pd.read_csv(pathfile_ex).get_values()
    T_temp = T_temp.reshape(T_temp.shape[0])
    cov_m_lst.append(np.diag([T_temp.std()**2 for j in xrange(N_discr-2)]))
    T_recup.append(T_temp)
    
    # We conserve the T_disc
    T_disc = pd.read_csv(pathfile_mod).get_values()
    T_disc = T_disc.reshape(T_disc.shape[0])
    T_mod.append(T_disc)
    # Compute prior cov for each prior sigma given in the article
    cov_prior_lst.append(np.diag([prior_s**2 for j in xrange(N_discr-2)]))

verbose = False
if verbose == True :
    for i, (T_inf, T, TT) in enumerate(zip(T_inf_lst,T_recup, T_mod)) :
        if i%3 == 0 :
            plt.plot(line_z, T, label="EX_T_inf_: {}".format(T_inf))
            plt.plot(line_z, TT,label="DISC_T_inf_{}".format(T_inf), marker='s', linestyle='none', markerfacecolor = 'none' )

    plt.legend(loc='best')
    
#[[int(i) for i in line.split()] for line in data]

## Construction fonction et d'autres variables à optimiser
beta_prior = [1 for i in xrange(N_discr-2)]
for i, T_inf in enumerate(T_inf_lst) :
    cov_m,  cov_beta    =   cov_m_lst[i],   cov_prior_lst[i]
    d_obs,  h_beta      =   T_recup[i],     T_mod[i]
    
    err_beta, eps_beta = 1., 0.1
    compteur_beta = 0
    
    while err_beta > eps_beta and compteur_beta < 1000  :
        
        curr_beta = beta if compteur_beta is not 0 else beta_prior
        
        err, compteur, tol = 1., 0 , 1e-2
        
        T_n_beta = map(lambda x : -4*T_inf*x*(x-1), line_z)
        T_nNext_beta = T_n_beta
        
        while (np.abs(err) > tol) and (compteur < 800) :
            if compteur > 0 :
                T_n_beta = T_nNext_beta
            compteur += 1
            B_n_beta = T_n_beta
            
            for j in xrange(1, N_discr-2) :
                B_n_beta[i] = curr_beta[i] * T_n_beta[i]+dt*5*10**(-4)*(T_inf**4-T_n_beta[i]**4)
            T_nNext_beta = np.dot(np.linalg.inv(A), np.transpose(B_n_beta))
            
            err = np.linalg.norm(T_nNext_beta-T_n_beta, 2)
            ## Exit if -tol < err < tol
        # We now have an solution approximated solution 
        # for the termal problem according to a certain beta
        
        # We go one step further implementing steepest descent algorithm
        # Line search https://en.wikipedia.org/wiki/Line_search
        
        misfit = lambda Beta : 0.5*( ( np.dot( np.dot(np.transpose(d_obs - T_nNext_beta),np.linalg.inv(cov_m)) , (d_obs - T_nNext_beta) ) ) + np.dot( np.dot(np.transpose(Beta - beta_prior), np.linalg.inv(cov_beta) ) , (Beta - beta_prior) )  )
        
        XX = np.arange(-1,3,2/float(N_discr-2))
        Grad = np.gradient(map(misfit, XX), 2/float(N_discr-2)) 

#        interp_der  =   np.polyfit(XX, Grad, 4)
#        polyn_der   =   np.poly1d(interp_der)(XX)
#        
        F = [(T_nNext_beta[i] - d_obs[i])/()]
        err_beta = 0
         
        
        
        
         
        
