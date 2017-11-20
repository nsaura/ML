#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys, warnings

from scipy import optimize as op
import numdifftools as nd
### Problem Constants and modules which will be converted into self when creating the class
##### Modules
#-----------------
def tab_normal(mu, sigma, length) :
    return sigma * np.random.randn(length) + mu, (sigma * np.random.randn(length) + mu).mean() , (sigma * np.random.randn(length) + mu).std()
#-----------------
#-----------------
def h_beta(beta, T_inf, A, N_discr= 33, noise= 'none') :
    err, tol, compteur, compteur_max = 1., 1e-3, 0, 1000
    T_n = map(lambda x : -4*T_inf*x*(x-1), line_z)
    T_nNext = T_n
    B_n = np.zeros((N_discr-2))
    eps_0 = 5.*10**(-4) 
    
    while (np.abs(err) > tol) and (compteur <= compteur_max) :
        if compteur >= 1 :
            T_n = T_nNext
        compteur +=1 
        
        for i in xrange(N_discr-2) :
            B_n[i] = T_n[i] + dt*(beta[i])*eps_0*(T_inf**4 - T_n[i]**4)
        
        T_nNext = np.dot(np.linalg.inv(A), np.transpose(B_n))
        err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne

# We can print the solution to be sure to keep the same result
#    print "convergence et \n: Mean : {:.4f} and std {:.4f}".format(T_nNext.mean(), T_nNext.std())

    if compteur == compteur_max :
        warnings.warn("\x1b[7;1;255mH_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {} \x1b[0m".format(err, tol))
    return T_nNext 
#-----------------

#####--------------------------------------------------------------------------

##### Constants
N_discr = 33
z_init, z_final = 0., 1.
dt, dz =  0.001, np.abs(z_init-z_final)/float(N_discr)

line_z = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]

kappa, h = 0.1, 0.5

T_inf_lst = [i*5 for i in xrange(1, 11)]

## Matrice pour la résolution
A_diag = np.diag(np.transpose([(1+( 2.0)*dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale
M1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Extra inférieure
P1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Extra supérieure

A = A_diag + M1 + P1

np.random.seed(1000) ## Tenir le bruit 
noise = tab_normal(0, 0.1, N_discr-2)[0]

eps_0 = 5.*10**(-4) 
#####--------------------------------------------------------------------------

plt.ion()

cov_obs_dict, cov_prior_dict = dict(), dict() ## dict of covariances for different temperature
cov_prior_prior = dict()
prior_sigma = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]

T_obs_mean = dict()
## We first store the different covariances and the temperature profiles
for i, (T_inf, prior_s) in enumerate(zip(T_inf_lst, prior_sigma)) : 
    #i : xrange(len(T_inf_lst), T_inf = T_inf_lst[i]
    T_obs, T_prior = [], []     
    T_sum = np.zeros((N_discr-2))
    for it in xrange(0,10) :
    
    # Initialize appropriate pathfile for exact solution and modelled (which has discrepancy)
        path_abs = os.path.abspath(os.path.curdir)
        pf_obs = os.path.join(path_abs, 'data', 'obs_T_inf_{}_{}.csv'.format(T_inf, it))
        pf_prior = os.path.join(path_abs, 'data', 'prior_T_inf_{}_{}.csv'.format(T_inf, it))
        
        # Compute covariance from data 
        T_temp = pd.read_csv(pf_obs).get_values()
        T_temp = T_temp.reshape(T_temp.shape[0])
        T_sum += T_temp / float(len(xrange(0,10)))
        T_obs.append(T_temp)
        
        # We conserve the T_disc
        T_disc = pd.read_csv(pf_prior).get_values()
        T_disc = T_disc.reshape(T_disc.shape[0])
        T_prior.append(T_disc)
    
        # Compute prior cov for each prior sigma given in the article
#    TT = np.asarray(np.sum([T_obs[i]/float(len(T_obs)) for i in xrange(len(T_obs))]))
    T_obs_mean[str(T_inf)] = T_sum
    std_mean_obs    =   np.mean(np.asarray([np.std(T_obs[i]) for i in xrange(len(T_obs))]))
    std_mean_prior  =   np.mean(np.asarray([np.std(T_prior[i]) for i in xrange(len(T_prior))]))
        
    cov_obs_dict[str(T_inf)]    =   np.diag([std_mean_obs**2 for j in xrange(N_discr-2)])
    cov_prior_dict[str(T_inf)]  =   np.diag([prior_s**2 for j in xrange(N_discr-2)])
    cov_prior_prior[str(T_inf)] =   np.diag([std_mean_prior**2 for j in xrange(N_discr-2)])

T_inf = 10 ## Begin with that

sT_inf = str(T_inf)
beta_prior = np.asarray([1 for i in xrange(N_discr-2)])

curr_d  =   T_obs_mean[sT_inf] 
cov_m,  cov_prior   =   cov_obs_dict[sT_inf],    cov_prior_dict[sT_inf]

tol_mini, max_iter_mini =   1e-5, 5000
err_mini, compteur_mini =   1., 0 

curr_beta = beta_for_mini = beta_prior

#while err_mini > tol_mini and compteur_mini < max_iter_mini :
compteur_mini += 1 
J = lambda beta : 0.5*  ( 
              np.dot( np.dot(np.transpose(curr_d - h_beta(beta, T_inf, A)),
                np.linalg.inv(cov_m)) , (curr_d - h_beta(beta, T_inf, A) )  )  
            + np.dot( np.dot(np.transpose(beta - beta_prior), 
                np.linalg.inv(cov_prior) ) , (beta - beta_prior) )   
                        ) ## Fonction de coût
                        
curr_h_beta = h_beta(curr_beta, T_inf, A)

#Construction des dérivées pour le problème adjoint
dJ_dT = np.asarray([(curr_h_beta[i] - curr_d[i])/cov_prior[0,0] for i in xrange(N_discr-2)])
dR_dT = np.asarray([ 4*curr_h_beta[i]*curr_beta[i]*eps_0 for i in xrange(N_discr-2) ])

psi      =  - dJ_dT/dR_dT 
lambda_dR_dBeta =  lambda curr_h_beta: np.asarray([-eps_0*(T_inf**4 - curr_h_beta[i]**4) for i in xrange(N_discr-2) ])
dR_dBeta =  lambda_dR_dBeta(curr_h_beta)

lambda_grad_J   =  lambda psi, dR_dB, i : np.asarray(- psi[i] * dR_dB[i] )
Jprime = np.asarray([lambda_grad_J(psi, dR_dBeta, i) for i in xrange(N_discr-2)])


op.fmin_bfgs(J, beta_prior, fprime = Jprime)
#    
#    op.fmin_bfgs(grad_J, 1)
        
    ## Incrémentations
#    curr_beta   = new_beta = curr_beta + alpha_n * P_k  
#    curr_h_beta =   h_beta(new_beta, T_inf, A) # To better understand

err_mini = np.abs(J(new_beta))
print compteur_mini, err_mini

    

