#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys, warnings

from scipy import optimize as op

from itertools import cycle
import matplotlib.cm as cm


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
def true_beta(T, noise ,T_inf, N_discr, h=0.5, eps_0 = 5.*10**(-4)) :  
    
    beta = [ 1./eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i]) + noise[i] ) *10**(-4) + h/eps_0*(T_inf - T[i])/(T_inf**4 - T[i]**4)  for i in xrange(N_discr-2)]
    return np.asarray(beta)
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

T_inf = 50 ## Begin with that


#for T_inf in T_inf_lst :
sT_inf = str(T_inf)
beta_prior = np.asarray([1 for i in xrange(N_discr-2)])

curr_d  =   T_obs_mean[sT_inf] 
cov_m,  cov_prior   =   cov_obs_dict[sT_inf],    cov_prior_dict[sT_inf]

J = lambda beta : 0.5*  ( 
              np.dot( np.dot(np.transpose(curr_d - h_beta(beta, T_inf, A)),
                np.linalg.inv(cov_m)) , (curr_d - h_beta(beta, T_inf, A) )  )  
            + np.dot( np.dot(np.transpose(beta - beta_prior), 
                np.linalg.inv(cov_prior) ) , (beta - beta_prior) )   
                        ) ## Fonction de coût

optimizer_dict, optimizer_betamap, optimizer_vec = dict(), dict(), dict()

hess_dict = dict()

#    opti_bfgs[sT_inf] = op.fmin_bfgs(J, beta_prior)
# BFGS : quasi Newton method that approximate the Hessian on the fly
optimizer_dict["bfgs" + sT_inf]     =   op.minimize(J, beta_prior, method="BFGS")

# Low rank hessian version sitd between BFGS and CG 
optimizer_dict["l-bfgs-b" + sT_inf] =   op.minimize(J, beta_prior, method="L-BFGS-B")

# Conjugate gradient use twe last value of the gradient to reduce jumps refering to the classical gradient descent algorithm

## Les méthodes suivantes ne permettent pas de calculer la hessienne
#optimizer_dict["CG" + sT_inf]       =   op.minimize(J, beta_prior, method="CG")
#    
## Newton method : based on the calculus of the gradient and the Hessian.
##optimizer_dict["Newton-CG" + sT_inf] =  op.minimize(J, beta_prior, method="Newton-CG")   

## Nelder-mead : robust to noise usefull for exp. data point. Generalization of dichotomy
#optimizer_dict["Nelder-Mead" + sT_inf]       =   op.minimize(J, beta_prior, method="Nelder-Mead")

for item in zip(optimizer_dict.items()) :
    optimizer_betamap[item[0][0]] = item[0][1].x

#hess_inv_dict[sT_inf] = opti_.hess_inv
        

#beta_opti = op.fmin_bfgs(J, beta_prior)

pb_lst = []
s= np.asarray([i for i in tab_normal(0,1,N_discr-2)[0]])
for k,v in zip(optimizer_dict.keys(), optimizer_dict.values()) :
    if k == 'l-bfgs-b50' :
        hess_dict[k] = v.hess_inv(optimizer_betamap[k])
        print hess_dict[k]
    else :
        try :
            hess_dict[k] = v.hess_inv
            
        except AttributeError :
            pb_lst.append(k)
#            hess_dict[k] = np.dot(v.jac.T, v.jac) # Ressort une seule variable car la jacobienne ressortie est déjà évoluée au point qui minimise l'erreur
            pass
    try :
        R = np.linalg.cholesky(hess_dict[k])
    except np.linalg.LinAlgError :    
        pass
    
    optimizer_vec[k] = optimizer_betamap[k] + np.dot(R, s)
    
colors = cm.rainbow(np.linspace(0, 1, len(optimizer_dict.keys())))
fig, axes = plt.subplots(1,2,figsize=(30,10))

for item_vec, c in zip(optimizer_vec.items(), colors) :
    axes[0].plot(line_z, item_vec[1], label="beta %s" %(item_vec[0]), color=c, marker='o', linestyle='none')
    
    axes[1].plot(line_z, h_beta(item_vec[1], T_inf, A), label="h_beta %s" %(item_vec[0]), color=c, marker='+', linestyle='none')

axes[0].plot(line_z, optimizer_betamap['bfgs50'], label='betamap-bfgs')            
axes[0].plot(line_z, true_beta(curr_d, noise, T_inf, N_discr), label='Beta True T_inf {}'.format(T_inf))    
axes[1].plot(line_z, h_beta(optimizer_betamap['bfgs50'], T_inf, A), label='H_Beta Betamap bfgs', marker='o', linestyle='none')
axes[1].plot(line_z, curr_d, label='T_obs T_inf = {}'.format(T_inf))



axes[0].set_title("Comparaison des betas optimises et beta Duraisamy")
axes[1].set_title("Champs de temperature avec beta optimise et observe")

axes[0].legend(loc='best', fontsize = 10, ncol=2)
axes[1].legend(loc='best', fontsize = 10, ncol=2)

#plt.plot(line_z, curr_d, label='T_inf {} OBS'.format(T_inf))
#plt.plot(line_z, h_beta(beta_opti, T_inf, A), label='T_inf {} opti'.format(T_inf))
#plt.legend()
