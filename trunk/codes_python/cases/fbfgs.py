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
def h_beta(beta, T_inf, A, N_discr= 50, noise= 'none') :
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
N_discr = 50
z_init, z_final = 0., 1.
dt, dz =  0.001, np.abs(z_init-z_final)/float(N_discr)

line_z = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]

kappa, h = 1.0, 0.5

T_inf_lst = [i*5 for i in xrange(10, 11)]

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
    sT_inf = "T_inf_"+str(T_inf)
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
    T_obs_mean[sT_inf] = T_sum
    std_mean_obs    =   np.mean(np.asarray([np.std(T_obs[i]) for i in xrange(len(T_obs))]))
    std_mean_prior  =   np.mean(np.asarray([np.std(T_prior[i]) for i in xrange(len(T_prior))]))
        
    cov_obs_dict[sT_inf]    =   np.diag([std_mean_obs**2 for j in xrange(N_discr-2)])
    cov_prior_dict[sT_inf]  =   np.diag([prior_s**2 for j in xrange(N_discr-2)])
    cov_prior_prior[sT_inf] =   np.diag([std_mean_prior**2 for j in xrange(N_discr-2)])


optimizer_dict, optimizer_betamap, optimizer_vec = dict(), dict(), dict()
hess_dict, cholesky_dict = dict(), dict()

s = np.asarray(tab_normal(0,1,N_discr-2)[0])

for T_inf in T_inf_lst :
    sT_inf = "T_inf_"+str(T_inf)
    print "{} ".format(sT_inf)
    beta_prior = np.asarray([1 for i in xrange(N_discr-2)])

    curr_d  =   T_obs_mean[sT_inf] 
    cov_m,  cov_prior   =   cov_obs_dict[sT_inf],    cov_prior_dict[sT_inf]

    J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - h_beta(beta, T_inf, A)),
                    np.linalg.inv(cov_m)) , (curr_d - h_beta(beta, T_inf, A) )  )  
                + np.dot( np.dot(np.transpose(beta - beta_prior), 
                    np.linalg.inv(cov_prior) ) , (beta - beta_prior) )   
                            ) ## Fonction de coût

    # BFGS : quasi Newton method that approximate the Hessian on the fly
    optimizer_dict[sT_inf] = op.minimize(J, beta_prior, method="BFGS")

    optimizer_betamap[sT_inf] = optimizer_dict[sT_inf].x
    hess_dict[sT_inf] = optimizer_dict[sT_inf].hess_inv
    cholesky_dict[sT_inf] = np.linalg.cholesky(hess_dict[sT_inf])

    optimizer_vec[sT_inf] = optimizer_betamap[sT_inf] + np.dot(cholesky_dict[sT_inf].T, s)
    
    fig, axes = plt.subplots(1,2,figsize=(20,10))
    colors = 'green', 'orange'
    
    axes[0].plot(line_z, optimizer_vec[sT_inf], label="Beta for {}".format(sT_inf), marker='o',
                     linestyle='None', color=colors[0])
    axes[0].plot(line_z, optimizer_betamap[sT_inf], label='Betamap for {}'.format(sT_inf),      
                        marker='o', linestyle='None', color=colors[1])
    axes[0].plot(line_z, true_beta(curr_d, noise, T_inf, N_discr), label = "True beta profile {}".format(sT_inf))
    
    axes[1].plot(line_z, h_beta(optimizer_vec[sT_inf], T_inf, A), label= "h_beta {}".format 
                    (sT_inf), marker='o', linestyle='None', color=colors[0])
    axes[1].plot(line_z, h_beta(optimizer_betamap[sT_inf], T_inf, A), 
                    label= "h_betamap {}".format
                    (sT_inf), marker='o', linestyle ='None', color=colors[1])
    axes[1].plot(line_z, curr_d, label= "curr_d {}".format(sT_inf))

    axes[0].set_title("Optimized beta and Duraisamy beta")
    axes[1].set_title("Temperature field with optimized betas and true solution")

    axes[0].legend(loc='best', fontsize = 10, ncol=2)
    axes[1].legend(loc='best', fontsize = 10, ncol=2)
    
    plt.show()

    sT_inf = "T_inf_"+str(T_inf)
    df = pd.DataFrame(T_obs_mean[sT_inf]) # Curr_d
    df.to_csv("./data/T_obs_mean_{}.csv".format(sT_inf), index=False, header=True)
    
    df = pd.DataFrame(optimizer_betamap[sT_inf])
    df.to_csv("./data/betamap_{}.csv".format(sT_inf), index=False, header=True)
    
beta = dict()
for i in range(10) :
    T_inf = 50
    sT_inf = "T_inf_"+str(T_inf)
    s = tab_normal(0,1,N_discr-2)[0]
    beta[sT_inf+str(i)] = optimizer_betamap[sT_inf] + np.dot(cholesky_dict[sT_inf], s)
   
T_inf = 50    
colors = cm.rainbow(np.linspace(0, 1, 10))
plt.figure()
for i, c in enumerate(colors) :
    # Initialize appropriate pathfile for exact solution and modelled (which has discrepancy)
    plt.plot(line_z, T_prior[i], label='prior bruit {}'.format(i), color=c, marker='o', linestyle='None')

plt.plot(line_z, curr_d, label='curr_d')
plt.legend(loc='best')
plt.show()
