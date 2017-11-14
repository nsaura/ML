#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys

import T_data_exact_obs as td

plt.ion()


cov_obs_lst, cov_prior_lst = [], [] 
cov_prior_prior = []
prior_sigma = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]


for i, (T_inf, prior_s) in enumerate(zip(td.T_inf_lst, prior_sigma)) : 
    #i : xrange(len(T_inf_lst), T_inf = T_inf_lst[i]
    T_obs, T_prior = [], []     
    for it in xrange(0,10) :
    
    # Initialize appropriate pathfile for exact solution and modelled (which has discrepancy)
        path_abs = os.path.abspath(os.path.curdir)
        pf_obs = os.path.join(path_abs, 'data', 'obs_T_inf_{}_{}.csv'.format(T_inf, it))
        pf_prior = os.path.join(path_abs, 'data', 'prior_T_inf_{}_{}.csv'.format(T_inf, it))
        
        # Compute covariance from data 
        T_temp = pd.read_csv(pf_obs).get_values()
        T_temp = T_temp.reshape(T_temp.shape[0])
        T_obs.append(T_temp)
        
        # We conserve the T_disc
        T_disc = pd.read_csv(pf_prior).get_values()
        T_disc = T_disc.reshape(T_disc.shape[0])
        T_prior.append(T_disc)
    
        # Compute prior cov for each prior sigma given in the article
    std_mean_obs    =   np.mean(np.asarray([np.std(T_obs[i]) for i in xrange(len(T_obs))]))
    std_mean_prior  =   np.mean(np.asarray([np.std(T_prior[i]) for i in xrange(len(T_prior))]))
        
    cov_obs_lst.append(np.diag([std_mean_obs**2 for j in xrange(td.N_discr-2)]))
    cov_prior_lst.append(np.diag([prior_s**2 for j in xrange(td.N_discr-2)]))
    cov_prior_prior.append(np.diag([std_mean_prior**2 for j in xrange(td.N_discr-2)]))

