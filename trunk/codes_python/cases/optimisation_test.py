#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys

plt.ion()

T_recup, sigma_obs_lst, cov_prior = [], [], []
T_inf_lst = [i*5 for i in xrange(1, 11)]

N_discr = 33

line_z = np.linspace(0.,1.,N_discr)[1:N_discr-1]
sigma_prior = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]

for i, (T_inf, prior_s) in enumerate(zip(T_inf_lst, sigma_prior)) : 
    #i : xrange(len(T_inf_lst), T_inf = T_inf_lst[i]
    pathfile = os.path.join('./data', 'T_inf_{}.csv'.format(T_inf))
    T_temp = pd.read_csv(pathfile).get_values()
    T_temp = T_temp.reshape(T_temp.shape[0])
    sigma_obs_lst.append(T_temp.std()**2)
    T_recup.append(T_temp)
    
    cov_prior.append(np.diag([prior_s for j in xrange(len(T_inf_lst))])) # Liste de matrices de prior covariance pour différente valeurs de prior_sigma

verbose = False
if verbose == True :
    for i, (T_inf, TT) in enumerate(zip(T_inf_lst,T_recup)) :
        if i%3 == 0 :
            plt.plot(line_z, TT, label="T_inf: {}".format(T_inf))

    plt.legend(loc='best')

## Construction fonction et d'autres variables à optimiser

cov_m = np.diag(sigma_obs_lst)


#misfit = 0.5*( ( np.dot( np.dot(np.transpose(d_obs - model),np.linalg.inv(cov_m)) , (d_obs - model) ) ) 
#                   + np.dot( np.dot(np.transpose(curr_beta - beta_prior), np.linalg.inv(cov_beta) ) , (curr_beta - beta_prior) )  ) 
