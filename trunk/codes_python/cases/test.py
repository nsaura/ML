#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys
from numpy.linalg import norm as norm #norme euclidienne : norm(vec, 2)
import T_data_exact_obs as td

gauss = td.lst_gauss
plt.ion()
plt.close('all')
N_discr = 33

T_inf_lst = [i*5 for i in xrange(1,11)] ; line_z = np.linspace(0.,1.,td.N_discr)[1:td.N_discr-1]

std_obs, std_prior = dict(), dict()
mean_obs, mean_prior = dict(), dict()

obs_blist, prior_blist = [], []
for k, T_inf in enumerate(T_inf_lst) :

    prior_lst, obs_lst = [], []
    for i in xrange(10) :
       
       
        path_abs = os.path.abspath(os.path.curdir)
        pathfile_ex = os.path.join(path_abs, 'data', 'obs_T_inf_{}_{}.csv'.format(T_inf, i))
        pathfile_mod = os.path.join(path_abs, 'data', 'prior_T_inf_{}_{}.csv'.format(T_inf, i))
        
        
        dfprior = pd.read_csv(pathfile_mod).get_values()
        dfprior = dfprior.reshape(dfprior.shape[0])
        prior_lst.append(dfprior)
        
        
        dfobs = pd.read_csv(pathfile_ex).get_values()
        dfobs = dfobs.reshape(dfobs.shape[0])
        obs_lst.append(dfobs)
    
    
    std_mean_obs = np.mean(np.asarray([np.std(obs_lst[i]) for i in xrange(10)]))
    std_mean_prior = np.mean(np.asarray([np.std(prior_lst[i]) for i in xrange(10)]))
    
    mean_mean_obs = np.mean(np.asarray([np.mean(obs_lst[i]) for i in xrange(10)]))
    mean_mean_prior = np.mean(np.asarray([np.mean(prior_lst[i]) for i in xrange(10)]))
    
    std_obs[str(T_inf)] = std_mean_obs
    std_prior[str(T_inf)] = std_mean_prior
    
    mean_obs[str(T_inf)], mean_prior[str(T_inf)] = mean_mean_obs, mean_mean_prior
    
for io, ip in zip(std_obs.items(), std_prior.items()) :
    print("key : {} \t; obs : {:.5f} ; prior {:.5f}".format(io[0], io[1], ip[1]))
    
    plt.figure("Spectre %s" %(io[0]))
    plt.plot(line_z, td.tab_normal( mean_obs[io[0]], std_obs[io[0]], len(line_z))[0] , label='obs')
    plt.plot(line_z, td.tab_normal( mean_prior[ip[0]], std_prior[ip[0]], len(line_z))[0] , label='prior')
    plt.legend()
    
    
    
        
        
