#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import csv, os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

from numdifftools import Gradient, Jacobian

from scipy.stats import norm as norm 
import class_temp_ML as ctml
ctml = reload(ctml)
 
def gaussienne (x, mu, sigma) :
    fac = 1./np.sqrt(2*np.pi*sigma**2)
    
    return fac * np.exp(-(x-mu)**2/2.0/sigma**2)
    
p = ctml.parser()
T = ctml.Temperature(p)
T.obs_pri_model()
T.get_prior_statistics()

dict_discr = dict()
for t_inf in T.T_inf_lst:
    for i in xrange(T.N_discr-2) :
        key = "%d_%d" %(t_inf, i)
        dict_discr[key] = []

for t_inf in T.T_inf_lst :
    for it in xrange(T.num_real) :
        filename    =   'obs_T_inf_{}_{}.csv'.format(t_inf, it)
        T_obs_temp  =   T.pd_read_csv(filename)
        for n in xrange(T.N_discr-2) :
              key = "%d_%d" %(t_inf, n)
              dict_discr[key].append(T_obs_temp[n])

for k in dict_discr.keys() :
    dict_discr[k] = np.asarray(dict_discr[k])

plt.hist(dict_discr["%d_27" %(t_inf)]-T.T_obs_mean['T_inf_50'][27], 30, label='Histogramme des temperature au point 27 shifte par la moyenne de toutes ces valeurs' )

plt.figure("pdf au point %.2f" %(T.line_z[10]))
plt.plot()

T.minimization_with_first_guess()

#x = T.tab_normal(0,0.1,1000)
#T.get_prior_statistics()
#norm.pdf(x)


