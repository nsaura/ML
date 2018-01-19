#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import time

import numpy as np
import pandas as pd
import os 

import os.path as osp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import optimize as op

import class_temp_ML as ctm  #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

# run script_data_generation.py -T_inf_lst 15 -g_sup 1e-3 -tol 1e-5 -beta_prior 1. -num_real 100 -cov_mod 'full' -N 33 -dt 1e-4 -cptmax 300

ctm = reload(ctm)
cfa = reload(cfa)

parser = cfa.parser()

temp = ctm.Temperature(parser)
print(parser)

temp.obs_pri_model()
temp.get_prior_statistics()

temp.adjoint_bfgs(inter_plot=False)
temp.optimization()

cfa.subplot(temp, save=True)
cfa.subplot(temp, method="opti", comp=False, save=True)
cfa.sigma_plot(temp, save=True)

