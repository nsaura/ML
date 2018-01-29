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

import Class_Temp_NonCst as nctc  #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

#run script_non_cst.py -T_prop 35 20 -N 71 -cov_mod "full" -g_sup 1e-2 -cptmax 200 

nctc = reload(nctc)
cfa = reload(cfa)

parser = cfa.parser()

TL = lambda z: parser.T_prop[0] + parser.T_prop[1] * np.cos(np.pi*z)
BL = str(int(parser.T_prop[0])) + "_" + str(int(parser.T_prop[1])) + "cospiz"
    
temp = nctc.Temperature_Noncst(parser, TL, BL)
print(parser)

temp.obs_pri_model()
temp.get_prior_statistics(verbose=True)

plt.ion()
temp.adjoint_bfgs(inter_plot=True)
temp.optimization()

plt.ioff()
cfa.subplot_Ncst(temp, save=True)
#cfa.subplot(temp, method="opti", comp=False, save=True)
#cfa.sigma_plot(temp, save=True)
