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

import Class_Temp_Cst as ctc  #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

plt.ion()

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

#run script_data_generation.py -T_inf_lst 30 -cptmax 150 -N 71 -g_sup 1e-4 -cov_mod "full"
ctc = reload(ctc)
cfa = reload(cfa)

parser = cfa.parser()

temp = ctc.Temperature_cst(parser)
print(parser)

# On conmmence par calculer les solutions du problème exact et les solutions du problème non exact avec beta = beta_prior + bruit. 
# Si ces calcules ont déjà été efffectué, on récupère les données dans des fichiers créés à dessein.
temp.obs_pri_model()

# On calcule les covariances : cov_obs diag ou full puis cov_pri prenant en compte la condition sur sigma prior (voir code).
temp.get_prior_statistics()

# Optimization de la fonction de coût J définie équation (7) par deux méthodes :
temp.adjoint_bfgs(inter_plot=True, verbose=True) # Optimization "maison";
#temp.adjoint_circle(inter_plot=True, verbose=False)
temp.optimization()                 # Optimization de Scipy qui sert de référence.

temp.write_fields()
# Tracés:
# Tracés de beta_map pour les deux solutions, et on compare les distribution autour de beta_map.
# On trace également le champ de température obtenu en injectant beta_map dans le problème non exact
cfa.subplot_cst(temp, save=True)
cfa.subplot_cst(temp, method="opti", comp=False, save=True)

# On compare les sigma de la covariance a posteri pour les deux méthodes, avec les sigmas attendus.
cfa.sigma_plot_cst(temp, save=True)
plt.ion()
plt.show()




