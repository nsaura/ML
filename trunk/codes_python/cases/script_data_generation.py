#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import time

import numpy as np
import pandas as pd
import os.path as os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import optimize as op

import class_temp_ML as ct #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

ct = reload(ct)
cfa = reload(cfa)

parser = cfa.parser()

temp = ct.Temperature(parser)
print(parser)

temp.obs_pri_model()
temp.get_prior_statistics()

ct.adjoint_bfgs(inter_plot=True)
ct.optimization()

cfa.subplot(temp)
cfa.sigma_plot(temp, save=True)

