#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd
import sympy as sy
sy.init_printing()

import matplotlib.pyplot as plt
import csv, os, sys, warnings, argparse

from scipy import optimize as op
from itertools import import
matplotlib cycle.cm as cm

from numdifftools import Gradient, Jacobian
from scipy.stats import norm as norm 

import class_temp_ML as ctml
ctml = reload(ctml)

def func_XY_to_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))

p = ctml.parser()
T = ctml.Temperature(p)
T.get_prior_statistics()

bp = T.beta_prior
key = "obs_T_inf_50_9.csv"

obs =   T.pd_read_csv(key)
hbp =   T.h_beta(bp, 50)

cov_obs     =   np.linalg.inv(T.cov_obs_dict["T_inf_50"])
cov_prior   =   np.linalg(T.cov_pri_dict["T_inf_50"])
J   =   lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - T.h_beta(beta, T_inf)),
                    np.linalg.inv(cov_m)) , (curr_d - T.h_beta(beta, T_inf) )  )  
                + np.dot( np.dot(np.transpose(beta - bp), 
                    np.linalg.inv(cov_prior) ) , (beta - bp) )   
                            ) ## Fonction de coût 
                            

