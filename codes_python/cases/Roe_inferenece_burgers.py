import sys, warnings, argparse

import os
import os.path as osp

import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time
import solvers

plt.ion()

pathtosave = osp.join(osp.abspath("./data"), "roe")
if osp.exists(pathtosave) == False :
    os.mkdir(pathtosave)

conditions= {'L'    :   1,
             'Nt'   :   250,
             'Nx'   :   50,
             'tf'   :   0.7,
             'f'    :   lambda u : u**2,
             'fprime' : lambda u : u,
             'type_init' : "sin",
             'amp'  :   1.
             }

dt = conditions['tf'] / conditions['Nt']
dx = float(conditions['L']) / (conditions['Nx']-1)

line_x = np.linspace(0,1,conditions['Nx'])

def true_solve(u) :
    u_next = np.zeros_like(u)
    for j in range(1, len(line_x)-1) :
        u_next[j] = solvers.timestep_roe(u, j, dt/dx, conditions['f'], conditions['fprime'])
    u_next[0] = u_next[-3]
    u_next[-1] = u_next[2]
    
    return u_next
