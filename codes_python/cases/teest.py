# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

dim = 5

def test_func(x):
    return (x[0])**2+(x[1])**2
    
def test_grad(x):
    return [2*x[0],2*x[1]]
    

print op.line_search(test_func, test_grad, np.array([1.8, 1.7]), np.array([-1,-1]))

#f = lambda x : x**2 + x + 4
#fprime = lambda x : 2*x + 1

#M = np.diag(np.asarray([i for i in range(dim)]))
#H_n = np.eye(dim)

#g_n = fprime(np.asarray([1 for i in range(dim)]))
#d_n = -np.dot(H_n, g_n)

#dk = lambda x : -np.dot(H_n, fprime(x))
#op.line_search(f, fprime, np.asarray([i for i in range(dim)]), d_n)

