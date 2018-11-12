#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import sympy as sy

sy.init_printing()

def func_XY_to_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))
    
x1, x2 = sy.symbols("x1,x2")
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)] # dérivé de f_sym par rapport aux variables
#⎡                  3                           ⎤
#⎣-2⋅x₂ + 4⋅(x₁ - 1) , -2⋅x₁ + 10⋅x₂ - 10 ⎦

grad_mat = sy.Matrix(fprime_sym)
#⎡                               3⎤
#⎢-2⋅x₂ + 4⋅(x₁ - 1) |
#⎣-2⋅x₁ + 10⋅x₂ - 10 ⎦

# Pour la hessienne, on dérive par rapport à ces deux variables
fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)] 
hess_mat = sy.Matrix(fhess_sym)
#⎡                  2        ⎤
#⎢12⋅(x₁ - 1)   -2⎥
#⎣     -2       10   ⎦




