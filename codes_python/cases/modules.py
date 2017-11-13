#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def tab_normal(mu, sigma, length) :
    return s

def test (**kwargs) :
    dico = dict()
    for kway in kwargs :
        dico[kway] = kwargs[kway]
    
    print dico['thrd']


X_init = 0; X_fin = 5 ; NN = 150

dx = np.abs(X_init - X_fin)/float(NN)
der= []
f = lambda X : X*X

for i in np.arange(X_init, X_fin+dx, dx) :
    der.append((f(i+dx) - f(i))/dx)
    


