#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(100000)

def E(k):
    if k >=1 and  k<= 5 :
        return 5.**(-5./3.)
    if k > 5 :
        return float(k)**(-5./3.)

def beta_k(inter):
    return np.random.choice(inter)

def complex_init_sin(X, kc, inter_deph, L, A=25, plot=False) :
    uu = np.zeros((len(X)))
    
    if plot==True :
        plt.figure("Harmonic composition")

    for k in range(1, kc+1) :
        harm = 0.5*np.sqrt(2.*A*E(k))*np.sin(2*np.pi/L*k*X + beta_k(inter_deph))
        for i in range(len(uu)) :
            uu[i] += harm[i]
        
#        if plot==True :
#            plt.plot(X, harm, label="harm %d" % k, linestyle='--')

    if plot==True:
        plt.plot(X, uu, label="Somme de toutes les harmoniques", marker="o", ms=4)
        plt.legend(ncol=3, loc="best", prop={'size': 5})

    return uu
    
if __name__ == "__main__" :
    plt.ion()

    A = 25
    X = np.linspace(0, 2*np.pi, 500)
    inter = np.linspace(-np.pi, np.pi, 1028)

