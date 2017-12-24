#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

plt.ion()
plt.close("all")

def Next_hess(prev_hess_inv, y_nN, s_nN, dim ) :
    
    rho_nN  =   1./np.dot(y_nN.T, s_nN) if np.dot(y_nN.T, s_nN) is not 0 else 1./1e-5
    print rho_nN
        
    Id      =   np.eye(dim)
        
    A1 = Id - rho_nN * s_nN[:, np.newaxis] * y_nN[np.newaxis, :]
    A2 = Id - rho_nN * y_nN[:, np.newaxis] * s_nN[np.newaxis, :]
        
    return np.dot(A1, np.dot(prev_hess_inv, A2)) + (rho_nN* s_nN[:, np.newaxis] * s_nN[np.newaxis, :])

def search_alpha(func, func_prime, curr_beta, alpha=1.) :
    alpha_m = lambda m : float(alpha) / (2**(m-1))
    mm, cptmax = 1, 50
    while ((func(curr_beta - alpha_m(mm)*func_prime) - func(curr_beta)) <= \
                 -0.5*alpha_m(mm) * np.linalg.norm(func_prime)**2 ) == False  and mm < cptmax:
        mm +=1
        
    return alpha_m(mm)

dim = 8

func = lambda x: (x)**2 + (x+4)**4 + np.sinc(x**2) +7

xx = np.linspace(-7, 3, 1000)

plt.plot(xx, func(xx))
plt.show()

J = lambda x : np.dot(func(x.T), func(x))
g_J =   lambda x : 2*x + 4*(x+4)**3 + 2*(x**2*np.cos(x**2) - np.sin(x**2))/x**4

#plt.plot(np.arange(-1,1,0.01), J(np.arange(-1,1,0.01)))

x_init = np.asarray([4 for i in range(dim)], dtype=np.float)
x_init[0], x_init[6] = 0.1, 1.7
J_lst, alpha_lst = [], []
H_n = np.eye(dim)
sup_g, cpt, cptmax = np.linalg.norm(g_J(x_init), np.inf), 0, 100

x_n =   x_init
g_n =   g_J(x_init)

x_evol_compo = dict()
if dim < 10 :
    for i in range(dim) :
        x_evol_compo["compo %d" %(i)] = []

plt.figure("Erreur du gradient ")
plt.scatter(cpt, sup_g)

for j,k in enumerate(x_evol_compo.keys()) :
    x_evol_compo[k].append(x_init[j])

J_lst.append(J(x_n))
while cpt<cptmax and sup_g > 1e-10 :
    if cpt > 0 :
        g_n =   g_nN
        x_n =   x_nN
        H_n =   H_nN
    
    d_n     =   np.dot(H_n, g_n)
    print "d_n = ", d_n
    alpha   =   search_alpha(J, g_n, x_n) 
    alpha_lst.append(alpha)  
    print "alpha = ", alpha
    x_nN    =   x_n - alpha*d_n
    
    if dim < 10 :
        for j,k in enumerate(x_evol_compo.keys()) :
            x_evol_compo[k].append(x_nN[j])
    
    print "x_nN = ", x_nN
    g_nN    =   g_J(x_nN)
    print "g_nN = ", g_nN
    s_nN    =   x_nN - x_n
    y_nN    =   g_nN  - g_n
    
    if np.linalg.norm(s_nN) ==0 or np.linalg.norm(y_nN) == 0 : print"out";  break
    
    H_nN    =   Next_hess(H_n, y_nN, s_nN, dim)
    
    sup_g   =   np.linalg.norm(g_J(x_n), np.inf)
    cpt    +=   1
    J_lst.append(J(x_nN))
    plt.scatter(cpt, sup_g)

print "Fin de la routine bfgs recode"
print "Début de  la routine python"

opti = op.minimize(J, x_init)
print("Fin de la routine python")

plt.figure("Val abs des differences des deux methodes")
plt.plot(range(dim), np.abs(x_nN - opti.x))
plt.ylabel("np.abs(x_nN - python_opti.x)")

plt.figure("trace de J(x_n)")
plt.plot(range(cpt+1), J_lst)

plt.figure("Evolution de alpha")
plt.plot(range(len(alpha_lst)), alpha_lst)

if dim < 10 :
    fig, axes = plt.subplots(2, dim/2, figsize = (15, 10))
    for i, (ax, item) in enumerate(zip(axes.ravel(), x_evol_compo.iteritems())) :
        ax.set_title("k = %s" %(item[0]))
        ax.scatter(range(cpt+1), item[1])
        

