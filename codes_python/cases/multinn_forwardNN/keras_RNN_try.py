#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp
import tensorflow as tf

from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, LSTM

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split

from tensorflow import reset_default_graph

## Import du chemain cases ##
case_folder = osp.abspath(osp.dirname("../"))
sys.path.append(case_folder)

import Class_write_case as cwc
import Class_Vit_Choc as cvc
import harmonic_sinus as harm

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

cvc = reload(cvc)
harm = reload(harm)

# run keras_RNN_try.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 52 -Nt 32 -beta_prior 10 -typeJ "u"

np.random.seed(1000000)
parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser) 


wdir = osp.abspath("./data/burger_dataset/complex_init_NN/ELW/")
curr_work = osp.join(wdir, "Nx:%d_Nt:%d_nu:%.4f_CFL:%0.2f" % (cb.Nx, cb.Nt, cb.nu, cb.CFL)) #beta_Nx:52_Nt:32_nu:0.025_typei:sin_CFL:0.4_it:017.npy

pi_line = np.linspace(-np.pi/2., np.pi/2., 1000)

#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------

def LW_solver(u_init, itmax=cb.itmax, filename="u_test", write=False, plot=False) :
    r = cb.dt/cb.dx
    t = it = 0
    u = np.copy(u_init)
    u_nNext = []
    
    while it <= itmax  :
        if "test" in filename.split("_") :
            abs_work = osp.join(curr_work, "tests_case")
            if osp.exists(abs_work) == False:
                os.makedirs(abs_work)
        else :
            abs_work = curr_work
            if osp.exists(abs_work) == False:
                os.makedirs(abs_work)
            
        curr_filename = filename + "_it%d.npy"%(it+1)
        curr_filename = osp.join(abs_work, curr_filename)
        
        if osp.exists(filename) == True :
            it += 1
            continue
            
        fu = np.asarray([0.5*u_x**2 for u_x in u])
        
        der_sec = [cb.fac*(u[k+1] - 2*u[k] + u[k-1]) for k in range(1, len(u)-1)]
        der_sec.insert(0, cb.fac*(u[1] - 2*u[0] + u[-1]))
        der_sec.insert(len(der_sec), cb.fac*(u[0] - 2*u[-1] + u[-2]))

        for i in range(1,cb.Nx-1) : # Pour prendre en compte le point Nx-2
            u_m, u_p = cvc.intermediaires(u, fu, i, r)
            fu_m =  0.5*u_m**2
            fu_p =  0.5*u_p**2

            u_nNext.append( u[i] - r*( fu_p - fu_m ) + der_sec[i] )
                                        
        # Conditions aux limites 
        u[1:cb.Nx-1] = u_nNext  
        u_nNext  = []
        
        u[0] = u[-2]
        u[-1]= u[1]
        
        u = np.asarray(u) 
        
        if write == True :
            if osp.exists(curr_filename) :
                os.remove(curr_filename)
            np.save(curr_filename, u)
        
        if plot==True :
            if it % 10 == 0:
                plt.figure("Evolution de la solution")
                plt.clf()
                plt.plot(cb.line_x, u, label="iteration = %d" % it)
                plt.legend()
                plt.ylim(-2,2)
                plt.pause(0.15)
            
        it += 1

    return u, abs_work

#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------

def XYFor_RNN(cb, nsamples, amp_line, pi_line, kc = 1, plot=False, write=False) :
    """
    Considerant plusieurs conditions initiales, cette fonction fait evoluer ces conditions avec un solver LW. 
    Les solutions sont ensuites concatenees en deux matrices X et y pour etre utilisees comme support 
    d\'entrainement d\'une intelligence artificielle.
    
    Arguments :
    -----------
    cb : 
    nsamples    : nombre de tirages pour chaque cas
    amp_line    : np.array des amplitudes que l\'on veut considerer pour les conditions intiale ex : [0.4, 0.8, 1.2] 
    pi_line     : intervalle entre -pi/2 et pi/2 a partir duquel un dephasage est choisi aleatoirement pour chaque CI
    kc      :   (optionnel) : gere la complexite des conditions initiales, fixe a 1
    plot    :   (optionnel) : boolen, True si l\'on veut voir l\'evolution de la CI
    write   :   (optionnel) : booleen, True si l\'on veut ecrire les champs de vitesses pour toutes les iterations
    
    
    """
    X = np.zeros((4))
    y = np.zeros((1))
    
    cb.Nx = 202
    cb.Nt = 202
    
    cb.line_x = np.linspace(0, cb.L, cb.Nx)
#    cb.itmax = 250
#    colors = iter(cm.gist_heat(np.arange(len(amp_line)*10)))
    colors = iter(cm.gnuplot(np.arange(len(amp_line)*10)))
    
    for amp in amp_line :
        print ("nsample(s) = %d" % nsamples) 
        for n in range(nsamples) :
            print ( "amp = %.2f \t n = %d" %(amp, n) )
            
            filename = "%.2f_init_kc%d_%d" % (amp, kc, n)
            uu = cb.init_u(amp, phase = np.random.choice(pi_line))
            
            for cc in range(8) :
                next(colors)
            
            plt.figure("Initialisation cases")
            plt.plot(cb.line_x, uu, color=next(colors))
            plt.xlabel("X-domain")
            plt.ylabel("init conditions")
            plt.ylim((-2.1, 2.1))
            plt.pause(0.01)
            
            _, abs_work = LW_solver(uu, cb.itmax, filename = filename, write=write, plot=plot)
                    
            for it in range(1, cb.itmax) :
                u_curr = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it)))
                u_next = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it+1)))

                for j in range(1, len(uu)-1) :
                    X = np.block([[X], [u_curr[j-1], u_curr[j], u_curr[j+1],\
                                (u_curr[j+1] - u_curr[j-1])*0.5/cb.dx]])
                    y = np.block([[y], [float(u_next[j] - u_curr[j])/cb.dt]])
    
    X = np.delete(X, 0, axis=0)        
    y = np.delete(y, 0, axis=0)
    
    write_X_y(X, y)
    
    return X, y 

#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------

def write_X_y(X, y) :
    xys_loc = osp.join(wdir, "Xys")
    
    if osp.exists(xys_loc) == False :
        os.mkdir(xys_loc)
    
    PathToCheckx = osp.join(xys_loc, "ELWDerX_multi_" + osp.split(curr_work)[1])
    PathToChecky = osp.join(xys_loc, "ELWDery_multi_" + osp.split(curr_work)[1])
    
    if osp.exists(PathToCheckx) == False or osp.exists(PathToChecky) == False :
        np.save(PathToCheckx, X)
        np.save(PathToChecky, y)
    
    print ("ELWDerX_multi and ELWDery_multi written in %s" %(xys_loc))
    
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
    
#def ELW_xs_compute_true_u(cb, nsamples, amp_line, pi_line, kc=1, plot=False, write=False) :
#    X = np.zeros((n_inputs))
#    y = np.zeros((1))
#    colors = iter(cm.plasma(np.arange(len(amp_line))))
#    
#    for amp in amp_line :
#        for n in range(nsamples) :
#            filename = "%.2f_init_kc%d_%d" % (amp, kc, n)
#            uu = 1.+cb.init_u(amp, phase = np.random.choice(pi_line))
#            _, abs_work = LW_solver(uu, cb.itmax, filename=filename, write=write, plot=plot)
#                
#            for it in range(1, cb.itmax) :
#                u_curr = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it)))
#                u_next = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it+1)))
#                    
#                for j in range(1, len(uu)-1) :
#                    X = np.block([[X], add_block(u_curr, cb.line_x, j)])        
#                    y = np.block([[y], [float(u_next[j] - u_curr[j])/cb.dt]])
#        
#    X = np.delete(X, 0, axis=0)        
#    y = np.delete(y, 0, axis=0)
#    
#    return X, y 

X = 

def build_keras_model(X, y):
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    


