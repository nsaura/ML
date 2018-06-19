#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time
import sys

import harmonic_sinus as harm
import Class_MHD_Thomas as cmt

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)
import NN_class_try as NNC

harm = reload(harm)
cmt = reload(cmt)
NNC = reload(NNC)

parser = cmt.THparser()
tc = cmt.Thomas_class(parser)

wdir = osp.abspath("./data/thomas_dataset/complex_init_NN/")
curr_work = osp.join(wdir, "Nx:%d_Nt:%d_nu:%.4f_mag_diff:%.4f_CFL:%.2f" % (tc.Nx, tc.Nt, tc.nu, tc.mag_diff, tc.CFL))

try : 
    os.makedirs(curr_work)
except OSError :
    pass

##---------------------------------------------------         
##--------------------------------------------------- 

def create_init(tc ,dephasage, number, kc=1, plot=False): 
    if tc.type_init == "random" :
        U_intervals = np.linspace(-tc.U, tc.U, 10000)
        H_intervals = np.linspace(-tc.H, tc.H, 10000)
    
        u_init_field, h_init_field = [], []
    
        for i in range(1, tc.Nx-1): 
            u_init_field.append(np.random.choice(U_intervals))
            h_init_field.append(np.random.choice(H_intervals))
        
        u_init_field.insert(0, 0.)
        h_init_field.insert(0, 0.)
        
        u_init_field.insert(len(u_init_field), 0.0)
        h_init_field.insert(len(u_init_field), 0.0)

    if tc.type_init == "sin" :
        u_init_field = tc.U*np.sin(2*np.pi*tc.line_x/tc.L + phase)
        h_init_field = tc.H*np.sin(2*np.pi*tc.line_x/tc.L + phase)
    
    if tc.type_init == "choc" :
        u_init_field, h_init_field = [], []
        
        for x in tc.line_x :
            if 0.5 < x < tc.L/2. + 0.5 :
                u_init_field.append(tc.U)
                h_init_field.append(tc.H)
                
            else :
                u_init_field.append(0.)
                h_init_field.append(0.)
    
    if tc.type_init == 'complex'  :
        inter_deph = np.linspace(-np.pi, np.pi, 10000)
        u_init_field = harm.complex_init_sin(tc.line_x, kc, inter_deph, tc.L, A=25)
        h_init_field = tc.H / tc.U * u_init_field
        
    if plot :
        plt.figure()
        plt.plot(tc.line_x, u_init_field, label="u init", color='blue')
        plt.plot(tc.line_x, h_init_field, label="h init", color='red', linestyle='--')
        plt.legend(loc='best')
    
    return u_init_field, h_init_field

##---------------------------------------------------         
##---------------------------------------------------         
##--------------------------------------------------- 

def solve_case(tc, u_init_field, h_init_field, filename) :

    cpt = 0
    u_nNext = np.zeros((tc.Nx))
    g_nNext = np.zeros((tc.Nx))
    
    u = np.copy(u_init_field)
    h = np.copy(h_init_field)
    
    t = tc.dt
    g = h * np.sqrt(3)
    
    np.save(osp.join(tc.u_dir, filename[0] + "_it%d.npy"%(0)), u)
    np.save(osp.join(tc.h_dir, filename[1] + "_it%d.npy"%(0)), h)
        
    fig, axes = plt.subplots(1, 2, figsize=(8,8), num="Evolution u et g")
    
    while cpt <= tc.itmax :
        
        if cpt % 10 == 0 :
            for i in [0,1]: axes[i].clear()
            axes[0].plot(tc.line_x, u, label="u it %d" % cpt, color="blue")
            axes[0].set_ylim(-round(tc.U*1.5) , round(tc.U*1.5))

            axes[1].plot(tc.line_x, g/np.sqrt(3), label="h it %d" % cpt, color="red")
            axes[1].set_ylim(-tc.U,tc.U)
            
            for i in [0,1]: axes[i].legend(loc="best")
                            
            plt.pause(0.01)

        for j in range(1, tc.Nx-1) :
            u1 = u[j] * (1 - tc.r*( u[j+1] - u[j-1] ))
            g1 = g[j] * (1 + tc.r*( u[j+1] - u[j-1] ))
        
            u2 = g[j]*tc.r*(g[j+1] - g[j-1])
            g2 = u[j]*tc.r*(g[j+1] - g[j-1])
        
            u3 = tc.fac*(u[j+1] - 2.*u[j] + u[j-1])
            g3 = tc.mag_diff * tc.dt / tc.dx**2 * (g[j+1] - 2.*g[j] + g[j-1])
        
            u_nNext[j] = u1 + u2 + u3
            g_nNext[j] = g1 - g2 + g3
        
        
        u_nNext[0] = u_nNext[-2]
        u_nNext[-1] = u_nNext[1]
        
        g_nNext[0] = g_nNext[-2]
        g_nNext[-1] = g_nNext[1]
        
        u = u_nNext
        g = g_nNext
        
        if True in np.isnan(u) :
            sys.exit("CPT %d nan in u" % cpt)

        cpt += 1
        
        utowrite = filename[0] + "_it%d.npy"%(cpt)
        htowrite = filename[1] + "_it%d.npy"%(cpt)

        if osp.exists(utowrite) == True:
            os.remove(utowrite)
        
        if osp.exists(htowrite) == True :
            os.remove(htowrite)

        np.save(utowrite, u)
        np.save(htowrite, h)

        t += tc.dt

##---------------------------------------------------             
##---------------------------------------------------         
##--------------------------------------------------- 

inter_deph = np.linspace(-np.pi/2., np.pi/2., 1000)

choices = [np.random.choice(inter_deph) for j in range(10)]
add_block = lambda u,h,j : [ u[j-1], u[j], u[j+1], h[j-1], h[j], h[j+1] ]

def compute_true_u(tc, nsamples, pi_line, plot=False, write=False) :
    """
    tc : object from Class MHD Thomas
    nsamples : number of drawn
    pi_line : Intervall in which phases are drawn
    plot : boolean if you want to plot different things
    write : boolean if you want to write some files
    """
    X = np.zeros((6))
    y = np.zeros((2))
    
    for n in range(nsamples) :
        filename = [osp.join(curr_work, "u"), osp.join(curr_work, "h")]
        
        for i, s in enumerate(filename) :
            if osp.exists(s) == False :
                os.mkdir(s)
                
            filename[i] = osp.join(s,"cpx_init_kc%d_%d" % (1, n))
        
        dephasage = np.random.choice(pi_line)
        
        uu, hh  = create_init(tc ,dephasage, n, kc=1, plot=False)
        solve_case(tc, uu, hh, filename)
        
        for it in range(1, tc.itmax) :
            u_curr = np.load(filename[0] + "_it%d.npy" % (it))
            h_curr = np.load(filename[1] + "_it%d.npy" % (it))

            u_next = np.load(filename[0] + "_it%d.npy" % (it+1))
            h_next = np.load(filename[1] + "_it%d.npy" % (it+1))
            
            for j in range(1, len(uu)-1) :
                X = np.block([ [X], add_block(u_curr, h_curr, j) ])        
                y = np.block([ [y], [ u_next[j], h_next[j] ] ])
                
    X = np.delete(X, 0, axis=0)        
    y = np.delete(y, 0, axis=0)
    
    return X, y 

##---------------------------------------------------             
##---------------------------------------------------         
##--------------------------------------------------- 
          
dict_layers = {"I": 3, "O" :2, "N1":80, "N2":80, "N3":80, "N4":80, "N5":80, "N6":80}#, "N7":80, "N8":80, "N9":80, "N10":80}#, "N11":40}

def MHD_buildNN(lr, X, y, act, opti, loss, max_epoch, reduce_type, scaler, N_=dict_layers, step=50, early_stop=False, **kwargs) :
    plt.ion()
    print kwargs
    
    # Define an NN object
    nn_obj = NNC.Neural_Network(lr, scaler = scaler, N_=N_, max_epoch=max_epoch, reduce_type=reduce_type, **kwargs)
    
    # Spliting The Data
    nn_obj.split_and_scale(X, y, shuffle=True)
    
    #Preparing the Tensorflow Graph
    nn_obj.tf_variables()
    nn_obj.layer_stacking_and_act(act)
    
    #Setting Optimizer and Loss for the graph    
    nn_obj.def_optimizer(opti)
    nn_obj.cost_computation(loss)
    
    #Display a Recap
    nn_obj.case_specification_recap()
    
    kwargs = nn_obj.kwargs
    
#    return nn_obj
    print nn_obj.X_train.shape
    try :
        nn_obj.training_phase(tol=1e-3, early_stop=early_stop)

    except KeyboardInterrupt :
        print ("Session closed")
        del nn_obj.sess

    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    
    score = nn_obj.score(beta_test_preds, nn_obj.y_test)
    print ("score of this session is : {}".format(score))
    
    test_line = range(len(nn_obj.X_test))
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in test_line])
    error_estimation = sum(deviation)

    dev_lab = "Pred_lr_{}_{}_{}_Maxepoch_{}".format(lr, opti, act, scaler, max_epoch)
    
    if plt.fignum_exists("Comparaison sur le test set") :
        plt.figure("Comparaison sur le test set")
        plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                    fillstyle='none', linestyle='none', c=nn_obj.kwargs["color"])

    else :
        plt.figure("Comparaison sur le test set")
        plt.plot(test_line, nn_obj.y_test, label="Expected value", marker='o', fillstyle='none',\
                    linestyle='none', c='k')   
        plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                    fillstyle='none', linestyle='none', c=nn_obj.kwargs["color"])
 
    plt.legend(loc="best", prop={'size': 7})
    
    if plt.fignum_exists("Deviation of the prediction") :
            plt.figure("Deviation of the prediction")
            plt.plot(nn_obj.y_test, beta_test_preds, c=nn_obj.kwargs["color"], marker='o',\
                     linestyle='none', label=dev_lab, ms=3)
        
    else :
        plt.figure("Deviation of the prediction")
        plt.plot(nn_obj.y_test, nn_obj.y_test, c='k', label="reference line")
        plt.plot(nn_obj.y_test, nn_obj.y_test, c='navy', marker='+', label="wanted value",linestyle='none')
        plt.plot(nn_obj.y_test, beta_test_preds, c=nn_obj.kwargs["color"], marker='o',\
                      linestyle='none', label=dev_lab, ms=3)

    plt.legend(loc="best", prop={'size': 7}) 

#    print("Modèle utilisant N_dict_layer = {}".format(N_))\\
    print("Modèle pour H_NL = {}, H_NN = {} \n".format(len(N_.keys())-2, N_["N1"]))
    print("Fonction d'activation : {}\n Fonction de cout : {}\n\
    Méthode d'optimisation : {}".format(act, loss, opti))
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()

    return nn_obj
    
    
    
    
    
# run MHDmulti_init_NN_thomas.py -Nx 1000 -nu 2.5e-3 -Nt 500 -ratio 0.5 -U 1.5 -init_u "complex"
# X, y = compute_true_u(tc, 3, inter_deph, True, True)
# nn =  MHD_buildNN(1e-3, X, y, "selu", "Adam", "MSEGrad", 70, "sum", "Standard", N_=dict_layers, color="purple",  bsz=64,  BN=True)    
    
