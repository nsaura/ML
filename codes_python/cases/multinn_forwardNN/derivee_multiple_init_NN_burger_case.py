#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split

from tensorflow import reset_default_graph


## Import du chemin TF ##
tf_folder = osp.abspath(osp.dirname("../../TF/"))
sys.path.append(tf_folder)

## Import du chemain cases ##
case_folder = osp.abspath(osp.dirname("../"))
sys.path.append(case_folder)

import Bootstrap_NN as BNN
import NN_class_try as NNC

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

NNC = reload(NNC)
BNN = reload(BNN)

cvc = reload(cvc)
harm = reload(harm)

#run multiple_init_NN_burger_case.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 52 -Nt 32 -beta_prior 10 -typeJ "u"

np.random.seed(1000000)
parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser) 

wdir = osp.abspath("./../data/burger_dataset/complex_init_NN/Der")
curr_work = osp.join(wdir, "Nx:%d_Nt:%d_nu:%.4f_CFL:%0.2f" % (cb.Nx, cb.Nt, cb.nu, cb.CFL)) #beta_Nx:52_Nt:32_nu:0.025_typei:sin_CFL:0.4_it:017.npy

if osp.exists(curr_work) == False :
    os.makedirs(curr_work)

def LW_solver(u_init, itmax=cb.itmax, filename="u_test", write=False, plot=False) :
    r = cb.dt/cb.dx
    t = it = 0
    u = np.copy(u_init)
    u_nNext = []
    
    while it <= itmax  :
        if "test" in filename.split("_") :
            abs_work = osp.join(curr_work, "tests_case")
            if osp.exists(abs_work) == False:
                os.mkdir(abs_work)
        else :
            abs_work = curr_work
            
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

def Der_compute_true_u(cb, nsamples, pi_line, kcmax = 1, plot=False, write=False) :
    X = np.zeros((4))
    y = np.zeros((1))
    
    for n in range(nsamples) :
        for kc in range(1,kcmax+1) :
            filename = "cpx_init_kc%d_%d" % (kc, n)
            uu = harm.complex_init_sin(cb.line_x, kc=kc, inter_deph=pi_line, L=cb.L, plot=plot)
            _, abs_work = LW_solver(uu, cb.itmax, filename = filename, write=write, plot=plot)
                
            for it in range(1, cb.itmax) :
                u_curr = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it)))
                u_next = np.load(osp.join(abs_work, filename + "_it%d.npy" % (it+1)))

                for j in range(1, len(uu)-1) :
                    X = np.block([[X], [u_curr[j-1], u_curr[j], u_curr[j+1], (u_curr[j+1] - u_curr[j-1])*0.5/cb.dx]])        
                    y = np.block([[y], [u_next[j]]])
    
    X = np.delete(X, 0, axis=0)        
    y = np.delete(y, 0, axis=0)
    
    return X, y 
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
####

pi_line = np.linspace(-np.pi/2., np.pi/2., 1000)
choices = [np.random.choice(pi_line) for i in range(7)]


dict_layers = {"I": 3, "O" :1, "N1":80, "N2":80, "N3":80}#, "N4":80, "N5":80, "N6":80}#, "N7":80, "N8":80, "N9":80, "N10":80}#, "N11":40}

def Der_multi_buildNN(lr, X, y, act, opti, loss, max_epoch, reduce_type, scaler, N_=dict_layers, step=50, early_stop=False, **kwargs) :
    plt.ion()
    print (kwargs)
    
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
    print (nn_obj.X_train.shape)
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

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def Der_bootstrap(num_predictors, lr, X, y, dict_layers, act, opti, loss, max_epoch, reduce_type, scaler, step=50, rdn = 0, early_stop=False, **kwargs) :
    dataset = { "data" : X,
                "target" : y}
    
    bts = BNN.Bootstraped_Neural_Network(num_predictors, dataset) 
    bts.resample_dataset()       

    color = ["blue", "darkred", "aqua", "olive", "magenta", "orange", "mediumpurple", "chartreuse", "tomato", "saddlebrown", "powderblue", "khaki", "salmon", "darkgoldenrod", "crimson", "dodgerblue", "limegreen"]
    
    color = iter(np.array(color)[np.random.permutation(len(color))])
    
    bts.build_NN(1e-3, dict_layers, act, opti, loss, palet = color, max_epoch=max_epoch, scaler=scaler, reduce_type=reduce_type, rdn=rdn, **kwargs)

    return bts

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def Der_bootstrap_solver(boot_obj, rescale, cb=cb) :
    plt.figure()
    p = min(np.random.choice(pi_line), np.random.choice(pi_line))
    
    u = harm.complex_init_sin(cb.line_x, 1, pi_line, cb.L)
    
    _, abs_work = LW_solver(u, cb.itmax, "u_test", write=True)
    print (abs_work)
    
    fetch_real_u = lambda it : np.load(osp.join(abs_work, "u_test_it%d.npy"%(it)))
    
    u_nNext = []
    var = []
    
    colors = np.array([mpl.colors.to_rgb(cc) for cc in boot_obj.colors_dict.values()])
    color = np.mean(colors, axis=0)
    
    inf_errs = []
    
    ax, axx = [], []
    
    ax1 = plt.subplot(221) # Erreur a l'iteration n
    ax2 = plt.subplot(222) # Evolution de la norme infinie de l'erreur 
    ax3 = plt.subplot(212) # Evolution de la prédiction
    
    ax.append(ax1) ; ax.append(ax2) ; ax.append(ax3)
    axx.append(ax1); axx.append(ax3)
    
    for it in range(1, cb.itmax) :
        if it > 1 :
            u = u_nNext
            u_nNext = []
            var = []
            
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1], (u[j+1] - u[j-1])/(2*cb.dx) ])
            
            mean_curr_pred = boot_obj.bootstrap_prediction(xs, rescale=rescale, variance=False)
            
            var_curr_pred = boot_obj.bootstrap_variance(xs, rescale = rescale)
            
            u_nNext.append(mean_curr_pred)
            var.append(var_curr_pred)
            
        # u_nNext.shape = 30 
        # use of list type to insert in a second time boundary condition
        
        u_nNext.insert(0, u[-2])
        u_nNext.insert(len(u), u[1])
        
        u_nNext = np.array(u_nNext)
        u_nNext_ex = fetch_real_u(it+1)
        
        errs = np.array([(u_nNext[i] -  u_nNext_ex[i]) for i in range(cb.Nx)])
      
        inf_err = np.linalg.norm(errs, np.inf)
        inf_errs.append(inf_err)
        
        if it % 5 == 0 :
            axx[0] = ax[0]
            axx[1] = ax[-1]
            
            for a in axx :
                a.cla()
            
            # Erreur a l'iteration n
            ax[0].plot(cb.line_x, np.abs(errs), label="Abs error : $\hat{u}^{n+1} - u^{n+1}_t$", c=color)
            
            # Evolution de la norme infinie de l'erreur 
            ax[1].scatter(it, inf_err, c=color, s=12)
            
            ax[-1].plot(cb.line_x[1:cb.Nx-1], u_nNext_ex[1:cb.Nx-1], label="True it = %d" %(it+1), c='k')
            ax[-1].plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted at it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c=color)
            
            ax[-1].fill_between(cb.line_x[1:cb.Nx-1], -10*np.array(var), 10*np.array(var), facecolor= "0.2", alpha=0.4, interpolate=True, label="$\pm \sigma$")  
            
            for a in ax :
                a.legend(prop={'size': 8})
            
            fig = plt.gcf()
            fig.tight_layout()
            
            plt.title("Iteration %d" %it)

            plt.pause(2)

    return inf_errs
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def Der_multiNN_solver(nn_obj, cb=cb):
    plt.figure()
    p = min(np.random.choice(pi_line), np.random.choice(pi_line))
    
#    u = cb.init_u(p, cpx=2)
    u = harm.complex_init_sin(cb.line_x, 1, pi_line, cb.L)
    
    _, abs_work = LW_solver(u, cb.itmax, "u_test", write=True)
    print (abs_work)
    fetch_real_u = lambda it : np.load(osp.join(abs_work, "u_test_it%d.npy"%(it)))
    
    u_nNext = []
    inf_errs = []
    
    ax, axx = [], []
    
    ax1 = plt.subplot(221) # Erreur a l'iteration n
    ax2 = plt.subplot(222) # Evolution de la norme infinie de l'erreur 
    ax3 = plt.subplot(212) # Evolution de la prédiction
    
    ax.append(ax1) ; ax.append(ax2) ; ax.append(ax3)
    axx.append(ax1); axx.append(ax3)
    
    for it in range(1, cb.itmax) :
        if it > 1 :
            u = u_nNext
            u_nNext = []
            
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1], (u[j+1] - u[j-1])/(2*cb.dx) ])
            
            xs = nn_obj.scale_inputs(xs)
            xs = xs.reshape(1, -1)

            u_nNext.append(nn_obj.predict(xs)[0,0])
        # u_nNext.shape = 30 
        # use of list type to insert in a second time boundary condition
        
        u_nNext.insert(0, u[-2])
        u_nNext.insert(len(u), u[1])
        
        u_nNext = np.array(u_nNext)
        u_nNext_ex = fetch_real_u(it+1)
        
        errs = np.array([(u_nNext[i] -  u_nNext_ex[i]) for i in range(cb.Nx)])
        
        inf_err = np.linalg.norm(errs, np.inf)
        
        inf_errs.append(inf_err)
        
        if it % 5 == 0 :
            axx[0] = ax[0]
            axx[1] = ax[-1]
            
            for a in axx :
                a.cla()
            
            # Erreur a l'iteration n
            ax[0].plot(cb.line_x, np.abs(errs), label="Relative Erreur $\hat{u}^{n+1} - u^{n+1}_t$", c=nn_obj.kwargs["color"])
            
            # Evolution de la norme infinie de l'erreur 
            ax[1].scatter(it, inf_err, c=nn_obj.kwargs["color"], s=12)
            
            ax[-1].plot(cb.line_x[1:cb.Nx-1], u_nNext_ex[1:cb.Nx-1], label="True it = %d" %(it+1), c='k')
            ax[-1].plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted at it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c=nn_obj.kwargs["color"])
            
            for a in ax :
                a.legend(prop={'size': 8})
            
            fig = plt.gcf()
            fig.tight_layout()
            
            plt.title("Iteration %d" %it)
            
            plt.pause(2)

    return inf_errs
    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
   
def Der_multiRF_solver(X, y): 
    xtr, xte, ytr, yte = train_test_split(X, y.ravel(), shuffle=True, random_state=0)
    rf = rfr(n_estimators=20)
    rf.fit(xtr, ytr)
    
    plt.figure()
    p = np.random.choice(pi_line)
    
    u = cb.init_u(p) + cb.init_u(np.random.choice(pi_line))
    
    _, abs_work = LW_solver(u, "u_test", write=True)
    
    fetch_real_u = lambda it : np.load(osp.join(abs_work, "u_test_it%d.npy"%(it)))
    
    u_nNext = []
    
    for it in range(1, cb.itmax) :
        if it > 1 :
            u = u_nNext
            u_nNext = []
            
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1]]).reshape(1,-1)
            
            u_nNext.append(rf.predict(xs))
        
        u_nNext.insert(0, u[-2])
        u_nNext.insert(len(u), u[1])
        
        u_nNext = np.array(u_nNext).ravel()
        
        plt.clf()        
        plt.plot(cb.line_x[1:cb.Nx-1], fetch_real_u(it+1)[1:cb.Nx-1], label="True it = %d" %(it+1), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="RF Predicted at it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c="green")
        plt.legend()
        plt.pause(2)


# run derivee_multiple_init_NN_burger_case.py -nu 2.5e-2 -itmax 80 -CFL 0.4 -num_real 5 -Nx 52 -Nt 32 -beta_prior 10 -dp "../data/burger_dataset/"
# Der_X_multi, Der_y_multi = Der_compute_true_u(cb, 12, pi_line, plot=True, write=True)
# Der_nn = Der_multi_buildNN(1e-3, Der_X_multi, Der_y_multi, "selu", "Adam", "MSEGrad", 70, "sum", "Standard", N_=dict_layers, color="purple",  bsz=64,  BN=True)
# errs = Der_multiNN_solver(Der_nn)

# Booststrap : 
# run derivee_multiple_init_NN_burger_case.py -nu 2.5e-2 -itmax 80 -CFL 0.4 -num_real 5 -Nx 52 -Nt 32 -beta_prior 10 -dp "../data/burger_dataset/"
# Der_X_multi, Der_y_multi = Der_compute_true_u(cb, 12, pi_line, plot=True, write=True)
# boot_obj = Der_bootstrap(5, 1e-3, Der_X_multi, Der_y_multi, dict_layers, "selu", "Adam", "Lasso", 100, "sum", "Standard", bsz = 128, BN=True)
# Der_bootstrap_solver(boot_obj, True)
