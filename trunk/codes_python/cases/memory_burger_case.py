#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

# Author : NS
# Use of Various Helps From SO and CV; stack community
# Ridge https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb
# Vanilla Gradient descent https://towardsdatascience.com/improving-vanilla-gradient-descent-f9d91031ab1d

# To run
# run burger_case_u_NN.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10 -typeJ "u"
# 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from itertools import cycle
from matplotlib.pyplot import cm

from scipy import optimize as op

import numdifftools as nd

import glob
import time

import Class_write_case as cwc

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Vit_Choc as cvc

NNC = reload(NNC)
cvc = reload(cvc)
cwc = reload(cwc)

parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

#cb.obs_res(True, True)

# Dataset Construction
def xy_burger (num_real, cb=cb, n_inputs=6, n_points=3, verbose=False) :
    #cb.minimization(maxiter=50, step=5)

    u_name = cb.u_name
    b_name = cb.beta_name

    root = osp.split(cb.beta_path)[0]

    uloc = cb.inferred_U
    betaloc = cb.beta_path
    cholloc = cb.chol_path

    # On va se servir de ça
    b_files = glob.glob(betaloc+'/*')
    u_files = glob.glob(uloc+'/*')
    c_files = glob.glob(cholloc+'/*')

    b_u = dict()
    b_c = dict()
    lst_pairs_bu = []
    lst_pairs_bc = []
    
    print betaloc
    
    l = osp.split(b_files[0])[1].split("_")
    ll = [i.split(":") for i in l[1:-1]]

    # To be able to modify u,b and c_files we have first to copy them 
    u_sorted = sorted(u_files, key= lambda x: int(osp.splitext(x)[0][-3:]))
    b_sorted = sorted(b_files, key= lambda x: int(osp.splitext(x)[0][-3:]))
    c_sorted = sorted(c_files, key= lambda x: int(osp.splitext(x)[0][-3:]))


    cpt_rm = 0
    for u, b, c in zip(u_sorted, b_sorted, c_sorted) :
        cond = lambda f : int(osp.splitext(f)[0][-3:]) > cb.itmax-1
        if cond(u) and cond(b) and cond(c) :
            u_files.remove(u)
            b_files.remove(b)
            c_files.remove(c)
#            print "RM"
            
            cpt_rm += 1
    print ("%d files removed from u.b and c_files lists" %(cpt_rm))

    #print ("u_files = \n{}b_files = \n{}c_files = {}\n".format(u_files, b_files, c_files))

    # Init
    for elt in ll :
        b_u[elt[0]] = []
        b_c[elt[0]] = []
        
    for b in b_files :
        to_find = osp.split(b)[1][4:]
        
        u_to_find = uloc + "/U" + to_find
        
        if osp.exists(u_to_find) :
            lst_pairs_bu.append((b, u_files[u_files.index(u_to_find)]))
        else :
            print ("%s does not exist" %(u_to_find))

        c_to_find = cholloc + "/chol" + to_find
        
        if osp.exists(c_to_find)  :
            lst_pairs_bc.append((b, c_files[c_files.index(c_to_find)]))
        else :
            print ("%s does not exist" %(c_to_find))
    # For checking    
    #    for i in range (10) :
    #        print lst_pairs[np.random.randint(len(lst_pairs))]

    X = np.zeros((n_inputs))
    y = np.zeros((1))
#    flux = []
    
#    print (lst_pairs_bu[0][0])

    # pairs : for u
    #              p[0] -> beta
    #              p[1] -> u
    #         for chol 
    #              p[0] -> beta
    #              p[1] -> c

    duc, duc_p, duc_pp = dict(), dict(), dict()
    dbc, dbc_p, dbc_pp = dict(), dict(), dict()

    lst_pairs_bu = sorted(lst_pairs_bu)
    lst_pairs_bc = sorted(lst_pairs_bc)

    #color=cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0]))
    color=iter(cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0])))
    dx = cb.dx
    
    if n_points == 3 :
        add_block     = lambda u, up, upp, xj : [u[xj-1], u[xj], u[xj+1],\
                                                 up[xj-1], up[xj], up[xj+1],\
                                                 upp[xj-1], upp[xj], upp[xj+1]\
                                                ] 
    
    if n_points == 2 :
        add_block     = lambda u, up, upp, xj : [u[xj-1], u[xj],\
                                                 up[xj-1], up[xj],\
                                                 upp[xj-1], upp[xj]\
                                                ] 
    
#    add_block     = lambda u, up, upp, xj : [u[xj-1], u[xj], u[xj+1],\
#                                             up[xj-1], up[xj], up[xj+1],\
#                                             upp[xj-1], upp[xj], upp[xj+1]\
#                                            ] 
    beta_chol_fct = lambda beta, chol : beta + chol.dot(np.random.rand(len(beta)))
    u_chol_fct    = lambda beta_co, u : cb.u_beta(beta_co, u) + np.random.rand(len(u))) * 0.1
    
    for it, (pbu, pbc) in enumerate(zip(lst_pairs_bu, lst_pairs_bc)) :
        beta  =  np.load(pbu[0])
        chol  =  np.load(pbc[1])
        u     =  np.load(pbu[1])
        
        if it > 1 :
            b_p = np.load(lst_pairs_bu[it-1][0])
            b_pp = np.load(lst_pairs_bu[it-2][0])
            
            u_p = np.load(lst_pairs_bu[it-1][1])
            u_pp = np.load(lst_pairs_bu[it-2][1])
            
            c_p = np.load(lst_pairs_bc[it-1][1])
            c_pp = np.load(lst_pairs_bc[it-2][1])
            
        else :
            b_p = np.load(lst_pairs_bu[0][0])
            b_pp = np.load(lst_pairs_bu[0][0])
            
            u_p = np.load(lst_pairs_bu[0][1])
            u_pp = np.load(lst_pairs_bu[0][1])
            
            c_p = np.load(lst_pairs_bc[0][1])
            c_pp = np.load(lst_pairs_bc[0][1])        
        
        col = next(color)

        for i in range(num_real) :
            duc[str(i)] = u_chol_fct(beta_chol_fct(beta, chol), u) # u_chol
            duc_p[str(i)] = u_chol_fct(beta_chol_fct(b_p, c_p), u_p) # u_chol_previous
            duc_pp[str(i)] = u_chol_fct(beta_chol_fct(b_pp, c_pp), u_pp) # u_chol_previous_previous
            
        for j in range(1, len(u)-1) :
            new_block = add_block(u, u_p, u_pp, j)
            
            X = np.block([ [X], new_block ])
            y = np.block([ [y], [beta[j]] ])
                
        for i in range(num_real) :
            uc, uc_p, uc_pp = duc[str(i)], duc_p[str(i)], duc_pp[str(i)]
                                        
            for k in range(1, len(u)-1) :
                new_block = add_block(uc, uc_p,  uc_pp, k)
                
                X = np.block([ [X], new_block ])
                y = np.block([ [y], [beta[k]]])
        
        if verbose == True :
            print X.shape[0]-1
            time.sleep(0.2)
#       X Gets (Nx-2)*(num_real+1) inputs for each iteration        
    print pbu
    
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    
    return X, y

#X, y = xy_burger(num_real=4, cb=cb, n_inputs=9)

def dict_layer(X) :
    dict_layers = {"I" : X.shape[1],\
                   "N1" : 200,\
                   "N2" : 100,\
                   "N3" : 50,\
                   "N4" : 10,\
                   "O"  : 1}
    return dict_layers
#print dict_layers

def build_memory_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, scaler, N_, step=50, **kwargs) :
    plt.ion()
    print kwargs
    nn_obj = NNC.Neural_Network(lr, N_=N_, max_epoch=max_epoch, reduce_type=reduce_type, **kwargs)
    
    nn_obj.split_and_scale(X, y, scaler=scaler, shuffle=True)
    nn_obj.tf_variables()
    nn_obj.def_optimizer(opti)
    nn_obj.layer_stacking_and_act(activation=act)
    nn_obj.cost_computation(loss)
    nn_obj.case_specification_recap()
    
    kwargs = nn_obj.kwargs
    
#    return nn_obj
    print nn_obj.X_train.shape
    try :
        nn_obj.training_session(tol=1e-3)

    except KeyboardInterrupt :
        print ("Session closed")
        nn_obj.sess.close()

    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
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

# see burger_case_u_NN for writing something into ols fashion
    
    return nn_obj
##------------------------------------------------------------------------------------------------------------
def mNN_solver(nn_obj, cb=cb, typeJ="u", n_points=3):
    beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
    
    evol = 0
    
    if cb.typeJ != typeJ : cb.typeJ = typeJ
    
    # Initialisation it = 1
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=1))
    u_p = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=0))
    u_pp = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=0))
    
    
    if n_points == 3 :
        add_block     = lambda u, up, upp, xj : [u[xj-1], u[xj], u[xj+1],\
                                                 up[xj-1], up[xj], up[xj+1],\
                                                 upp[xj-1], upp[xj], upp[xj+1]\
                                                ] 
    
    if n_points == 2 :
        add_block     = lambda u, up, upp, xj : [u[xj-1], u[xj],\
                                                 up[xj-1], up[xj],\
                                                 upp[xj-1], upp[xj]\
                                                ] 
    
    plt.figure()
    
    for it in range(1, cb.itmax) :
        if it > 1 :
            u_pp = u_p 
            u_p = u
            u = u_nNext
                   
        beta = []

        for j in range(1, cb.Nx-1) :
            xs = np.array(add_block(u, u_p, u_pp, j))
            
            xs = nn_obj.scale_inputs(xs)
            xs = xs.reshape(1, -1)

            beta.append(nn_obj.predict(xs)[0,0])
        
        # u_nNext.shape = 30 
        # use of list type to insert in a second time boundary condition
        u_nNext = cb.u_beta(np.asarray(beta), u)
        
#        u_nNext.insert(0, u[-2])
#        u_nNext.insert(len(u_nNext), u[1])
        
#        u = u_nNext
#        u[0] = u[-2]
#        u[-1] = u[1]
        
        plt.clf()        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))[1:cb.Nx-1], label="True it = %d" %(it+1), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted at it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c=nn_obj.kwargs["color"])
        
        plt.legend()
        plt.pause(5)

###---------------------------------------------------------------
def processing(nn_obj, cb=cb, n_neigh = 3) :
    plt.ion()
    
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor(n_neighbors=n_neigh).fit(nn_obj.X_train, nn_obj.y_train)
    
    beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
    
    # Initialisation it = 1
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=0))
    for it in range(1, cb.itmax) :
        beta = []
        u_mean = np.mean(u)
        for j in range(1, cb.Nx-1) :
            xs = np.array([cb.line_x[j-1], cb.line_x[j], cb.line_x[j+1], u[j-1], u[j], u[j+1]])
            if nn_obj.scale == True :
                xs = nn_obj.scale_inputs(xs)
            xs = xs.reshape(-1,6)
            print xs.shape
            beta.append(reg.predict(xs)[0,0])
            print(beta)

        print(beta, type(beta), np.shape(beta))
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
        u[0] = u[-2]
        u[-1] = u[1]
        
        plt.figure("Dynamique knn")
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it+1))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u[1:cb.Nx-1], label="Predicted it = %d"%(it), marker='o', fillstyle = 'none', linestyle= 'none', c='steelblue') 
        plt.legend()
        plt.pause(5)
        plt.clf()
#    grid_search = GridSearchCV(SVC(), param_grid, cv=5) # Objet a entrainer et evaluer

    print("Test differences entre prediction et Y_test : ")
    print("{}".format(y_test - reg.predict(X_test)))
    
    plt.figure("KNN vs True")
    plt.plot(T.line_z, T_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
    plt.plot(T.line_z, T_true, label='True', linestyle='--', c='k')
    plt.legend()

    plt.figure("beta KNN vs True")
    plt.plot(T.line_z, beta_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
    plt.plot(T.line_z, true_beta, label='True', linestyle='--', c='k')
    plt.legend()

#                                                          #
############################################################
############################################################
#                  To run this program :                   #
############################################################
############################################################
#                                                          #
# You should first have inferred data for your problem (this can be done by running 
#       cb.minimization(maxiter=50) see Class_Vit_Choc at the end of the file 

# Any way you have to run this class in order to create a cb object needed 
#       run burger_case_u_NN.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10 -typeJ "u"

# Then you want to build and train a Neural Network. This can be done by typing (for example) :
#        nn_test = build_case(5e-4, X, y , act="selu", opti="Adam", loss="Custom", custom_param=1e-4, max_epoch=300\,
#        reduce_type="sum", verbose=True, N_=N_, color="red", scaler="Standard", bsz=256, BN=True)

# To tune those parameters see in NN_class_try in TF directory

# After having trained the NN you want to test its prediction ability :

#       NN_solver(nn_test) # Its important to use the NN you've just trained

# If you want to train a KNN on top of that, for comparision (e.g.) :
# processing(nn-test, cb=cb, n_neigh=3) # We use here the NN previously trained to have access to important variables

# Stacking on process
