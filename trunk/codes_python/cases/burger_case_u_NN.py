#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

# Author : NS
# Use of Various Helps From SO and CV; stack community
# Ridge https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb
# Vanilla Gradient descent https://towardsdatascience.com/improving-vanilla-gradient-descent-f9d91031ab1d

# To run
# run burger_NN.py -nu 2.5e-2 -itmax 100 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10    
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

cb.obs_res(True, True)

# Dataset Construction
def xy_burger (num_real, cb=cb, n_input=6) :
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

    X = np.zeros((6))
    y = np.zeros((1))
#    flux = []
    
    print (lst_pairs_bu[0][0])
    print (X[0])
    #return X, y 
    # pairs :   p[0] -> beta
    #           p[1] -> u
    beta_chol = dict()
    u_chol = dict()

    lst_pairs_bu = sorted(lst_pairs_bu)
    lst_pairs_bc = sorted(lst_pairs_bc)

    #color=cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0]))
    color=iter(cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0])))
    dx = cb.dx
    
#    for h in range(1, len(u)-1):    
#        flux.append((u[h+1] - u[h-1])/dx)
    
    for it, (pbu, pbc) in enumerate(zip(lst_pairs_bu, lst_pairs_bc)) :
        beta = np.load(pbu[0])
        chol = np.load(pbc[1])
        u = np.load(pbu[1])
        
        u_mean = np.zeros_like(u[1:cb.Nx-1])
        c = next(color)
#        if osp.splitext(pbu[1])[0][-3:] != "000" :
        for i in range(num_real) :
            # Pour eviter de reiterer la premiere (Obsolete?)
            beta_chol[str(i)] = beta + chol.dot(np.random.rand(len(beta)))            
#            print np.shape(beta_chol[str(i)])
            u_chol[str(i)]= cb.u_beta(beta_chol[str(i)], u)
                
    #            for k in range(0, len(u)-2) : #Pour aller de 1 à Nx-1 vav de u-beta_chol mais commencer aux bons endroits
    #                u_mean[k] += u_beta_chol[str(i)][k+1] / float(num_real)
        
        for j in range(1, len(u)-1) :
#            if osp.splitext(pbu[1])[0][-3:] != "000" :
            flux = (u[j+1] - u[j-1])/dx
            X = np.block([[X], [cb.line_x[j-1], cb.line_x[j], cb.line_x[j+1], u[j-1], u[j], u[j+1]]])
            y = np.block([[y], [beta[j]]])
                
        for i in range(num_real) :
            bb = beta_chol[str(i)]
            uu = u_chol[str(i)]
                                        
            for k in range(1, len(u)-1) :
                X = np.block([[X], [cb.line_x[k-1], cb.line_x[k], cb.line_x[k+1], uu[k-1], uu[k], uu[k+1]]])
                y = np.block([[y], [beta[k]]])
        print ("it= ", it) 
        print "X.shape[0] -1 = ",X.shape[0] - 1 
#        u_last = u
#        beta_last = beta  
#        chol_last = chol
        
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    
    return X, y

X, y = xy_burger(num_real=1, cb=cb)

dict_layers = {"I" : X.shape[1],\
#               "N1":10000,\
#               "N2" : 100,\
#               "O": 1\
#              }
               "N1" : 1000,\
               "N2" : 1000,\
               "N3" : 1000,\
#               "N5" : 30,\
#               "N10" : 30,\
#               "N11" : 30,\
#               "N12" : 30,\
#               "N13" : 30,\
#               "N14" : 10,\
#               "N15" : 10,\
#               "N16" : 10,\
#               "N17" : 10,\
#               "N18" : 10,\
               "O"  : 1}
#for j in range(5,100) :
#    dict_layers["N%d" % j] = 10

#nn_adam_sum = build_case(1e-3, X, y , act="selu", opti="Adam", loss="OLS", max_epoch=1000, reduce_type="sum", verbose=True, N_=dict_layers, color="blue",  scale=True, bsz=256, BN=True)

print dict_layers
def recentre(xs, X_train_mean, X_train_std):
    """
    This function refocuses the input before prediction. It needs three arguments :
    
    xs              :   The input to refocuses
    X_train_mean    :   The mean vector calculated according to the training set
    X_train_std     :   The standard deviation vector calculated according to the training set   
    
    xs is a vector of shape [N] 
    X_train_mean and X_train_std are vectors of whose shape is [N_features]
    """
    for i in range(np.size(X_train_mean)-1) :
        xs[i] -= X_train_mean[i]
        if np.abs(X_train_std[i]) > 1e-12 :
            xs[i] /= X_train_std[i]

    return xs

def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=True, step=50, **kwargs) :
    plt.ion()
    print kwargs
    nn_obj = NNC.Neural_Network(lr, N_=dict_layers, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.def_optimizer(opti)
    nn_obj.layer_stacking_and_act(activation=act)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
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

    plt.figure("Comaparaison sur le test set")
    plt.plot(test_line, beta_test_preds, label="Prediction sur Test set", marker='+',\
    fillstyle='none', linestyle='none', c='r')
    plt.plot(test_line, nn_obj.y_test, label="Expected value", marker='o', fillstyle='none', linestyle='none', c='k')   
    plt.legend()
#    plt.figure("Evolution de l\'erreur %s" %(loss))
#    plt.plot(range(len(nn_obj.costs)), nn_obj.costs, c='r', marker='o', alpha=0.3,\
#            linestyle="none")
#    error_estimation /= (len(nn_obj.X_test) -1)
    
    plt.figure("Deviation of the prediction")
    plt.plot(nn_obj.y_test, nn_obj.y_test, c='k', label="reference line")
    plt.plot(nn_obj.y_test, nn_obj.y_test, c='b', marker='+', label="wanted     value",linestyle='none')
    plt.plot(nn_obj.y_test, beta_test_preds, c='r', marker='o', linestyle='none',   label="predicted value", ms=3)
    plt.legend(loc="best") 

#    print("Modèle utilisant N_dict_layer = {}".format(N_))\\
    print("Modèle pour H_NL = {}, H_NN = {} \n".format(len(N_.keys())-2, N_["N1"]))
    print("Fonction d'activation : {}\n Fonction de cout : {}\n\
    Méthode d'optimisation : {}".format(act, loss, opti))
    print("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()
    
#    lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=True, step=50, **kwargs
    
#    f = cwc.File("traceback_u_burger_nn.ods")
#    f.read_file()
#    
#    data = {}
#    
#    data["lr"] = lr
#    data["act"]=act
#    data["opti"] = opti
#    data["maxepoch"] = max_epoch
#    data["loss"] = loss
#    data["structure"] = dict_layers
#    data["finalcost"] = nn_obj.costs[-1]
#    
#    if nn_obj.batched == True :
#        data["batchsize"] = kwargs["b_sz"]
#    
#    else :
#        data["batchsize"] = " "
#        
#    if opti == "Nadam" or opti == "Adam":
#        data["beta1"] = kwargs["beta1"]
#        data["beta2"] = kwargs["beta2"]
#        
#        data["decay"] = " "
#        data["momentum"] = " "
#        
#    if opti == "RMS" :
#        data["decay"] = kwargs["decay"]
#        data["momentum"] = kwargs["momentum"]
#        
#        data["beta1"] = " "
#        data["beta2"] = " "
#    
#    if opti == "GD" or opti == "SGD":
#        data["beta1"] = " "
#        data["beta2"] = " "
#        data["decay"] = " "
#        data["momentum"] = " "
#    
#    print data 
#    f.write_in_file(data)
    return nn_obj

#nn_adam_mean = build_case(1e-4, X, y , act="relu", opti="Adam", loss="OLS", decay=0.5, momentum=0.8, max_epoch=5000, reduce_type="sum", verbose=True)

#nn_adam_mean = build_case(5e-4, X, y , act="selu", opti="Adam", loss="OLS", max_epoch=700, reduce_type="mean", verbose=True, b_sz=250, N_=dict_layers, color="k", BN=True)
#nn_adam_mean = build_case(1e-3, X, y , act="selu", opti="Adam", loss="OLS", max_epoch=2000, reduce_type="mean", verbose=True, b_sz=250, N_=dict_layers, color="tan", BN=True)

def NN_solver(nn_obj, cb=cb, typeJ="u"):
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
    plt.figure()
    for it in range(1, cb.itmax) :
        beta = []
        u_mean = np.mean(u)
        for j in range(1, cb.Nx-1) :
            xs = np.array([cb.line_x[j-1], cb.line_x[j], cb.line_x[j+1], u[j-1], u[j], u[j+1]])
            print xs
            if nn_obj.scale == True :
                xs = recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
            print xs
            xs = xs.reshape(-1, nn_obj.X.shape[1])
            print "xs.shape= ", xs.shape
            beta.append(nn_obj.predict(xs)[0,0])
        
#        print(beta, type(beta), np.shape(beta))
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
        u[0] = u[-2]
        u[-1] = u[1]
        
        plt.clf()        
        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it+1))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u[1:cb.Nx-1], label="Predicted it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c='navy')
        
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
                xs = recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
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
#    
#    
#    

# To run this program : 
# You should first have inferred data for your problem (this can be done by running cb.minimization(maxiter=50) see Class_Vit_Choc at the end of the file 
# Then you want to build and train a Neural Network. This can be done by typing (for example) :
# nn_test = build_case(1e-2, X, y , act="selu", opti="Proximal_Adag", loss="OLS", max_epoch=3000, reduce_type="sum", verbose=True, N_=N_, color="red",  scale=True, bsz=256, BN=True)
# To tune those parameters see in NN_class_try in TF directory

# After having trained the NN you want to test its prediction ability :
# NN_solver(nn_test) # Its important to use the NN you've just trained

# If you want to train a KNN on top of that, for comparision (e.g.) :
# processing(nn-test, cb=cb, n_neigh=3) # We use here the NN previously trained to have access to important variables

# Stacking on process
