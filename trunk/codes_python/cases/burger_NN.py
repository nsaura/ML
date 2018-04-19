#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from matplotlib.pyplot import cm

from scipy import optimize as op
from itertools import cycle


import numdifftools as nd

import time
import glob

#run burger_NN.py -nu 2.5e-2 -itmax 200 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Vit_Choc as cvc

cvc = reload(cvc)
NNC = reload(NNC)

#def x_y_burger () :
parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

cb.obs_res(True, True)
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

X = np.zeros((3))
y = np.zeros((1))

print lst_pairs_bu[0][0]
print X[0]
#return X, y 
# pairs :   p[0] -> beta
#           p[1] -> u
beta_chol = dict()
u_beta_chol = dict()

lst_pairs_bu = sorted(lst_pairs_bu)
lst_pairs_bc = sorted(lst_pairs_bc)

plt.figure("betas")

num_real = 5

#color=cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0]))
color=iter(cm.rainbow(np.linspace(0,1,np.shape(lst_pairs_bu)[0])))

for it, (pbu, pbc) in enumerate(zip(lst_pairs_bu, lst_pairs_bc)) :
    beta = np.load(pbu[0])
    chol = np.load(pbc[1])
    u = np.load(pbu[1])
    
    c = next(color)
    if osp.splitext(pbu[1])[0][-3:] != "000" :
        for i in range(num_real) :
            beta_chol[str(i)] = beta_last + chol_last.dot(np.random.rand(len(beta_last)))            
#            print np.shape(beta_chol[str(i)])
            u_beta_chol[str(i)] = cb.u_beta(beta_chol[str(i)], u_last)
    
        plt.plot(cb.line_x[1:cb.Nx-1], beta[1:cb.Nx-1], c=c, label="beta iteration %d" %(it))
    
    for j in range(1, len(u)-1) :
        if osp.splitext(pbu[1])[0][-3:] != "000" :
            X = np.block([[X], [u[j-1], u[j], u[j+1]]])
            y = np.block([[y], [beta[j]]])
            
            for i in range(num_real) :
                bb = beta_chol[str(i)]
                uu = u_beta_chol[str(i)]
                X = np.block([[X], [uu[j-1], uu[j], uu[j+1]]])
                y = np.block([[y], [bb[j]]])
        
    u_last = u
    beta_last = beta  
    chol_last = chol
    
X = np.delete(X, 0, axis=0)
y = np.delete(y, 0, axis=0)
plt.legend(ncol=3)

dict_layers = {"I" : 3,\
               "N1" : 100,\
               "N2" : 100,\
               "N3" : 100,\
               "N4" : 100,\
               "N5" : 100,\
               "N6" : 100,\
               "O"  : 1}

def recentre(xs, X_train_mean, X_train_std):
    """
    This function refocuses the input before prediction. It needs three arguments :
    
    xs              :   The input to refocuses
    X_train_mean    :   The mean vector calculated according to the training set
    X_train_std     :   The standard deviation vector calculated according to the training set   
    
    xs is a vector of shape [N] 
    X_train_mean and X_train_std are vectors of whose shape is [N_features]
    """
    for i in range(np.size(X_train_mean)) :
        xs[i] -= X_train_mean[i]
        if np.abs(X_train_std[i]) > 1e-12 :
            xs[i] /= X_train_std[i]

    return xs

def build_case(lr, X, y, act, opti, loss, max_epoch, reduce_type, N_=dict_layers, scale=True, **kwargs) :
    plt.ion()
    nn_obj = NNC.Neural_Network(lr, N_=dict_layers, max_epoch=max_epoch)
    
    nn_obj.train_and_split(X,y,strat=False,shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.feed_forward(activation=act)
    nn_obj.def_training(opti, **kwargs)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.def_optimization()
    try :
        b_sz = kwargs["batch_sz"]
        print b_sz
        nn_obj.training_session(tol=1e-3, batch_sz=b_sz, step=50, verbose=True)
    
    except KeyboardInterrupt :
        print "Session closed"
        nn_obj.sess.close()
    beta_test_preds = np.array(nn_obj.predict(nn_obj.X_test))
    test_line = range(len(nn_obj.X_test))
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    deviation = np.array([ abs(beta_test_preds[j] - nn_obj.y_test[j]) for j in test_line])
    error_estimation = sum(deviation)

    if verbose == True :
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
    print ("Moyenne de la somme des écart sur le test set = {}\n".format(error_estimation))
    
    plt.show()
    return nn_obj


#nn_adam_mean = build_case(1e-4, X, y , act="relu", opti="Adam", loss="OLS", decay=0.5, momentum=0.8, max_epoch=20000, reduce_type="sum", verbose=True)
#nn_adam_mean = build_case(1e-4, X, y , act="relu", opti="Adam", loss="OLS", decay=0.5, momentum=0.8, max_epoch=5000, reduce_type="sum", verbose=True)

def NN_solver(nn_obj, cb=cb):
    beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
    chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(cb.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
    
    # Initialisation it = 1
    u = np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it=1))
    for it in range(1, cb.itmax) :
        beta = []
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1]])
            if nn_obj.scale == True :
                xs = recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
            xs = xs.reshape(-1,3)
            beta.append(nn_obj.predict(xs)[0,0])
        

        print beta, type(beta), np.shape(beta)
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
        u[0] = u[-2]
        u[-1] = u[1]
        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted it = %d" %(it), marker='o', fillstyle = 'none', linestyle= 'none', c='steelblue') 
        plt.legend()
        plt.pause(5)
        plt.clf()

def processing(nn_obj, cb=cb, n_neigh = 3) :
#    run knn_ML.py -T_inf_lst 5 10 15 20 25 30 35 40 45 50 -N_sample 10

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
        for j in range(1, cb.Nx-1) :
            xs = np.array([u[j-1], u[j], u[j+1]])
            if nn_obj.scale == True :
                xs = recentre(xs, nn_obj.X_train_mean, nn_obj.X_train_std)
            xs = xs.reshape(-1,3)
            beta.append(nn_obj.predict(xs)[0,0])
            print beta

        print beta, type(beta), np.shape(beta)
        u_nNext = cb.u_beta(np.asarray(beta), u)
        u = u_nNext
#        u[0] = u[-2]
#        u[-1] = u[1]
        
        plt.plot(cb.line_x[1:cb.Nx-1], np.load(u_name(cb.Nx, cb.Nt, cb.nu, cb.type_init, cb.CFL, it))[1:cb.Nx-1], label="True it = %d" %(it), c='k')
        plt.plot(cb.line_x[1:cb.Nx-1], u_nNext[1:cb.Nx-1], label="Predicted it = %d"%(it), marker='o', fillstyle = 'none', linestyle= 'none', c='steelblue') 
        plt.legend()
        plt.pause(5)
        plt.clf()
#    grid_search = GridSearchCV(SVC(), param_grid, cv=5) # Objet a entrainer et evaluer

    print("Test differences entre prediction et Y_test : ")
    print("{}".format(y_test - reg.predict(X_test)))
#    
#    
#    
#    plt.ion()
#    plt.figure("KNN vs True")
#    plt.plot(T.line_z, T_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
#    plt.plot(T.line_z, T_true, label='True', linestyle='--', c='k')
#    plt.legend()

#    plt.figure("beta KNN vs True")
#    plt.plot(T.line_z, beta_ML, marker='o', linestyle='none', fillstyle='none', c='purple', label="ML")
#    plt.plot(T.line_z, true_beta, label='True', linestyle='--', c='k')
#    plt.legend()