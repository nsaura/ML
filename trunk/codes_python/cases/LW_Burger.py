#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import append,size
import scipy.linalg as sp
import matplotlib.pyplot as plt

import pandas as pd
import os.path as osp

plt.ion() ; plt.close('all')
###----------------------------------------------------------------------------------------------
###-----------------------------------Quelques fonctions-----------------------------------------
###----------------------------------------------------------------------------------------------
def pd_write_csv(filename, data) :
    """
    Argument :
    ----------
    filename:   the path where creates the csv file;
    data    :   the data to write in the file
    """
    path = osp.join("./data", filename)
    pd.DataFrame(data).to_csv(path, index=False, header= True)
###--------------------------------------------------------------------###
def pd_read_csv(filename) :
    """
    Argument :
    ----------
    filename : the file's path with or without the extension which is csv in any way. 
    """
    if osp.splitext(filename)[-1] is not ".csv" :
        filename = osp.splitext(filename)[0] + ".csv"
    data = pd.read_csv(filename).get_values()
    data = pd.read_csv(filename).get_values()
#        print data
#        print data.shape
    if np.shape(data)[1] == 1 : 
        data = data.reshape(data.shape[0])
    return data
###--------------------------------------------------------------------###
def intermediaires (var, flux ,incr, r) :
    """ Fonction qui calcule les valeurs de la variables  aux points (n + 1/2 , i - 1/2) et (n + 1/2 , i + 1/2)   
        Parametres :
        -------------
        var : Celle dont on veut calculer l'étape intermediaire. C'est un tableau de valeur
        flux : le flux correspondant à la valeur var        
        incr : l'indice en cours indispensable pour evaluer l'etape intermediaire
        r : rapport dt/dx
        
        Retour : 
        ------------
        f_m = la valeur intermediaire au point (n + 1/2 , i - 1/2)
        f_p = la valeur intermediaire au point (n + 1/2 , i + 1/2)
    """
    f_m = 0.5 * ( var[incr] + var[incr-1] ) - 0.5 * r * ( flux[incr]- flux[incr-1] )
    f_p = 0.5 * ( var[incr+1] + var[incr] ) - 0.5 * r * ( flux[incr+1] - flux[incr] )
    return f_m,f_p
###--------------------------------------------------------------------###
def resolution(Nx=202, tf=10, L = float(3))  : 
    ## Déf des données du probleme 
    CFL = 1 
    dx = L/(Nx-1)
    dt =  dx * CFL
    r = dt/dx

    X = np.arange(0,L+dx,dx)

    bruits = [0.0005 * np.random.randn(Nx) for time in range(10)]

    for j, bruit in enumerate(bruits) :
        # Initialisation des champs u (boucles while)
        u = []
        u_nNext = [] 
        plt.close()
        for i in range(len(X)) :
            if X[i] >=0 and X[i] <=1 :
                u.append(1 + bruit[i])
            if X[i] > 1 :
                u.append(0 + bruit[i])
            i+=1
            
        fu = np.asarray([0.5*u_x**2 for u_x in u])
#            print ("size u it 0 = %d" %(np.size(u)))
#            print ("size fu it 0 = %d" %(np.size(fu)))

        # Tracés figure initialisation : 
        plt.figure("Resolution")
        plt.plot(X, u)
        plt.title("U vs X iteration 0 bruit %d" %(j))
        plt.ylim((-0.75, 1.4))
        plt.pause(0.01)
        
        pd_write_csv("u_it0_%d.csv" %(j), u)
            
        t = cpt = 0
        while t < tf :
            i=1
            while i <= Nx-2 :
                u_m, u_p = intermediaires(u, fu, i, r)
                
                fu_m =  0.5*u_m**2
                fu_p =  0.5*u_p**2
                        
                u_nNext.append( u[i] - r*( fu_p - fu_m ) + bruit[i] )
                fu = np.asarray([0.5*u_x**2 for u_x in u])
                                         
                i += 1  # On passe à la case d'après
            
#            print ("size u_nNext it 0 = %d" %(np.size(u_nNext)))
            
            u[1:Nx-1] = u_nNext  
            u_nNext  = []
            
            # Conditions aux limites 
            u[0] = u[-1]
            u[1] = u[0]
            u[-1]= u[-2]
            
#            print np.size(u)
            
            cpt += 1
            pd_write_csv("u_it%d_%d.csv" %(cpt, j), u)
            
            t+=dt # Itération temporelle suivante
            
            u = np.asarray(u) 
            
            if cpt % 10 == 0 :
                plt.clf()
                plt.plot(X, u, c='k') 
                plt.title("u vs X, iteration %d bruit %d" %(cpt, j)) 
                plt.ylim((-0.75, 1.4))  
                plt.pause(0.01)
    return cpt
###--------------------------------------------------------------------###
def see_moy_u(cpt=202, Nx = 202, L = float(3)) :
    u = dict()
    CFL = 1 
    dx = L/(Nx-1)
    dt =  dx * CFL
    r = dt/dx

    X = np.arange(0,L+dx,dx)
    
    plt.figure("Evolution moyenne")
    
    for it in range(cpt+1) :
        u["it_%d" %(it)] = np.zeros((Nx))
    
        for b in range(10) :
            u_curr = pd_read_csv("./data/u_it%d_%d.csv" %(it, b))
    
            for i in range(len(u_curr)) :
                u["it_%d"%(it)][i] += u_curr[i] / 10.
        
        plt.clf()
        plt.plot(X, u["it_%d"%(it)], c='k') 
        plt.title("u vs X, iteration %d moyenne des bruits" %(it)) 
        plt.ylim((-0.75, 1.4))  
        plt.pause(0.1)        

