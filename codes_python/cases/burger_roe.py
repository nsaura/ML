#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Shout out to https://math.unice.fr/~ribot/enseignement/m2/TP3.pdf

plt.ion()

###############################
# --------------------------- #
# --------------------------- #
######## Schema de Roe ########
# --------------------------- #
# --------------------------- #
###############################

#----------------------------------------------
def which_A (u, v, f, fprime) :
    """
    Determine la forme de A; retourne la bonne valeur 

    Args :
    -------
    u : valeur de u a i     (reelle)
    v : valeur de u a i+1   (reelle)
    f : fonction de flux    (lambda)
    fprime : derivee de la fonction de flux (lambda)
    """
    val = (f(u) - f(v)) / (u-v)
    if val > 0 :
        return val
    else :
        return fprime(u)
#----------------------------------------------
def which_F (u, v, f, fprime) :
    """
    Determine la forme du flux; retourne la bonne valeur
    Utilise which_A
    
    Args :
    -------
    u : valeur de u a i     (reelle)
    v : valeur de u a i+1   (reelle)
    f : fonction de flux    (lambda)
    fprime : derivee de la fonction de flux (lambda)    
    """ 
    val = which_A(u, v, f, fprime)
    if val > 0 :
        return f(u)    
    else :
        return f(v)
#----------------------------------------------
def timestep_roe (u, j, r, f, fprime) :
    """
    Calcule u^n+1_i utilisant la formule du pas de temps de Roe 	
    Utilise which_A et which_F
    
    Args :
    -------
    u : vecteur de vitesse   
    j : indice de base   
    r : rapport dt/dx
    f : fonction de flux    (lambda)
    fprime : derivee de la fonction de flux (lambda)    
    """ 
    F_ip = which_F(u[j], u[j+1], f, fprime)
    F_im = which_F(u[j-1], u[j], f, fprime)
    
    return u[j] - r*(F_ip - F_im)
#----------------------------------------------    
def doitROE(L ,tf, Nx, Nt, f, fprime, type_init = "sin") :
    """
    Fonction qui resout un probleme avec un schema de Roe
    
    Args :
    -------
    L   :   Longueur du domaine
    tf  :   Temps final
    Nx  :   Nombre de points de discretisation spatiale
    Nt  :   Nombre de points de discretisation temporelle
    f       :   fonction de flux    (lambda)
    fprime  :   derivee de la fonction de flux (lambda) 
    type_init : Type d initialisation
    """
    dx = float(L) / (Nx-1)
    X = np.arange(0, L+dx, dx) # X.shape=Nx
    
    dt = float(tf) / (Nt-1)
    T = np.arange(0, tf+dt, dt)
    
    r = dt / dx
    
    u = np.zeros_like(X)
    if type_init == "choc" :
        for i, x in enumerate(X) :
            u[i] = 0 if x < L/2. else 1
            
    u_nNext = np.zeros_like(u)
    
    t=0.
    plt.figure("Evolution solution Schema de Roe")
    plt.plot(X, u, label="Solution t = %.4f" %t)
    plt.ylim((-1.1, 1.1))
    plt.legend()
    
    plt.pause(1)
   
    for cnt in range(len(T)-1) :
        for j in range(1, len(X[:-1])) :
           u_nNext[j] = timestep_roe(u, j, r, f, fprime)
        u_nNext[0] = u_nNext[-2]
        u_nNext[-1] = u_nNext[1]
        
        t += dt
        
        if cnt % 5 == 0:
            if cnt % 15 == 0 and t > 0 :
                plt.clf() 
                print u_nNext
                
            plt.figure("Evolution solution Schema de Roe")
            plt.plot(X, u_nNext, label="Solution t = %.4f" %t)
            plt.ylim((-1.1, 1.1))

            plt.legend()
            plt.pause(0.5)
            
        u = u_nNext

#doitROE(1, 1, 100, 3000, lambda u : 0.5*u**2, lambda u : u, type_init="choc")
#----------------------------------------------
###############################
# --------------------------- #
# --------------------------- #
######## Schema de LWe ########
# --------------------------- #
# --------------------------- #
###############################
#----------------------------------------------
def F_LW(u, j, r, f) :
    """
    Calcule le flux selon la forme de Lax Wendroff
    
    Args :
    -------
    u : vecteur de vitesse   
    j : indice de base   
    r : rapport dt/dx
    f : fonction de flux    (lambda)
    """
    return 0.5 * (f(u[j]) + f(u[j+1]) - r * 0.5 * (u[j+1] + u[j]) * (f(u[j+1]) - f(u[j])))
#----------------------------------------------    
def timestep_LW (u, j, r, f) :
    """
    Calcule u^n+1_i utilisant la formule du pas de temps de Roe 	
    Utilise which_A et which_F
    
    Args :
    -------
    u : vecteur de vitesse   
    j : indice de base   
    r : rapport dt/dx
    f : fonction de flux    (lambda)
    """     
    F_ip = F_LW(u, j, r, f)
    F_im = F_LW(u, j-1, r, f)
    
    return u[j] - r*(F_ip - F_im)   
#----------------------------------------------
def doitLW(L ,tf, Nx, Nt, f, type_init = "sin") :
    """
    Fonction qui resout un probleme avec un schema de Roe
    
    Args :
    -------
    L   :   Longueur du domaine
    tf  :   Temps final
    Nx  :   Nombre de points de discretisation spatiale
    Nt  :   Nombre de points de discretisation temporelle
    f       :   fonction de flux    (lambda)
    type_init : Type d initialisation
    """
    dx = float(L) / (Nx-1)
    X = np.arange(0, L+dx, dx) # X.shape=Nx
    
    dt = float(tf) / (Nt-1)
    T = np.arange(0, tf+dt, dt)
    
    r = dt / dx
    
    u = np.zeros_like(X)
    if type_init == "choc" :
        for i, x in enumerate(X) :
            u[i] = 0. if x < L/2. else 1.
            
    u_nNext = np.zeros_like(u)
    
    t=0.
    plt.figure("Evolution solution Schema de Roe")
    plt.plot(X, u, label="Solution t = %.4f" %t)
    plt.ylim((-1.1, 1.1))
    plt.legend()
    
    plt.pause(1)
    
    print u.shape
    print type(f)
    
    for cnt in range(len(T)-1) :
        for j in range(1, len(X[:-1])) :
           u_nNext[j] = timestep_LW(u, j, r, f)
        u_nNext[0] = u_nNext[-2]
        u_nNext[-1] = u_nNext[1]
        t += dt
        
        if cnt % 5 == 0:
            if cnt % 15 == 0 and t > 0 :
                plt.clf() 
                print u_nNext
                
            plt.figure("LW")
            plt.plot(X, u_nNext, label="Solution t = %.4f" %t)
            plt.ylim((-1.1, 1.1))

            plt.legend()
            plt.pause(0.5)
            
        u = u_nNext
#----------------------------------------------
def compa_Roe_LW(L ,tf, Nx, Nt, f, fprime, type_init = "sin") :
    """
    Fonction qui compare les solution du schema de Roe et de Lax Wendroff
    
    Args :
    -------
    L   :   Longueur du domaine
    tf  :   Temps final
    Nx  :   Nombre de points de discretisation spatiale
    Nt  :   Nombre de points de discretisation temporelle
    f       :   fonction de flux    (lambda)
    fprime  :   derivee de la fonction de flux (lambda) 
    type_init : Type d initialisation
    """
    dx = float(L) / (Nx-1)
    X = np.arange(0, L+dx, dx) # X.shape=Nx
    
    dt = float(tf) / (Nt-1)
    T = np.arange(0, tf+dt, dt)
    
    r = dt / dx
    
    u = np.zeros_like(X)
    if type_init == "choc" :
        for i, x in enumerate(X) :
            u[i] = 0 if x < L/2. else 1
            
    u_lw = np.copy(u)
    u_roe = np.copy(u)
    
    u_nNext_Roe = np.zeros_like(u)
    u_nNext_Lw = np.zeros_like(u)
    
    t=0.
    plt.figure("Evolution solution Schema de Roe")
    plt.plot(X, u, label="Solution t = %.4f" %t)
    plt.ylim((-1.1, 1.1))
    plt.legend()
    
    plt.pause(1)
   
    for cnt in range(len(T)-1) :
        for j in range(1, len(X[:-1])) :
           u_nNext_Lw[j] = timestep_LW(u_lw, j, r, f)
           u_nNext_Roe[j] = timestep_roe(u_roe, j, r, f, fprime)

        u_nNext_Lw[0] = u_nNext_Lw[-2]
        u_nNext_Lw[-1] = u_nNext_Lw[1]
        
        u_nNext_Roe[0] = u_nNext_Roe[-2]
        u_nNext_Roe[-1] = u_nNext_Roe[1]
        
        t += dt
        
        if cnt % 5 == 0:
            if cnt % 15 == 0 and t > 0 :
                plt.clf() 
                print ([abs(u_nNext_Lw[k] - u_nNext_Roe[k]) for k in range(len(u_nNext_Lw)-1)])
                
            plt.figure("Comparaison solutions Schema de Roe vs LW")
            plt.plot(X[1 :len(X[:-1])], u_nNext_Roe[1 :len(X[:-1])], label="ROE Solution t = %.4f " %t, color = 'k')
            plt.plot(X[1 :len(X[:-1])], u_nNext_Lw[1 :len(X[:-1])], label="LW Solution t = %.4f " %t, color = 'blue', fillstyle = 'none', marker='o', linestyle='none')
            plt.ylim((-0.4, 1.6))

            plt.legend()
            plt.pause(0.5)
            
        u_lw = u_nNext_Lw
        u_roe = u_nNext_Roe
