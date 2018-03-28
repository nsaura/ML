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
def resolution_diff(Nx=202, tf=10, L = float(3), write=False)  : 
    ## Déf des données du probleme 
    CFL = 0.45 # cfl <= 0.5 
    dx = L/(Nx-1)
    dt =  dx * CFL

    D = 1*10**(-5)
    dt_v = dx**2 / D * CFL
    
    if dt < dt_v :
        print "CFL lineaire"
    else:
        print "CFL visqueux"      
        dt = dt_v
        
    print dt 
    
    print D*dt/dx**2
    
    r = dt/dx
    
    X = np.arange(0,L+dx,dx)

    bruits = [0.0005 * np.random.randn(Nx) for time in range(10)]
    
    for j, bruit in enumerate(bruits) :
        # Initialisation des champs u (boucles while)
        u, u_nNext = [], []
        plt.close()
#        for i in range(len(X)) :
#            if X[i] >=0 and X[i] <=1 :
#                u.append(1 + bruit[i])
#            if X[i] > 1 :
#                u.append(0 + bruit[i])
#            i+=1
        u = np.sin(2*np.pi/L*X) + bruit
        fu = np.asarray([0.5*u_x**2 for u_x in u])
#        print ("size u it 0 = %d" %(np.size(u)))
#        print ("size fu it 0 = %d" %(np.size(fu)))
        
        
        # Tracés figure initialisation : 
        plt.figure("Resolution")
        plt.plot(X, u)
        plt.title("U vs X iteration 0 bruit %d" %(j))
        plt.ylim((-0.75, 1.4))
        plt.pause(0.01)
        
        if write == True : pd_write_csv("u_it0_%d.csv" %(j), u)
            
        t = cpt = 0
        while t < tf :
            i=1
            fu = []
#            fu.append(0.5*u[0]**2 - D/dx*(u[0] - u[-1]))
#            fu.extend([0.5*u[k]**2 - D/dx*(u[k] - u[k-1]) for k in range(len(u))])
#            fu = np.array(fu)
##            print fu.size
            fu = np.asarray([0.5*u_x**2 for u_x in u])

            der_sec = [D*dt/dx**2*(u[k+1] - 2*u[k] + u[k-1]) for k in range(1, len(u)-1)]
            der_sec.insert(0, D*dt/dx**2*(u[1] - 2*u[0] + u[-1]))
            der_sec.insert(len(der_sec), D*dt/dx**2*(u[0] - 2*u[-1] + u[-2]))

            while i <= Nx-2 :
                u_m, u_p = intermediaires(u, fu, i, r)
                
                fu_m =  0.5*u_m**2
                fu_p =  0.5*u_p**2

                u_nNext.append( u[i] - r*( fu_p - fu_m ) + bruit[i] + der_sec[i] )

                i += 1  # On passe à la case d'après
            
#            print ("size u_nNext it 0 = %d" %(np.size(u_nNext)))
            u[1:Nx-1] = u_nNext  
            u_nNext  = []
            
            # Conditions aux limites 
            u[0] = u[-1]
            u[-1]= u[1]
            
            cpt += 1
            if write == True : pd_write_csv("u_it%d_%d.csv" %(cpt, j), u)
            
            t += dt # Itération temporelle suivante

            u = np.asarray(u) 
            
            if cpt % 20 == 0 :
                plt.clf()
                plt.plot(X, u, c='k') 
                plt.title("u vs X, iteration %d bruit %d" %(cpt, j)) 
                plt.ylim((-0.75, 1.4))  
                plt.pause(0.1)
            if cpt == 500 :
                break
###--------------------------------------------------------------------###
def see_moy_u_diff(cpt=202, Nx = 202, L = float(3)) :
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
###--------------------------------------------------------------------###
def crank_nicholson(Nx=202, L = float(3), itmax = 10, write=False):
    dx = L/(Nx-1)
    CFL = 0.2
    nu = 1.*10**(-2)
    
    dt = {"dt_v" : CFL / nu * dx**2,
          "dt_l" : CFL*dx}

    if dt["dt_v"] < dt["dt_l"] :
        dt = dt["dt_v"]
        print ("dt visqueux")
    else :
        dt = dt["dt_l"]
        print ("dt lineaire")
    
    fac = nu * dt / dx**2 
    
    r = dt / dx
    X = np.arange(0,L+dx,dx)
    
    INF1 = np.diag(np.transpose([-fac/2 for i in range(Nx-3)]), -1)
    SUP1 = np.diag(np.transpose([-fac/2 for i in range(Nx-3)]), 1) 
    A_diag1 = np.diag(np.transpose([(1 + fac) for i in range(Nx-2)])) 

    INF2 = np.diag(np.transpose([fac/2 for i in range(Nx-3)]), -1) 
    SUP2 = np.diag(np.transpose([fac/2 for i in range(Nx-3)]), 1)
    A_diag2 = np.diag(np.transpose([(1 - fac) for i in range(Nx-2)]))
    
    SCDSUP = np.diag(np.transpose([-r for i in range(Nx-3)]), 1)
    SCD = np.diag(np.transpose([r for i in range(Nx-2)]))
    
    A1 = A_diag1 + INF1 + SUP1
    A2 = A_diag2 + INF2 + SUP2
    A3 = SCD + SCDSUP    
    
#    A1 = np.zeros((Nx,Nx))
#    A2 = np.zeros((Nx,Nx))
#    A3 = np.zeros((Nx,Nx))
#    
#    A1[0,0] = A1[-1,-1] = 1
#    A2[0,-2] = A2[-1,1] = 1
#    
#    A1[1:Nx-1, 1:Nx-1] = In1
#    A2[1:Nx-1, 1:Nx-1] = In2
#    A3[1:Nx-1, 1:Nx-1] = InSCD
    
    ## On fait la résolution sur A1, A2, A3 qui prennent en compte les conditions aux limites
    bruits = [0.0005 * np.random.randn(Nx) for time in range(5)]
    bruit = bruits[np.random.randint(5)]
    
#    u_n = np.sin(2*np.pi/L*X)
    u= []
#    for i in range(len(X)) :
#        if X[i] >=1 and X[i] <=1.4 :
#            u.append(1 + bruit[i])
#        else :
#            u.append(0 + bruit[i])
#        i+=1    
    u_n = np.sin(2*np.pi/(L-dx)*(X-dx)) 

    # Tracés figure initialisation : 
    plt.figure("Resolution")
    plt.plot(X, u_n)
    plt.title("U vs X iteration 0 bruit %d" %(0))
    plt.ylim((-2.5, 2.4))
    plt.pause(0.01)

    vec_ci = np.zeros((Nx-2))
    for it in range(itmax) :

        vec_ci[0] = fac/2 * u_n[-2]
        vec_ci[-1]= fac * u_n[1] 
        
        u_n_2 = np.array([0.5*ux**2 for ux in u_n])
        
        u_n_tmp = np.dot(A2, u_n)
        scd_term = np.dot(A3, u_n_2)
        
        B_n = np.zeros((Nx))
         
        for i in range(Nx-2) :
            B_n[i] = u_n_tmp[i] + scd_term[i]

        u_nNext = np.dot(np.linalg.inv(A1), B_n)
        
        u_n = u_nNext
        print it 
        print u_n
        
        if it % 1 == 0 :
            plt.clf()
            plt.plot(X, u_n, c='k') 
            plt.title("u vs X, iteration %d bruit %d" %(it, 0)) 
            plt.ylim((-1.5, 1.5))  
            plt.pause(0.1)
        if it == 500 :
            break
            
    return u_nNext

