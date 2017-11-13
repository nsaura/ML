#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys

from numpy.linalg import norm as norm #norme euclidienne : norm(vec, 2)

plt.ion()
plt.close('all')
###----------------------------------------------------------------------------------------
#def normal(mu, sigma) :
#    return  lambda T : 1./(np.sqrt(2*np.pi*sigma))*np.exp(-0.5*((T-mu)/sigma)**2)
#    
####----------------------------------------------------------------------------------------
####----------------------------------------------------------------------------------------
#def epsilonT(T, noise_z_term) :
#    """     
#        To calculate epsilon terme more quickly.
#        
#        Arg :
#        ----
#        T               :   Temperature 
#        noise_z_term    :   Gaussian noise associated with the z-location
#    """
#    print 10**(-4)*(1.+5.*np.sin(3.*T*np.pi/200.) + np.exp(0.02*T) + noise_z_term)
#    return 10**(-4)*(1.+5.*np.sin(3.*T*np.pi/200.) + np.exp(0.02*T) + noise_z_term)
####----------------------------------------------------------------------------------------
#def F(T, noise_z_term ,T_inf, h):
#    T4_inf = T_inf**4
#    return epsilonT(T, noise_z_term)*(T_inf**4 - T**4) + h*T_inf
###----------------------------------------------------------------------------------------

def tab_normal(mu, sigma, length) :
    return sigma * np.random.randn(length) + mu, (sigma * np.random.randn(length) + mu).mean() , (sigma * np.random.randn(length) + mu).std()

np.random.seed(500) # To keep the same random generator
z_init, z_final = 0., 1.
N_discr = 33
kappa=0.1

line_z = np.linspace(z_init,z_final,N_discr)
print(line_z[0])

dz = np.abs(z_init-z_final)/float(N_discr)
dt = 0.001

h = 0.5 

M1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Extra inférieure
P1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Extra supérieure
A_diag = np.diag(np.transpose([(1+( 2.0)*dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

A = A_diag + M1 + P1 #Construction de la matrice des coefficients

gauss_z = tab_normal(0,0.1,N_discr-2)[0] # Quadrillement du bruit

T_inf_lst = [i*5 for i in xrange(1, 11)]

#plt.figure()

T_init, T_nNext_lst = [], []

verbose = False
for T_inf in T_inf_lst :

    T_n =  map(lambda x : -4*T_inf*x*(x-1), line_z[1:N_discr-1])) # Profil de température initiale
    T_n_2 = map(lambda x : -4*T_inf*x*(x-1), line_z[1:N_discr-1])) # Profil de température initiale
    
    T_init.append(T_n)
    #T_n =  list(map(lambda x : T_inf, line_z[1:N_discr-1])) # Profil de température initiale

#    print(T_n)
    #print('line')
    #print(line_z)
#    if verbose == True :
#        plt.plot(line_z[1:N_discr-1], T_n, label='T_inf = %.2f,  Init sans bruit' %(T_inf))  
#        plt.plot(line_z[1:N_discr-1], T_n + gauss_z, label='T_inf = %.2f,  Init avec bruit' %(T_inf))

    #T_nNext = np.zeros((N_discr,1))
    #T_n = np.zeros((N_discr,1))

    T_nNext = T_n
    T_nNext_2 = T_n_2
    
    tol = 1e-2
    err, err_2, compteur = 1., 1.0, 0

    while (np.abs(err) > tol) and (compteur <800) and (np.abs(err_2) > tol):
        if compteur > 0 :
            T_n = T_nNext
            T_n_2 = T_nNext_2
        compteur += 1
#        print(compteur)
        
        #    B_n = np.zeros((N_discr,1))
        B_n    =   T_n
        B_n_2  =   T_n_2
         
        for i in range(1,N_discr-2) :
            B_n[i] = T_n[i]+dt*((10**(-4)*(1.+5.*np.sin(3.*T_n[i]*np.pi/200.) + np.exp(0.02*T_n[i]) + gauss_z[i]))*(T_inf**4-T_n[i]**4)+h*(T_inf-T_n[i]))   
            B_n_2[i] = T_n_2[i]+dt*5*10**(-4)*(T_inf**4-T_n_2[i]**4)  

        T_nNext = np.dot(np.linalg.inv(A), np.transpose(B_n))
        T_nNext_2 = np.dot(np.linalg.inv(A), np.transpose(B_n_2))
        
        err = np.linalg.norm(T_nNext - T_n, 2)
        err_2 = np.linalg.norm(T_nNext_2 - T_n_2, 2)
        
    T_nNext_lst.append(T_nNext)
    
    df = pd.DataFrame(T_nNext)
    df.to_csv("./data/T_inf_{}.csv".format(T_inf), index=False, header=True)
    
    df_2 = pd.DataFrame(T_nNext_2)
    df_2.to_csv("./data/prior_T_inf_{}.csv".format(T_inf), index=False, header=True)
    
    if verbose == True :
        plt.plot(line_z[1:N_discr-1], T_nNext, label='Convergence Exact -- {}'.format(T_inf))
        plt.plot(line_z[1:N_discr-1], T_nNext_2, label='Convergence Prior  -- {}'.format(T_inf), linestyle='--', marker='s', markerfacecolor='none', markersize=7 )
        plt.legend(loc='best', ncol=2, fontsize=7)
    
    print "T_inf = {} finie".format(T_inf)
    


