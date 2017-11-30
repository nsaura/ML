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

np.random.seed(1000) # To keep the same random generator
z_init, z_final = 0., 1.

N_discr = 100
kappa=0.10
line_z = np.linspace(z_init,z_final,N_discr)
dz = np.abs(z_init-z_final)/float(N_discr)
dt = 1e-9

CFL = kappa * dt / dz **2
print("CFL = %.8f" %(CFL))

h = 0.5 

M1 = np.diag(np.transpose([dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Extra inférieure
P1 = np.diag(np.transpose([dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Extra supérieure
A_diag1 = np.diag(np.transpose([(1- dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

A1 = A_diag1 + M1 + P1 #Construction de la matrice des coefficients

M2 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Extra inférieure
P2 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Extra supérieure
A_diag2 = np.diag(np.transpose([(1+ dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

A2 = A_diag2 + M2 + P2 #Construction de la matrice des coefficients

lst_gauss = [tab_normal(0,0.1,N_discr-2)[0] for i in range(50) ] # Quadrillement du bruit

#T_inf_lst = [i*5 for i in xrange(1, 11)]
T_inf_lst = [50]
#plt.figure()

T_init, T_nNext_lst = [], []

verbose = True

for T_inf in T_inf_lst :
    for it, bruit in enumerate(lst_gauss) :
        T_n =  np.asarray(map(lambda x : -4*T_inf*x*(x-1), line_z[1:N_discr-1]))  # Profil de température initiale
        T_n_2 = np.asarray(map(lambda x : -4*T_inf*x*(x-1), line_z[1:N_discr-1])) # Profil de température initiale
#        
#        T_n     =   T_n.reshape(N_discr-2, -1)
#        T_n_2   =   T_n_2.reshape(N_discr-2, -1)
        
        T_init.append(T_n)

        T_nNext = T_n
        T_nNext_2 = T_n_2
    
        tol = 1e-2
        err, err_2, compteur = 1., 1.0, 0
        B_n = np.zeros((N_discr-2,1))
        B_n_2 = np.zeros((N_discr-2,1))
        
        while (np.abs(err) > tol) and (compteur <800) and (np.abs(err_2) > tol):
            if compteur > 0 :
                T_n = T_nNext
                T_n_2 = T_nNext_2
            compteur += 1
    #        print(compteur)
            
            
            T_n    =   np.dot(A2,T_n)
            T_n_2  =   np.dot(A2, T_n_2)
             
            for i in range(N_discr-2) :
                B_n[i] = T_n[i]+dt*((10**(-4)*(1.+5.*np.sin(3.*T_n[i]*np.pi/200.) + np.exp(0.02*T_n[i]) + bruit[i]))*(T_inf**4-T_n[i]**4)+h*(T_inf-T_n[i]))
                   
                B_n_2[i] = T_n_2[i]+dt*5*10**(-4)*(T_inf**4-T_n_2[i]**4)*(1+bruit[i]) 

            T_nNext = np.dot(np.linalg.inv(A1), B_n)
            T_nNext_2 = np.dot(np.linalg.inv(A1), B_n_2)
            
            err = np.linalg.norm(T_nNext - T_n, 2)
            err_2 = np.linalg.norm(T_nNext_2 - T_n_2, 2)
            
        T_nNext_lst.append(T_nNext)
    
        path_abs = os.path.abspath(os.path.curdir)
        
        if os.path.exists(os.path.join(path_abs, "data")) == False :
            os.mkdir(os.path.join(path_abs, "data"))
        
        pf_obs = os.path.join(path_abs, 'data', 'obs_T_inf_{}_{}.csv'.format(T_inf, it))
        pf_prior = os.path.join(path_abs, 'data', 'prior_T_inf_{}_{}.csv'.format(T_inf, it))

        df = pd.DataFrame(T_nNext)
        df.to_csv(pf_obs, index=False, header=True)
    
        df_2 = pd.DataFrame(T_nNext_2)
        df_2.to_csv(pf_prior, index=False, header=True)
    
        if verbose == True :
            plt.plot(line_z[1:N_discr-1], T_nNext, label='Convergence Exact -- {}'.format(T_inf))
            plt.plot(line_z[1:N_discr-1], T_nNext_2, label='Convergence Prior  -- {}'.format(T_inf), linestyle='--', marker='s', markerfacecolor='none', markersize=7 )
            plt.legend(loc='best', ncol=2, fontsize=7)
            plt.title("Comparaison pour kappa = %.3f" %(kappa))
    print ("T_inf = {} finie".format(T_inf))
    


