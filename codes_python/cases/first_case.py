#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv, os, sys

from numpy.linalg import norm as norm #norme euclidienne : norm(vec, 2)

plt.ion()

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
N_discr = 31

line_z = np.linspace(z_init,z_final,N_discr)

dz = np.abs(z_init-z_final)/float(N_discr)

T_inf, h = 20.0, 0.5 

M1 = np.diag(np.transpose([1/dz**2 for i in xrange(N_discr-1)]), -1) # Extra inférieure
P1 = np.diag(np.transpose([1/dz**2 for i in xrange(N_discr-1)]), 1)  # Extra supérieure
A_diag = np.diag(np.transpose([(-h + 2.0/dz**2) for i in xrange(N_discr)])) # Diagonale

A = 10**-1*(A_diag + M1 + P1) #Construction de la matrice des coefficients

gauss_z = tab_normal(0,0.1,N_discr)[0] # Quadrillement du bruit

T_n =  map (lambda x : -4*T_inf*x*(x-1), line_z) # Profil de température initiale

verbose = True
if verbose == True :
    plt.figure()
    plt.plot(line_z, T_n, label='T_inf = %.2f,  Init sans bruit' %(T_inf))  
    plt.plot(line_z, T_n + gauss_z, label='T_inf = %.2f,  Init avec bruit' %(T_inf))

T_nNext = np.zeros((N_discr,1))

tol = 1e-6
err, compteur = 1., 0

while np.abs(err) > tol :
    if compteur > 0 :
        T_n = T_nNext
    compteur += 1
    
    B_n = np.zeros((N_discr,1))
    for i in xrange(N_discr) :
        B_n[i] = -(10**(-4)*(1.+5.*np.sin(3.*T_n[i]*np.pi/200.) + np.exp(0.02*T_n[i]) + gauss_z[i])*(T_inf**4-T_n[i]**4) + h*T_inf)
    
    print ("Compteur = {} B_n: \n{}".format(compteur, B_n))
#    T_nNext = np.linalg.solve(A, B_n) # Ils sont identiques 
    T_nNext = np.dot(np.linalg.inv(A), B_n)
    
    print ("Compteur = {} T-nNext: \n{}".format(compteur, T_nNext))
    err = norm(T_nNext - T_n, 2)
 
plt.plot(line_z, T_nNext, label='Apres \' convergence \'')
plt.legend(loc='best')
plt.show()
