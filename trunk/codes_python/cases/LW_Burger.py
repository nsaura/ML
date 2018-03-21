#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import append,size
import scipy.linalg as sp
import matplotlib.pyplot as plt

plt.ion() ; plt.close('all')
###----------------------------------------------------------------------------------------------
###-----------------------------------Quelques fonctions-----------------------------------------
###----------------------------------------------------------------------------------------------
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
    
###-------------------------------------------------------------------------------------------------
def Fws (W1, W2, W3, gamma) :
    """ calcule la valeur des flux fw2 et fw3 au temps n+1/2
        Parametres :
        ------------ 
        W1, W2, W3 : valeurs des Wi pour calculer les flux intermediaires 
        gamma : le rapport des chaleur specifiques
        
        Retours :
        ------------
        FW2, FW3 : les flux au temps n+1/2
    """
    g1 = (gamma -1.) ; WW = W2 / W1
    fw2_int = WW * W2 * (1 - 0.5 * g1 ) + g1 * W3
    fw3_int = (W3 *(1 + g1) - WW * W2 * g1 / 2.0) * WW
    
    return fw2_int, fw3_int   
    
###-------------------------------------------------------------------------------------------------    
def tracage (densite, vitesse, pression, axe, nom_figure, borne=4) :
    """ Trace un subplot de trois courbes centrées en une valeur borne.
        Parametres :
        ------------
        3 tableaux de trois grandeurs a tracer
        L'axe sur lequel tracer ces courbes 
        Le nom de la figure que l'on veut afficher (pour ne pas avoir de probleme de recouvrement des figures)
        la valeur de borne est optionnelle, initialisee a 4
    """
    plt.ion()
    plt.figure(nom_figure)
    plt.subplot(311)
    plt.xlim((-borne,borne))
    plt.plot(axe,densite) 
    plt.title('Densite')

    plt.subplot(312)
    #plt.xlim((-2,2))
    plt.xlim((-borne,borne))
    plt.plot(axe,vitesse, color='r')
    plt.title('Vitesse')

    plt.subplot(313)
    #plt.xlim((-2,2))
    plt.xlim((-borne,borne))
    plt.plot(axe,pression,color='g')
    plt.title('pression')

    plt.show()
    
###-------------------------------------------------------------------------------------------------
def trouve_pic (grandeur, axe) :
    grad = np.gradient(grandeur)
    exeception_indice = []
    for k in range(len(grandeur)) :
        if abs(grad[k] - grad[k-1]) >0.08 :
            exeception_indice.append(k)
    return exeception_indice
    
###-------------------------------------------------------------------------------------------------    
def rankine_hugoniot (p1, r1, u1, c1, gamma=1.4) :
    """ Fonction qui calcule les sauts de pression et de densité à partir des valeurs de pression, densité et mach (au travers la vitesse de l'ecoulement et celle du son) afin de vérifier les résultats numériques
    Parametres : 
    -----------
    p1 : pression dans le milieu amont du choc  N.m-2 (Pascal)
    r1 : densite dans le milieu amont du choc   kg.m-3
    u1 : vitesse de l'ecoulement a l'amont      m.s-1
    c1 : vitesse du son a l'amont du choc       m.s-1
    gamma : rapport des capacité calorifiques (optionnel intialise a 1.4)
    
    Retours :
    p2 : pression dans le milieu aval du choc   N.m-2
    r2 : densite dans le milieu l'aval du choc  kg.m-3             
    """
    M1 = u1/c1    
    # Rankine Hugoniot
    p2 = p1 * (1. + (2. * gamma) / (gamma + 1.) * (M1**2 - 1))
    r2 = r1 *  ( (gamma + 1 ) * (M1**2) ) / ( (gamma - 1.) * M1**2  + 2)
    
    return p2, r2
###--------------------------------------------------------------------###
###--------------------------------------------------------------------###

## Déf des données du probleme 
Nx = 202
L = float(3)

CFL = 1 

dx = L/(Nx-1)
dt =  dx * CFL

X = np.arange(0,L,dx)
sx = np.size(X)

tf = 10

r = dt/dx

u = [] 
u_nNext = []

# Initialisation des incréments (boucles while)
i, t= 0, 0

###### ------ intialisation premier incrément temporel ------ ######
for x in X :
    if x >=0 and x<=1 :
        u.append(1)
    if x>1 :
        u.append(0)

fu = np.asarray([0.5*u_x**2 for u_x in u])
###### --------- Fin intialisation --------- ######

# Tracés figure initialisation : 
plt.figure("Processing...")
plt.plot(X, u)
plt.title("U vs X")
plt.pause(0.01)

###### ------ Résolution t>0 ------ ######
i,j = 1,1
#w1_m_list, w1_p_list, w2_m_list, w2_p_list, w3_m_list, w3_p_list = [], [], [], [], [], [] ##Pour debbuger
cpt = 0

bruit = 0.01 * np.random.randn(Nx-2) + 0

while t < tf :
    while i < sx-1 :
        u_m, u_p = intermediaires(u, fu, i, r)
        
        fu_m =  0.5*u_m**2
        fu_p =  0.5*u_p**2
                
        u_nNext.append( u[i] - r*( fu_p - fu_m ) + bruit[i] )
        fu = np.asarray([0.5*u_x**2 for u_x in u])
                                 
        i += 1  # On passe à la case d'après
    
#    print(fw1, fw2, fw3) ## Pour debbuger
    u[1:sx-1] = u_nNext  
    u_nNext  = []
    
    # Conditions aux limites 
    u[0] = u[-1]
    u[1] = u[0]
    u[-1]= u[-2]
    
    i=1
    t+=dt # Itération temporelle suivante
    
    u = np.asarray(u) 
    if cpt % 10 == 0 :
        plt.clf()
        plt.plot(X, u, c='k')    
        plt.pause(1)

    cpt += 1
###------------------------- Fin du programme -------------------------###
###--------------------------------------------------------------------###
###--------------------------------------------------------------------###
# -- Essai de Filtrage pour éviter les pics. Abandonné -- #
#rho_filt = w1



#pp_filt = g1 * w3 - 0.5 * g1 * w2**2 / w1
#u_filt = w2 / w1
#rho_filt = np.asarray(rho_filt)

#autr_execp = []
#uexeption, pexeption, rexeption = [], [], []

#rexeption, pexeption, uexeption = trouve_pic(w1,X), trouve_pic(P_tf,X), trouve_pic(u_tf, X) 

###-------------------------------------
######### Filtrage des résultats #########
# Filtrage de la vitesse
#for k in range(len(w1)) :
#    if u_filt[k] > 9.25 or u_filt[k] < 9.25 and u_filt[k] > 9.0: # Test sur la vitesse
#        uexeption.append(k)

## Filtrage de la pression et de la densité
#for k in range(len(w1)-1) :
#    if abs(pp_filt[k]-pp_filt[k-1])>1.5 :
#        if X[k] > -0.01 and X[k] < 1.7 :
#            pp_filt[k] = pp_filt[k-1]
#    
#    if abs(rho_filt[k]-rho_filt[k-1]) > 0.008 :
#        if (X[k] >-0.001 and X[k] < 0.68) or (X[k] > 0.9 and X[k] < 1.63) :
#            rho_filt[k] = rho_filt[k-1]
#    
#    if abs(u_filt[k] - u_filt[k-1]) > 2 :
#        if (X[k] >-0.001 and X[k] < 1.60) :
#            u_filt[k] = u_filt[k-1]            

######### Résultats filtrés #########
#if trace_filtre == True :
#    fig2 = 'Resultats FILTRES avec tf=%.3f Nx=%.1f et Nt=%.1f' %(tf,Nx,Nt)
#    tracage(rho_filt, u_filt, pp_filt, X, fig2, borne)



