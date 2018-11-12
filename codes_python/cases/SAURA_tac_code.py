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
Nx = 100; Nt = 50; L = 20; tf = 0.01; ## Peut nécessiter un certain temps de calcul
gamma = 1.4 # Pour l'air
g1 = (gamma -1.) 

dx = float(L)/Nx; dt = float(tf/Nt); r = dt/dx
T = np.arange(0.,tf,dt) ; X = np.arange(-L/2.,L/2.+dx,dx)
sx, st = np.size(X), np.size(T)

## Booléens pour les tracés 
trace_initial = True
trace_final = True
trace_filtre = False

borne = 5 # Borne pour le tracage
## Listes
rho, u, p = [], [], []          ### On a rajouté rho pour le filtrage
w1, w2, w3      = [], [], []    ### Définition des wi 
fw1 ,fw2, fw3   = [], [], []    ### Déf des flux associés aux wi

w1_tp, w2_tp, w3_tp = [], [], []

####            rho fait office de w1    ####
####            w2 = fw1                 ####

# Initialisation des incréments (boucles while)
i, t= 0, 0

###### ------ intialisation premier incrément temporel ------ ######
while i <= sx-1 :
    if i < sx/2 :
        w1.append(1.)   # en kg.m-3
        p.append(100)   # en kilo newton
        u.append(0.)    # en m.s-1
    else :
        w1.append(0.125)# en kg.m-3
        p.append(10.)   # en kilo newton
        u.append(0.)    # en m.s-1
    
    #w1
    w2.append(w1[i]*u[i])
    w3.append((1./g1*p[i] + 0.5 *w1[i]*u[i]**2))
    
    fw1.append(w1[i]*u[i])
    fw2.append(w2[i]*u[i]+p[i])
    fw3.append((w3[i] + p[i])*u[i])
    i += 1
    
rho [:] = w1
cg = np.sqrt(gamma*p[0]/rho[0])
cd = np.sqrt(gamma*p[-1]/rho[-1])
###### --------- Fin intialisation --------- ######

# Tracés figure initialisation : 
if trace_initial == True :
    tracage(rho, u, p, X, 'initialisation', borne)
    
###### ------ Résolution t>0 ------ ######
i,j = 1,1
#w1_m_list, w1_p_list, w2_m_list, w2_p_list, w3_m_list, w3_p_list = [], [], [], [], [], [] ##Pour debbuger

while t < st :
    while i < sx-1 :
        w1_m, w1_p = intermediaires(w1,fw1,i,r) # w1(n + 1/2, i - 1/2) et w1(n + 1/2, i + 1/2)
        w2_m, w2_p = intermediaires(w2,fw2,i,r) # w2(n + 1/2, i - 1/2) et w2(n + 1/2, i + 1/2)
        w3_m, w3_p = intermediaires(w3,fw3,i,r) # w3(n + 1/2, i - 1/2) et w3(n + 1/2, i + 1/2)
        
        fw1_m, fw1_p = w2_m, w2_p # fw1(n + 1/2, i - 1/2) et fw1(n + 1/2, i + 1/2)
        fw2_m, fw3_m = Fws(w1_m,w2_m,w3_m,gamma) # fw2(n + 1/2, i - 1/2) et fw3(n + 1/2, i - 1/2)
        fw2_p, fw3_p = Fws(w1_p,w2_p,w3_p,gamma) # fw2(n + 1/2, i + 1/2) et fw3(n + 1/2, i + 1/2)
        
#        print(fw1_m)                                       ## Pour debbuger
#        w1_m_list.append(w1_m) ; w1_p_list.insert(i,w1_p)  ## Pour debbuger
        
        w1_tp.append( w1[i] - r*( fw1_p - fw1_m ) )
        w2_tp.append( w2[i] - r*( fw2_p - fw2_m ) )
        w3_tp.append( w3[i] - r*( fw3_p - fw3_m ) )
 
        uu = np.asarray(w2) / np.asarray(w1) # Vitesse
        fw1 = np.asarray(w2) # fw1 = w2
        fw2 = uu*np.asarray(w2) *( 1 - 0.5*g1 ) + g1*np.asarray(w3) # Ok
        fw3 = (np.asarray(w3)*(1+g1) - uu*np.asarray(w2)*(g1 * 0.5))*uu # ok
                         
        i += 1  # On passe à la case d'après
    
#    print(fw1, fw2, fw3) ## Pour debbuger
    w1[1:sx-1] = w1_tp; w2[1:sx-1] = w2_tp; w3[1:sx-1] = w3_tp # On enregistre les tableaux tampons dans les wi  

    # Conditions aux limites 
    w1[0] = w1[1] ; w1[-1] = w1[sx-1]
    w2[0] = w2[1] ; w2[-1] = w2[sx-1]
    w3[0] = w3[1] ; w3[-1] = w3[sx-1]
    
    w1_tp, w2_tp, w3_tp = [], [], [] # Réinitialisation des tampons
    
    i=1
    t+=1 # Itération temporelle suivante
    
    w1 = np.asarray(w1) ; w2 = np.asarray(w2) ; w3 = np.asarray(w3)
    
    tracage(w1, w2 / w1, g1 * w3 - 0.5 * g1 * w2**2 / w1, X, 'Processing ...', borne)
    plt.pause(0.01)
    plt.clf()
######### Résultats finaux #########
w1 = np.asarray(w1) ; w2 = np.asarray(w2) ; w3 = np.asarray(w3) 
P_tf =  g1 * w3 - 0.5 * g1 * w2**2 / w1
u_tf = w2 / w1

if trace_final == True : 
    fig1 = 'Resultats_avec_tf=%.3f_Nx=%.1f_et_Nt=%.1f' %(tf,Nx,Nt)    
    tracage(w1, u_tf, P_tf, X, fig1, borne)


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



