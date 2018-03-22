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
def pd_write_csv(filename, data) :
    """
    Argument :
    ----------
    filename:   the path where creates the csv file;
    data    :   the data to write in the file
    """
    path = osp.join("~/data", filename)
    pd.DataFrame(data).to_csv(path, index=False, header= True)
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

    u = []
    u_nNext = [] 
    
    # Initialisation des champs u (boucles while)
    for x in X :
        if x >=0 and x<=1 :
            u.append(1)
        if x>1 :
            u.append(0)
    
    fu = np.asarray([0.5*u_x**2 for u_x in u])
    print ("size u it 0 = %d" %(np.size(u)))
    print ("size fu it 0 = %d" %(np.size(fu)))

    # Tracés figure initialisation : 
    plt.figure("Resolution")
    plt.plot(X, u)
    plt.title("U vs X iteration 0")
    plt.pause(0.01)

    ###### ------ Résolution t>0 ------ ######
    t = cpt = 0

    bruit = 0.001 * np.random.randn(Nx-1)

    while t < tf :
        i=1
        while i <= Nx-2 :
            u_m, u_p = intermediaires(u, fu, i, r)
            
            fu_m =  0.5*u_m**2
            fu_p =  0.5*u_p**2
                    
            u_nNext.append( u[i] - r*( fu_p - fu_m ) + bruit[i] )
            fu = np.asarray([0.5*u_x**2 for u_x in u])
                                     
            i += 1  # On passe à la case d'après
        
        print ("size u_nNext it 0 = %d" %(np.size(u_nNext)))
        
        u[1:Nx-1] = u_nNext  
        u_nNext  = []
        
        # Conditions aux limites 
        u[0] = u[-1]
        u[1] = u[0]
        u[-1]= u[-2]
        
        print np.size(u)
        
        t+=dt # Itération temporelle suivante
        
        cpt += 1
        u = np.asarray(u) 
        
        if cpt % 10 == 0 :
            plt.clf()
            plt.plot(X, u, c='k') 
            plt.title("u vs X, iteration %d" %(cpt))   
            plt.pause(1)

###----- Fin du programme ---- ###



