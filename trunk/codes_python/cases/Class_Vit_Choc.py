#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

def parser() :
    parser=argparse.ArgumentParser(description='\
    This parser will be used in several steps both in inference and ML postprocessing\n\
    Each entry is detailed in the help, and each of it has the most common default value. (run ... .py -h)\
    This on is to initialize different aspect of Burger Equation problem')
    ## VaV T_inf
    #lists
#    parser.add_argument('--T_inf_lst', '-T_inf_lst', nargs='+', action='store', type=int, default=[5*i for i in range (1,11)],dest='T_inf_lst', 
#                        help='List of different T_inf. Default : all\n' )
    # Caractéristiques de la simulation voulue          
    parser.add_argument('--Nx', '-Nx', action='store', type=int, default=202, dest='Nx', 
                        help='Define the number of discretization points : default %(default)d \n' )
    parser.add_argument('--N_temp', '-Nt', action='store', type=int, default=202, dest='Nt', 
                        help='Define the number of time steps; default %(default)d \n' )
    parser.add_argument('--domain_length', '-L', action='store', type=int, default=float(3), dest='L',
                        help='Define the length of the domain; default %(default)f \n' )
    parser.add_argument('--CFL', '-CFL', action='store', type=float, default=0.45, dest='CFL', 
                        help='Define this simulations\'s CFL (under 0.5); default %(default)d\n' )
    parser.add_argument('--diffusion_rate', '-nu', action='store', type=float, default=2.5e-5, dest='nu', 
                        help='Define the convection coefficient h \n' )
    
    # Pour l'algorithme
#    parser.add_argument('--delta_t', '-dt', action='store', type=float, default=1e-4, dest='dt', 
#                        help='Define the time step disctretization. Default to %(default).5f \n' )
    parser.add_argument('--Iteration_max', '-itmax', action='store', type=int, default=500, dest='itmax', 
                        help='Define temporal maximum iteration (-itmax) in both solving problems : default %(default)d \n' )
    parser.add_argument('--compteur_max_adjoint', '-cptmax', action='store', type=int, default=100, dest='cpt_max_adj', 
                        help='Define compteur_max (-cptmax) for adjoint method: default %(default)d \n' )
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=100, dest='num_real', 
                        help='Define how many samples of epsilon(T) to draw for exact model. Default to %(default)d\n' )
    parser.add_argument('--g_sup_max', '-g_sup', action='store', type=float, default=0.001, dest='g_sup_max', 
                        help='Define the criteria on grad_J to stop the optimization. Default to %(default).5f \n' )
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', default=1 ,dest='beta_prior',\
                        help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    
    # Strings
    parser.add_argument('--init_u', '-init_u', action='store', type=str, default='sin', dest='type_init', 
                        help='Choose initial condition on u. Defaut sin\n')
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data/burger_dataset/', dest='datapath', 
                        help='Define the directory where the data will be stored and read. Default to %(default)s \n')
    parser.add_argument('--covariance_model', '-cov_mod', action='store', type=str, default='full', dest='cov_mod', 
                        help='Define the covariance model. Default to %(default)s \n')
    parser.add_argument('--logbook_path', '-p', action='store', type=str, default='./logbooks/', dest='logbook_path', 
                        help='Define the logbook\'s path. Default to %(default)s \n')
   
    parser.add_argument('--type_J', '-typeJ', action='store', type=str, default="u", dest='typeJ',\
                        help='Define the type of term you want to simulate')
#    # Booléen
#    parser.add_argument('--T_inf_type', '-T_cst', action='store', type=bool, default=False, dest='T_cst', 
#                        help='Define whether the T_inf is constant or not. Default to %(default)s \n')
    
    return parser.parse_args()

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

class Vitesse_Choc() :
##---------------------------------------------------------------
    def __init__ (self, parser):
        """
        This object has been made to solve optimization problem.
        """
#        np.random.seed(1000) ; #plt.ion()
#        if parser.cov_mod not in ['full', 'diag'] :
#            raise AttributeError("\x1b[7;1;255mcov_mod must be either diag or full\x1b[0m")
        Nx,  Nt   =   parser.Nx, parser.Nt,    
        CFL, nu   =   parser.CFL,     parser.nu
        
        L = parser.L
        dx = L/(Nx-1)
        
        dt = {"dt_v" : CFL / nu * dx**2,
              "dt_l" : CFL*dx}
        
        if dt["dt_v"] < dt["dt_l"] :
            dt = dt["dt_v"]
            print ("dt visqueux")
        else :
            dt = dt["dt_l"]
            print ("dt lineaire")
                
        fac = nu*dt/dx**2
        tf = Nt * dt
        
        r = dt / dx
        
        datapath    =   osp.abspath(parser.datapath)
        num_real    =   parser.num_real
        cpt_max_adj =   parser.cpt_max_adj
        cov_mod     =   parser.cov_mod
        g_sup_max   =   parser.g_sup_max
        itmax       =   parser.itmax
        typeJ       =   parser.typeJ
        
        self.beta_prior = np.asarray([parser.beta_prior for i in range(Nx)]) 
        
        ## Matrices des coefficients pour la résolution
        ## Attention ces matrices ne prennent pas les points où sont définies les conditions initiales
        ## Ces cas doivent faire l'objet de méthodes particulières avec la redéfinition des fonctions A1 et A2 
        
        #####
        INF1 = np.diag(np.transpose([-fac/2 for i in range(Nx-3)]), -1)
        SUP1 = np.diag(np.transpose([-fac/2 for i in range(Nx-3)]), 1) 
        A_diag1 = np.diag(np.transpose([(1 + fac) for i in range(Nx-2)])) 
        
        In1 = A_diag1 + INF1 + SUP1
        
        A1 = np.zeros((Nx,Nx))
        
        A1[0,0] = A1[-1,-1] = 1
        
        A1[1:Nx-1, 1:Nx-1] = In1
        #####
        self.A1 = A1
        
        bruits = [0.0009 * np.random.randn(Nx) for time in range(num_real)]
        self.bruits = bruits
        
        self.line_x = np.linspace(0, L, Nx)

        self.cpt_max_adj = cpt_max_adj
        self.g_sup_max = g_sup_max  
        self.itmax = itmax        

        self.num_real = num_real
        self.cov_mod = cov_mod
        
        r = dt/dx
        
        self.L ,    self.tf     =   L , tf
        self.nu,    self.CFL    =   nu, CFL
        self.dx,    self.dt     =   dx, dt        
        self.Nx,    self.Nt     =   Nx, Nt
        self.fac,   self.r      =   fac, r
        
        self.nu_str = str(self.nu).replace(".","_")
        self.CFL_str = str(self.CFL).replace(".","_")
        self.type_init = parser.type_init
        
        
        bool_method = dict()
        bool_written= dict()
        
        if osp.exists(datapath) == False :
            os.mkdir(datapath)
        
        bmatrice_path = osp.join(datapath, "burger_matrices")
        case_path = osp.join(bmatrice_path, "cas_%s" % typeJ)
        
        self.cov_path = osp.join(case_path, "post_cov")
        self.beta_path = osp.join(case_path, "betas")
        self.chol_path = osp.join(case_path, "cholesky")
        self.inferred_U = osp.join(case_path, "U")
        
        if osp.exists(osp.join(case_path)) == False :
            os.makedirs(case_path)
        
        if osp.exists(self.inferred_U) == False :
            os.mkdir(self.cov_path)
            os.mkdir(self.chol_path)
            os.mkdir(self.beta_path)
            os.mkdir(self.inferred_U)

        self.datapath   =   datapath
        
        self.beta_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(self.beta_path,\
            "beta_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
        self.u_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(self.inferred_U,\
            "U_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
        self.chol_name = lambda nx, nt, nu, type_i, CFL, it : osp.join(self.chol_path,\
            "chol_Nx:{}_Nt:{}_nu:{}_".format(nx, nt, nu) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))

        self.stats_done = False    

        self.typeJ = typeJ        
                        
        self.parser = parser
##---------------------------------------------------
    def set_beta_prior(self, new_beta) :
        """
        Descr :
        ----------
        Method designed to change the beta_prior array without running back the whole program.
        """
        self.beta_prior = np.asarray([new_beta for i in range(self.Nx)])
        print("Beta prior is now \n{}".format(self.beta_prior))
##---------------------------------------------------
    def set_cov_mod(self, new_cov_mod) :
        """
        Descr :
        ----------
        Method designed to change the covariance form  without running back the whole program.
        """
        self.cov_mod = new_cov_mod
        self.bool_method["stat"] = False
        print("cov_mod is now \n{}".format(self.cov_mod))
##---------------------------------------------------        
    def set_cpt_max_adj(self, new_compteur) :
        """
        Descr :
        ----------
        Method designed to change the adjoint maximal increment value without running back the whole program.
        """
        self.cpt_max_adj = new_compteur
        print("cpt_max_adj is now {}".format(self.cpt_max_adj))
##---------------------------------------------------
    def set_g_sup_max(self, new_criteria) :
        """
        Descr :
        ----------
        Method designed to change the adjoint criteria for exiting adjoint optimization without running back the whole program.
        """
        self.g_sup_max = new_criteria
        print("g_sup_max is now {}".format(self.g_sup_max))   
###---------------------------------------------------
    def init_u(self, amp=1, phase=0, cpx=1): 
        u = []
        if self.type_init == "choc":
            for i in range(len(self.line_x)) :
                if self.line_x[i] >=1 and self.line_x[i] <=1.4 :
                    u.append(1.5)
                else :
                    u.append(0)
            u = np.array(u) + 0.4
            
        if self.type_init == "sin" :
            u = amp*np.sin(2*np.pi/self.L*self.line_x + phase) 
        
        if self.type_init == "sin_decale" :
            u = amp*np.sin(2*np.pi/self.L*self.line_x + phase) + 0.5
        
        if self.type_init == "complex_sin" :
            u = np.zeros((len(self.line_x)))
            harmonics = [np.sin(2*np.pi/self.L*self.line_x + k*phase) for k in range(1, cpx+1)]
            
            for h in harmonics :
                for i in range(len(self.line_x)) :
                    u[i] += h[i]
        return u
##---------------------------------------------------
    def u_beta(self, beta, u_n, verbose=False) :
#        print beta.size, beta.shape
        if self.typeJ == "grad" : 
            INF2 = np.diag(np.transpose([self.fac/2 for i in range(self.Nx-3)]), -1) 
            SUP2 = np.diag(np.transpose([-self.r*beta[i] + self.fac/2 for i in range(self.Nx-3)]), 1)
            A_diag2 = np.diag(np.transpose([(self.r*beta[i] + 1 - self.fac) for i in range(self.Nx-2)]))
        
        if self.typeJ == "u" : 
            INF2 = np.diag(np.transpose([self.fac/2 for i in range(self.Nx-3)]), -1) 
            SUP2 = np.diag(np.transpose([self.fac/2 for i in range(self.Nx-3)]), 1)
            A_diag2 = np.diag(np.transpose([(- self.dt*beta[i] + 1 - self.fac) for i in range(self.Nx-2)]))
        
        if self.typeJ == "1" :
            INF2 = np.diag(np.transpose([self.fac/2 for i in range(self.Nx-3)]), -1) 
            SUP2 = np.diag(np.transpose([ self.fac/2 for i in range(self.Nx-3)]), 1)
            A_diag2 = np.diag(np.transpose([1 - self.fac for i in range(self.Nx-2)]))
        
        In2 = A_diag2 + INF2 + SUP2
        
        A2 = np.zeros((self.Nx,self.Nx))
        A2[1:self.Nx-1, 1:self.Nx-1] = In2

#        u_n_2 = np.array([0.5*ux**2 for ux in u_n])
        
        u_n_tmp = np.dot(A2, u_n)
        
        u_nNext = np.dot(np.linalg.inv(self.A1), u_n_tmp)

        u_nNext[-1] = u_nNext[1]
        u_nNext[0] = u_nNext[-2]
        
        return u_nNext
##---------------------------------------------------         
    def obs_res(self, write=False, plot=False)  : 
        ## Déf des données du probleme 
        for j, bruit in enumerate(self.bruits) :
            # Initialisation des champs u (boucles while)
            u, u_nNext = [], []
            u = self.init_u() 
#            u = np.sin(2*np.pi/self.L*self.line_x) + bruit
            
            r = self.dt/self.dx
            # Tracés figure initialisation : 
            if plot == True :
                plt.figure("Resolution")
                plt.plot(self.line_x, u)
                plt.title("U vs X iteration 0 bruit %d" %(j))
                plt.ylim((-2.5, 2.5))
                plt.pause(0.01)
            
            # pd_write_csv --->> np.save
            if write == True : 
                filename = osp.join(self.datapath, "u_it0_%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy"%(j ,self.Nt ,self.Nx, self.CFL_str, self.nu_str, self.type_init))
                np.save(filename, u)
                
            t = it = 0
            
            while it <= self.itmax+1 :
                filename = osp.join(self.datapath, "u_it%d_%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy"%(it+1, j, self.Nt, self.Nx, self.CFL_str, self.nu_str, self.type_init))
                if osp.exists(filename) == True :
                    it += 1
                    continue
                    
                fu = np.asarray([0.5*u_x**2 for u_x in u])
                
                der_sec = [self.fac*(u[k+1] - 2*u[k] + u[k-1]) for k in range(1, len(u)-1)]
                der_sec.insert(0, self.fac*(u[1] - 2*u[0] + u[-1]))
                der_sec.insert(len(der_sec), self.fac*(u[0] - 2*u[-1] + u[-2]))

                for i in range(1,self.Nx-1) : # Pour prendre en compte le point Nx-2
                    u_m, u_p = intermediaires(u, fu, i, r)
                    fu_m =  0.5*u_m**2
                    fu_p =  0.5*u_p**2

                    u_nNext.append( u[i] - r*( fu_p - fu_m ) + bruit[i] + der_sec[i] )
                                                
                # Conditions aux limites 
                u[1:self.Nx-1] = u_nNext  
                u_nNext  = []
                
                u[0] = u[-2]
                u[-1]= u[1]
    
                u = np.asarray(u) 
            
                if write == True : 
                    np.save(filename, u)
                
                it += 1
                t += self.dt # Itération temporelle suivante
                    
                if plot == True :
                    if it % 10 == 0 :
                        plt.clf()
                        plt.plot(self.line_x[0:self.Nx-1], u[0:self.Nx-1], c='k') 
                        plt.grid()
                        plt.title("u vs X, iteration %d bruit %d" %(it, j)) 
                        plt.xticks(np.arange(0, self.L-self.dx, 0.25))
#                        plt.yticks(np.arange(-2.5, 2.5, 0.5))
                        plt.ylim(-2.5,2.5)
                        plt.pause(0.1)  
                
##---------------------------------------------------
    def get_obs_statistics(self, write = True):
        U_moy_obs = dict()
        full_cov_obs_dict = dict()
        diag_cov_obs_dict = dict()
        init = self.type_init
        for it in range(self.itmax+1) :
            u_sum = np.zeros((self.Nx))

            # Calcul de la moyenne pour l'itération en cours
            for n in range(self.num_real) :
                file_to_get = osp.join(self.datapath, "u_it%d_%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy"%(it, n, self.Nt, self.Nx, self.CFL_str, self.nu_str, init))
                u_t_n = np.load(file_to_get)
                for i in range(len(u_t_n)) : u_sum[i] += u_t_n[i] / float(self.num_real)
                
            U_moy_obs["u_moy_it%d" %(it)] = u_sum
            full_cov = np.zeros((self.Nx, self.Nx))        
            
            # Calcul de la covariance associée à l'itération
            full_cov_filename = osp.join(self.cov_path, "full_cov_obs_it%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy"%(it, self.Nt, self.Nx, self.CFL_str, self.nu_str, init)) 
            diag_cov_filename = osp.join(self.cov_path, "diag_cov_obs_it%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy"%(it, self.Nt, self.Nx, self.CFL_str, self.nu_str, init)) 
            
            if osp.exists(full_cov_filename) == True and osp.exists(diag_cov_filename) :
                full_cov_obs_dict["full_cov_obs_it%d"%(it)] = np.load(full_cov_filename) 
                diag_cov_obs_dict["diag_cov_obs_it%d"%(it)] = np.load(diag_cov_filename)
#                print ("Lecture %s" %(cov_filename))
                continue
            
            for n in range(self.num_real) :
                file_to_get = osp.join(self.datapath, "u_it%d_%d_Nt%d_Nx%d_CFL%s_nu%s_%s.npy" %(it, n, self.Nt, self.Nx, self.CFL_str, self.nu_str, init))
                u_t_n = np.load(file_to_get)
                
                for ii in range(self.Nx)  :
                    for jj in range(self.Nx) : 
                        full_cov[ii,jj] += (u_t_n[ii] - u_sum[ii]) * (u_t_n[jj] - u_sum[jj]) / float(self.num_real)
            
            full_cov_obs_dict["full_cov_obs_it%d"%(it)] = full_cov 
            diag_cov_obs_dict["diag_cov_obs_it%d"%(it)] = np.diag(np.diag(full_cov))
            
            if write == True :
#                print ("Ecriture %s" %(cov_filename))
                if osp.exists(diag_cov_filename) == False :
                    np.save(diag_cov_filename, np.diag(np.diag(full_cov)))
                
                if osp.exists(full_cov_filename) == False :
                    np.save(full_cov_filename, full_cov)
                    
        self.U_moy_obs = U_moy_obs
        
        self.full_cov_obs_dict = full_cov_obs_dict
        self.diag_cov_obs_dict = diag_cov_obs_dict
        
        self.stats_done = True
##---------------------------------------------------
##---------------------------------------------------
##---------------------------------------------------
    def minimization(self, maxiter, solver="BFGS", step=5):
        fig, axes= plt.subplots(1, 2, figsize = (8,8))
        evol = 0
        if self.stats_done == False :
            self.get_obs_statistics(True)
            print("Get_Obs_Statistics lunched")
        #---------------------------------------------    
        #---------------------------------------------
        def DR_DU (beta) :
            if self.typeJ == "grad":
                INF1 = 0.5*np.diag([-self.fac/2. for i in range(self.Nx-1)], -1)
                SUP1 = 0.5*np.diag([beta[i]*self.r - self.fac/2. for i in range(self.Nx - 1)], +1)
                A1   = np.diag([self.fac - (1. + beta[i]*self.r) for i in range(self.Nx)])
            
            if self.typeJ == "u" :   
                INF1 = 0.5*np.diag([-self.fac/2. for i in range(self.Nx - 1)], -1)
                SUP1 = 0.5*np.diag([- self.fac/2. for i in range(self.Nx - 1)], +1)
                A1   = np.diag([beta[i]*self.dt - 1. + self.fac for i in range(self.Nx)])
            
            if self.typeJ == "1" :
                INF1 = 0.5*np.diag([-self.fac/2. for i in range(self.Nx - 1)], -1)
                SUP1 = 0.5*np.diag([- self.fac/2. for i in range(self.Nx - 1)], +1)
                A1   = np.diag([-1. + self.fac for i in range(self.Nx)])
        
            drdu = A1 + INF1 + SUP1
        
            return np.linalg.inv(drdu)
        #---------------------------------------------
        #---------------------------------------------
        def dR_dbeta(beta, u_n):
            DR_DBETA = np.zeros((self.Nx, self.Nx))
            
            if self.typeJ == "grad":
                uu = [(u_n[i+1] - u_n[i]) * self.r for i in range(len(u_n)-1)]
                uu.insert(len(uu), (u_n[1] - u_n[len(uu)] )*self.r)
                DR_DBETA = np.diag(uu)
            
            if self.typeJ == "u" :
                DR_DBETA = np.diag(u_n)
                    
            if self.typeJ == "1":
                DR_DBETA = np.diag([-self.dt for i in range(len(u_n))])
            
            return DR_DBETA
        #---------------------------------------------
        #---------------------------------------------
        def d_unp1_d_un(beta, u) :
            if self.typeJ == "grad":
                dij_LW = [ 1-self.r/8.*\
                            (\
                                2*(u[i+1] - u[i-1])\
                                -self.r*(u[i+1]**2 - 2*u[i]*(u[i+1] + u[i-1]) +u[i-1]**2)\
                                -self.r**2*u[i]*(u[i+1]**2 - u[i-1]**2)
                            )\
                            for i in range(1,len(u)-1)\
                         ]
#                dij_LW.insert(0, dij_LW[-1])
#                dij_LW.insert(len(dij_LW), dij_LW[1])
                dij_LW = np.asarray(dij_LW)

                dijM1_LW = [self.r/8. *\
                            (\
                                2*(u[i-1] + u[i])\
                                + self.r*(u[i-1]*(2*u[i] + 3*u[i-1])-u[i]**2)\
                                - self.r**2*u[i-1]*(u[i]**2 - u[i-1]**2)\
                            )\
                            for i in range(2,len(u)-1)\
                          ]
                i=0
                dijP1_LW = [-self.r/8.*\
                            (\
                                2*(u[i]+ u[i+1])\
                                - self.r*(u[i+1]*(2*u[i] + 3*u[i+1]) -u[i]**2)\
                                + self.r**2*u[i+1]*(u[i+1]**2 - u[i]**2)\
                            )\
                            for i in range(1,len(u)-2)\
                          ]

                dij_CN = np.asarray([self.fac-(1+beta[i]*self.r) for i in range(1,len(u)-1)])

                dijM1_CN = np.asarray([-self.fac*0.5 for i in range(2, len(u)-1)])
                dijP1_CN = np.asarray([beta[i] * self.r - self.fac*0.5 for i in range(1,len(u)-2)])

                dij = dij_LW + dij_CN
                diM1j = dijM1_LW + dijM1_CN
                diP1j = dijP1_LW + dijP1_CN
                
                IndJ_Du = np.diag(dij) + np.diag(diM1j, -1) + np.diag(diP1j, 1)
                Dj_Du = np.zeros((self.Nx, self.Nx))
                
                Dj_Du[0,0] = dij[-1] #ie N-2 sur le total
                Dj_Du[-1,-1] = dij[0] #ie le 1 sur le total
                
                return Dj_Du
        #---------------------------------------------
        #---------------------------------------------
        beta_n = self.beta_prior
        u_n = self.U_moy_obs["u_moy_it0"]
        
#        alpha = 1.e-4 # facteur de régularisation
        self.opti_obj = dict
        self.beta_n_dict = dict()
        self.U_beta_n_dict = dict()
        self.optimization_time = dict()
        
        t = 0
        reg_fac = 1e-2
        Id = np.eye(self.Nx)
        
        for it in range(self.itmax) :
            if it >0 :
                beta_n = beta_n_opti
                u_n = u_n_beta
                
            t1 = time.time()
            
            print ("it = ".format(it))
            print ("it + 1 = ".format(it+1))
            
            u_obs_nt = self.U_moy_obs["u_moy_it%d" %(it+1)]
            cov_obs_nt = self.full_cov_obs_dict["full_cov_obs_it%d"%(it+1)] # Pour avoir la bonne taille de matrice
            
            if it == 0 :
                u_n = self.u_beta(beta_n[1:self.Nx-1], u_obs_nt)
            
            print ("diag")

            cov_obs_nt = self.diag_cov_obs_dict["diag_cov_obs_it%d"%(it+1)]
            cov_obs_nt_inv = np.linalg.inv(cov_obs_nt)
            
            print (cov_obs_nt_inv.shape)
            #Pour avoir la bonne taille de matrice
            Uu = lambda beta : u_obs_nt - self.u_beta(beta[1:self.Nx-1], u_n) 
            
            J = lambda beta : 0.5 * (np.dot( np.dot(Uu(beta).T, cov_obs_nt_inv), Uu(beta)) +\
                                             reg_fac*np.dot( np.transpose(beta - beta_n).dot(Id), (beta - beta_n) ) \
                                    )
                                    
            # On utilise les différence finies pour l'instant
            DJ = nd.Gradient(J)
                                   
            
            print ("Opti Minimization it = %d" %(it))
            
            # Pour ne pas oublier :            
            # On cherche beta faisant correspondre les deux solutions au temps it + 1. Le beta final est ensuite utilisé pour calculer u_beta au temps it 
            for i in range(len(beta_n)) : # Un peu de bruit
                beta_n[i] *= np.random.random()    

            # Minimization 
            optimi_obj_n = op.minimize(J, self.beta_prior, jac=DJ, method=solver, options={"maxiter" : maxiter})
            
            print("\x1b[1;37;44mDifference de beta it {} = {}\x1b[0m".format(it, np.linalg.norm(beta_n - optimi_obj_n.x, np.inf)))
            beta_n_opti = optimi_obj_n.x
            
            #On ne prend pas les béta dans les ghost cell
            u_n_beta = self.u_beta(beta_n_opti[1:self.Nx-1], u_n)

            t2 = time.time()
            print (optimi_obj_n)
            print ("it {}, optimization time : {:.3f}".format(it, abs(t2-t1)))
            
            self.optimization_time["it%d" %(it)] = abs(t2-t1)
            
            # On enregistre pour garder une trace après calculs
            self.beta_n_dict["beta_it%d" %(it)]  = beta_n_opti
            self.U_beta_n_dict["u_beta_it%d" %(it)] = u_n_beta
            
            # On enregistre beta_n, u_n_beta et cholesky
                # Enregistrement vecteur entier
            np.save(self.beta_name(self.Nx, self.Nt, self.nu, self.type_init, self.CFL, it), beta_n_opti) 
            np.save(self.u_name(self.Nx, self.Nt, self.nu, self.type_init, self.CFL, it), u_n_beta)
            
                # Calcule de Cholesky et enregistrement
            hess_beta = optimi_obj_n.hess_inv
            cholesky_beta = np.linalg.cholesky(hess_beta)
            np.save(self.chol_name(self.Nx, self.Nt, self.nu, self.type_init, self.CFL, it), cholesky_beta)

            # On calcule l'amplitude en utilisant cholesky pour savoir si les calculs sont convergés ou pas             
            sigma = dict()
            mins, maxs = [], []
            
            fields_v = dict()
            
            for j in range(100) :
                fields_v["%03d" %(j)] = []
            
            cpt = 0
            # Tirage avec s un vecteur aléatoire tirée d'une distribution N(0,1)
            while cpt <100 :
                s = np.random.randn(self.Nx)
                beta_i = beta_n_opti + np.dot(cholesky_beta, s)
                cpt += 1
            
            # On enregistre les valeurs des tirages pour chaque i pour trouver les extrema
                for j in range(len(beta_i)): 
                    fields_v["%03d" %(j)].append(beta_i[j])
                
            for k in range(len(beta_i)) :
                mins.append(min(fields_v["%03d" %(k)]))
                maxs.append(max(fields_v["%03d" %(k)]))
            
#            print (np.shape(maxs), np.shape(mins))
            
            if it % step == 0 :
                if evol == 2 :
                    evol = 0
                    for i in [0,1] : 
                        axes[i].clear()

                if evol == 0 :
                    axes[0].plot(self.line_x[:-1], beta_n_opti[:-1], label="iteration %d" %(it), c= "r")
                    axes[0].fill_between(self.line_x[:-1], mins[:-1], maxs[:-1], facecolor= "0.2", alpha=0.2, interpolate=True, color="red")
                    
                    axes[1].plot(self.line_x[:-1], u_obs_nt[:-1], label="LW it = %d" %(it), c='k')                 
                    axes[1].plot(self.line_x[:-1], self.U_beta_n_dict["u_beta_it%d" %(it)][:-1], label='Opti it %d'%(it),\
                        marker='o', fillstyle='none', linestyle='none', c='r')
                if evol == 1 :
                    axes[0].plot(self.line_x[:-1], beta_n_opti[:-1], label="iteration %d" %(it), c = "b", marker="o")
                    axes[0].fill_between(self.line_x[:-1], mins[:-1], maxs[:-1], facecolor= "0.2", alpha=0.2, interpolate=True, color="b")
                    
                    axes[1].plot(self.line_x[:-1], u_obs_nt[:-1], label="LW it = %d" %(it), c='grey', marker="+")
                    axes[1].plot(self.line_x[:-1], self.U_beta_n_dict["u_beta_it%d" %(it)][:-1], label='Opti it %d'%(it),\
                        marker='o', fillstyle='none', linestyle='none', c='b')
                
                axes[0].legend(loc="best")

                axes[1].set_ylim((-2.0, 2.0))
                axes[1].legend(loc = "best")
                    
                plt.pause(0.01)
                evol += 1
            ## End of the while loop ## 
        
        # Plot the last iteration
        if evol == 2 :
            evol = 0
            for i in [0,1] : 
                axes[i].clear()
            
        if evol == 0 :
            axes[0].plot(self.line_x[:-1], beta_n_opti[:-1], label="iteration %d" %(it), c= "r")
            axes[0].fill_between(self.line_x[:-1], mins[:-1], maxs[:-1], facecolor= "0.2", alpha=0.2, interpolate=True, color="red")
            
            axes[1].plot(self.line_x[:-1], u_obs_nt[:-1], label="LW it = %d" %(it), c='k')                 
            axes[1].plot(self.line_x[:-1], self.U_beta_n_dict["u_beta_it%d" %(it)][:-1], label='Opti it %d'%(it),\
                marker='o', fillstyle='none', linestyle='none', c='r')
        if evol == 1 :
            axes[0].plot(self.line_x[:-1], beta_n_opti[:-1], label="iteration %d" %(it), c = "b", marker="o")
            axes[0].fill_between(self.line_x[:-1], mins[:-1], maxs[:-1], facecolor= "0.2", alpha=0.2, interpolate=True, color="b")
            
            axes[1].plot(self.line_x[:-1], u_obs_nt[:-1], label="LW it = %d" %(it), c='grey', marker="+")
            axes[1].plot(self.line_x[:-1], self.U_beta_n_dict["u_beta_it%d" %(it)][:-1], label='Opti it %d'%(it),\
                marker='o', fillstyle='none', linestyle='none', c='b')
        
        axes[0].legend(loc="best")

        axes[1].set_ylim((-2.0, 2.0))
        axes[1].legend(loc = "best")
            
        plt.pause(0.01)
###---------------------------------------------------##   
##----------------------------------------------------##
if __name__ == '__main__' :
#    run Class_Vit_Choc.py -nu 2.5e-2 -itmax 200 -CFL 0.4 -num_real 5 -Nx 52 -Nt 52
#    run Class_Vit_Choc.py -nu 2.5e-2 -itmax 200 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10
    parser = parser()
#    plt.close("all")
    
    cb = Vitesse_Choc(parser)   
#    cb.obs_res(True, True)
#    cb.obs_res(write=False, plot=True)
#    cb.get_obs_statistics(write=True)
    
#    cb.minimization()


## Pour lancer ce code : 
# run Class_Vit_Choc -h #Pour visualiser les parametres a tuner ou leur valeur par defaut 

# On pourra essayer :
# run Class_Vit_Choc.py -nu 2.5e-2 -itmax 40 -CFL 0.4 -num_real 5 -Nx 32 -Nt 32 -beta_prior 10 -typeJ "u"
# cb.obs_res(True, True)
# cb.minimization(maxiter=50, solver="BFGS", step=5) # Ici aussi on pourra modifier ces parametres

