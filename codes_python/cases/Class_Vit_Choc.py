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
    parser.add_argument('--N_discr', '-N', action='store', type=int, default=202, dest='N_discr', 
                        help='Define the number of discretization points : default %(default)d \n' )
    parser.add_argument('--N_temp', '-N', action='store', type=int, default=202, dest='Nt', 
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
    parser.add_argument('--compteur_max_adjoint', '-cptmax', action='store', type=int, default=100, dest='cpt_max_adj', 
                        help='Define compteur_max (-cptmax) for adjoint method: default %(default)d \n' )
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=100, dest='num_real', 
                        help='Define how many samples of epsilon(T) to draw for exact model. Default to %(default)d\n' )
    parser.add_argument('--g_sup_max', '-g_sup', action='store', type=float, default=0.001, dest='g_sup_max', 
                        help='Define the criteria on grad_J to stop the optimization. Default to %(default).5f \n' )
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', default=1 ,dest='beta_prior',\
                        help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    
    # Strings
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data', dest='datapath', 
                        help='Define the directory where the data will be stored and read. Default to %(default)s \n')
    parser.add_argument('--covariance_model', '-cov_mod', action='store', type=str, default='full', dest='cov_mod', 
                        help='Define the covariance model. Default to %(default)s \n')
    parser.add_argument('--logbook_path', '-p', action='store', type=str, default='./logbooks/', dest='logbook_path', 
                        help='Define the logbook\'s path. Default to %(default)s \n')
    
#    # Booléen
#    parser.add_argument('--T_inf_type', '-T_cst', action='store', type=bool, default=False, dest='T_cst', 
#                        help='Define whether the T_inf is constant or not. Default to %(default)s \n')
    
    return parser.parse_args()

class Vitesse_Choc() :
##---------------------------------------------------------------
    def __init__ (self, parser):
        """
        This object has been made to solve optimization problem.
        """
#        np.random.seed(1000) ; #plt.ion()
        
        if parser.cov_mod not in ['full', 'diag'] :
            raise AttributeError("\x1b[7;1;255mcov_mod must be either diag or full\x1b[0m")
        
        Nx,  Nt   =   parser.N_discr, parser.Nt,    
        CFL, nu   =   parser.CFL,     parser.nu
        
        if type(L) is int : L = float(L)
        
        dx = L/(Nx-1)
        dt = CFL / nu * dx**2
        
        tf = Nt * dt
        
        datapath            =   osp.abspath(parser.datapath)
        num_real,   tol     =   parser.num_real,parser.tol
        cpt_max_adj         =   parser.cpt_max_adj
        cov_mod,    g_sup_max  =   parser.cov_mod, parser.g_sup_max
        
        self.beta_prior = np.asarray([parser.beta_prior for i in range(parser.N_discr-2)]) 
        
        ## Matrices des coefficients pour la résolution
        ## Attention ces matrices ne prennent pas les points où sont définies les conditions initiales
        ## Ces cas doivent faire l'objet de méthodes particulières avec la redéfinitions des fonctions A1 et A2 
        
        #####
        
        INF1 = np.diag(np.transpose([-dt/dx**2*nu/2 for i in range(N_discr-3)]), -1)
        SUP1 = np.diag(np.transpose([-dt/dx**2*nu/2 for i in range(N_discr-3)]), 1) 
        A_diag1 = np.diag(np.transpose([(1 + dt/dx**2*nu) for i in range(N_discr-2)])) 

        INF2 = np.diag(np.transpose([dt/dx**2*nu/2 for i in range(N_discr-3)]), -1) 
        SUP2 = np.diag(np.transpose([dt/dx**2*nu/2 for i in range(N_discr-3)]), 1)
        A_diag2 = np.diag(np.transpose([(1 - dt/dx**2*nu) for i in range(N_discr-2)])) 

        self.A1 = A_diag1 + INF1 + SUP1
        self.A2 = A_diag2 + INF2 + SUP2
        
        #####
        
        bruits = [0.0005 * np.random.randn(Nx) for time in range(num_real)]
        self.bruits = bruits
        
        self.store_rho = []
        
        self.line_x = np.arange(0,L+dx, dx)
        self.cpt_max_adj = cpt_max_adj
        self.g_sup_max = g_sup_max  
        self.num_real = num_real
        self.eps_0 = 5.*10**(-4)
        self.cov_mod = cov_mod
        self.tol = tol

        self.nu, self.CFL = nu, CFL
        self.dx, self.dt = dx, dt        
        self.Nx, self.Nt = Nx, Nt
        
        bool_method = dict()
        bool_written= dict()
        
#        runs = set()
#        runs.add("stat")
#        
#        for t in self.T_inf_lst :
#            sT_inf = "T_inf_%s" %(str(t))
#            runs.add("opti_scipy_%s" %(sT_inf))
#            runs.add("adj_bfgs_%s" %(sT_inf))
#            
#        for r in runs :
#            bool_method[r] = False
#            bool_written[r]= False
#            
#        self.bool_method = bool_method
#        self.bool_written = bool_written
#        
#        #Création des différents fichiers
#        self.date = time.strftime("%m_%d_%Hh%M", time.localtime())
#        
#        # On test si les dossiers existent, sinon on les créé
#        # Pour écrire la valeur de cov_post. Utile pour le ML 
#        if osp.exists(osp.abspath("./data/post_cov")) == False :
#            os.mkdir(osp.abspath("./data/post_cov"))
#        
#        if osp.exists(osp.abspath("./data/matrices")) == False :
#            os.mkdir(osp.abspath("./data/matrices"))
#        self.path_fields = osp.abspath("./data/matrices")        
#                
#        if osp.exists(datapath) == False :
#            os.mkdir(datapath)
#        
#        if osp.exists(parser.logbook_path) == False :
#            os.mkdir(parser.logbook_path)
#        
#        if osp.exists("./err_check") == False :
#            os.mkdir("./err_check")
#            
#        self.err_title = osp.join("err_check", "%s_err_check.csv" %(self.date))
#        self.logout_title = osp.join(parser.logbook_path, "%s_logbook.csv" %(self.date))
#        
#        # On intialise les fichiers en rappelant la teneur de la simulation
#        for f in {open(self.logout_title, "w"), open(self.err_title, "w")} :
#            f.write("\n#######################################################\n")
#            f.write("## Logbook: simulation launched %s ## \n" %(time.strftime("%Y_%m_%d_%Hh%Mm%Ss", time.localtime())))
#            f.write("#######################################################\n")
#            f.write("Simulation\'s features :\n{}\n".format(parser))
#            f.close()

        self.datapath   =   datapath
        self.parser     =   parser
##---------------------------------------------------
    def set_beta_prior(self, new_beta) :
        """
        Descr :
        ----------
        Method designed to change the beta_prior array without running back the whole program.
        """
        self.beta_prior = np.asarray([new_beta for i in range(self.N_discr-2)])
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
##---------------------------------------------------
    def pd_read_csv(self, filename) :
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
##---------------------------------------------------
    def pd_write_csv(self, filename, data) :
        """
        Argument :
        ----------
        filename:   the path where creates the csv file;
        data    :   the data to write in the file
        """
        path = osp.join(self.datapath, filename)
        pd.DataFrame(data).to_csv(path, index=False, header= True)
##---------------------------------------------------   
    def u_beta(self, beta, u_n, iteration ,verbose=False) :
        u_n_tmp = np.dot(self.A2, u_n)
        
        for i in range(self.N_discr-2) :
            B_n[i] = u_n_tmp[i] + self.dt*(beta[i])

        u_nNext = np.dot(np.linalg.inv(self.A1), B_n)
            
        return u_nNext              
##---------------------------------------------------         
    def obs_res(self, write=False, plot=False)  : 
        ## Déf des données du probleme 
        for j, bruit in enumerate(self.bruits) :
            # Initialisation des champs u (boucles while)
            u, u_nNext = [], []
            plt.close()
            for i in range(len(self.line_x)) :
                if self.line_x[i] >=0 and self.line_x[i] <=1 :
                    u.append(1 + bruit[i])
                if self.line_x[i] > 1 :
                    u.append(0 + bruit[i])
                i+=1
                
            fu = np.asarray([0.5*u_x**2 for u_x in u])
            
            # Tracés figure initialisation : 
            if plot == True :
                plt.figure("Resolution")
                plt.plot(self.line_x, u)
                plt.title("U vs X iteration 0 bruit %d" %(j))
                plt.ylim((-0.75, 1.4))
                plt.pause(0.01)
            
            if write == True : pd_write_csv("u_it0_%d.csv" %(j), u)
                
            t = cpt = 0
            while t < tf :
                i=1
                fu = []
                fu = np.asarray([0.5*u_x**2 for u_x in u])

                der_sec = [self.CFL*(u[k+1] - 2*u[k] + u[k-1]) for k in range(1, len(u)-1)]
                der_sec.insert(0, self.CFL*(u[1] - 2*u[0] + u[-1]))
                der_sec.insert(len(der_sec), self.line_x*(u[0] - 2*u[-1] + u[-2]))

                while i <= self.Nx-2 :
                    u_m, u_p = intermediaires(u, fu, i, self.dt/self.dx)
                    
                    fu_m =  0.5*u_m**2
                    fu_p =  0.5*u_p**2

                    u_nNext.append( u[i] - self.dt/self.dx*( fu_p - fu_m ) + bruit[i] + der_sec[i] )

                    i += 1  # On passe à la case d'après
                
    #            print ("size u_nNext it 0 = %d" %(np.size(u_nNext)))
                u[1:self.Nx-1] = u_nNext  
                u_nNext  = []
                
                # Conditions aux limites 
                u[0] = u[-1]
                u[1] = u[0]
                u[-1]= u[-2]
    
                if write == True : pd_write_csv("u_it%d_%d.csv" %(cpt, j), u)
                
                u = np.asarray(u) 
                cpt += 1
                t += self.dt # Itération temporelle suivante
                

                if plot == True :
                    if cpt % 20 == 0 :
                        plt.clf()
                        plt.plot(self.line_x, u, c='k') 
                        plt.title("u vs X, iteration %d bruit %d" %(cpt, j)) 
                        plt.ylim((-0.75, 1.4))  
                        plt.pause(0.1)  
                
                if cpt == 200 :
                    break
##---------------------------------------------------
    def get_obs_statistics(self):
        
##---------------------------------------------------
    def minimization(self):
        
