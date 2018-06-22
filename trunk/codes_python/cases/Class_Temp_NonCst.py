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

class Temperature_Noncst() :
##---------------------------------------------------------------
    def __init__ (self, parser, TL, BL):
        """
        This object has been made to solve optimization problem when T_inf is not a constant.
        We added too more arguments namely : TL where L stands for lambda and BL where B stands for Body
        
        TL is the lambda expression corresponding to T_inf.
        BL is the string representation of this expression, we added it to avoid re-solving both exact and inexact problems.
        """
        np.random.seed(1000) ; #plt.ion()
        
        if parser.cov_mod not in ['full', 'diag'] :
            raise AttributeError("\x1b[7;1;255mcov_mod must be either diag or full\x1b[0m")
        
        T_inf_lst = parser.T_inf_lst

        N_discr,    kappa   =   parser.N_discr, parser.kappa,    
        dt,         h       =   parser.dt,      parser.h
        datapath            =   osp.abspath(parser.datapath)
        num_real,   tol     =   parser.num_real,parser.tol
        cpt_max_adj         =   parser.cpt_max_adj
        cov_mod,    g_sup_max  =   parser.cov_mod, parser.g_sup_max
        
        self.beta_prior = np.asarray([parser.beta_prior for i in range(parser.N_discr-2)]) 
        
        z_init, z_final =   0.0, 1.0
        dz = np.abs(z_final - z_init) / float(N_discr)
        self.line_z  = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]
        
        # On construit le vecteur de T_inf qui restera inchangé au cours de la simulation.
        # On résout TL sur les points de discrétisation
        self.T_inf = map(TL, self.line_z)
        
        # On construit la clé pour faciliter les actions aux dico et communcation entre fonctions
        self.body = BL.replace(" ", "_")
        
        ## Matrices des coefficients pour la résolution
        INF1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1)
        SUP1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1) 
        A_diag1 = np.diag(np.transpose([(1 + dt/dz**2*kappa) for i in range(N_discr-2)])) 

        INF2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) 
        SUP2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)
        A_diag2 = np.diag(np.transpose([(1 - dt/dz**2*kappa) for i in range(N_discr-2)])) 

        self.A1 = A_diag1 + INF1 + SUP1
        self.A2 = A_diag2 + INF2 + SUP2
        
        self.noise = self.tab_normal(0, 0.1, N_discr-2)[0]
        self.lst_gauss = [self.tab_normal(0,0.1,N_discr-2)[0] for i in range(num_real)]
        
        self.store_rho = []
        
        self.cpt_max_adj = cpt_max_adj
        self.g_sup_max = g_sup_max  
        self.num_real = num_real
        self.eps_0 = 5.*10**(-4)
        self.cov_mod = cov_mod
        self.N_discr = N_discr
        self.tol = tol
        self.dt = dt        
        self.h = h
        
        self.bool_method = dict()
        self.bool_written= dict()
        runs = set()
        runs.add("stat")
        runs.add("opti_scipy_%s" %(self.body))
        runs.add("adj_bfgs_%s" %(self.body))
            
        for r in runs :
            self.bool_method[r] = False
            self.bool_written[r] = False
        #Création des différents fichiers
        self.date = time.strftime("%m_%d_%Hh%M", time.localtime())
        
        # On test si les dossiers existent, sinon on les créé
        if osp.exists(datapath) == False :
            os.mkdir(datapath)
        
        if osp.exists(parser.logbook_path) == False :
            os.mkdir(parser.logbook_path)
        
        if osp.exists("./err_check") == False :
            os.mkdir("./err_check")
            
        if osp.exists(osp.abspath("./data/post_cov")) == False :
            os.mkdir(osp.abspath("./data/post_cov"))
            
        self.err_title = osp.join("err_check", "%s_Ncst_err_check.csv" %(self.date))
        self.logout_title = osp.join(parser.logbook_path, "%s_Ncst_logbook.csv" %(self.date))
        
        # On intialise les fichiers en rappelant la teneur de la simulation
        for f in {open(self.logout_title, "w"), open(self.err_title, "w")} :
            f.write("\n#######################################################\n")
            f.write("## Logbook: simulation launched %s ## \n" %(time.strftime("%Y_%m_%d_%Hh%Mm%Ss", time.localtime())))
            f.write("#######################################################\n")
            f.write("Simulation\'s features :\n{}\n".format(parser))
            f.write('T_inf equation : %s\n' %(self.body))
            f.close()
        
        self.datapath   =   datapath
        self.parser     =   parser
##----------------------------------------------------##        
##----------------------------------------------------##
######                                            ######
######           Modifieurs et built-in           ######
######                                            ######
##----------------------------------------------------## 
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
    def tab_normal(self, mu, sigma, length) :
        return ( sigma * np.random.randn(length) + mu, 
                (sigma * np.random.randn(length) + mu).mean(), 
                (sigma * np.random.randn(length) + mu).std()
               ) 
##---------------------------------------------------   
    def h_beta(self, beta, verbose=False) :
#        T_n = list(map(lambda x : -4*T_inf*x*(x-1), self.line_z))
#   Initial condition
        key = self.body
        T_n= self.T_obs_mean[key]

        B_n = np.zeros((self.N_discr-2))
        T_nNext = T_n
        
        err, tol, compteur, compteur_max = 1., 1e-4, 0, 1000
        if verbose == True :
            plt.figure()
            
        while (np.abs(err) > tol) and (compteur <= compteur_max) :
            if compteur > 0 :
                T_n = T_nNext
            compteur +=1 
            
            T_n_tmp = np.dot(self.A2, T_n)
            
            for i in range(self.N_discr-2) :
                try :
                    B_n[i] = T_n_tmp[i] + self.dt*(beta[i])*self.eps_0*(self.T_inf[i]**4 - T_n[i]**4)
                except IndexError :
                    print ("i = ", i)
                    print ("B_n = ", B_n)
                    print ("T_n = ", T_n)
                    print ("T_N_tmp = ", T_n_tmp)
                    raise Exception ("Check")    
                                
            T_nNext = np.dot(np.linalg.inv(self.A1), B_n)
            
            err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
            
            if verbose == True and compteur % 5 == 0 :
                print (err)
                plt.plot(self.line_z, T_nNext, label='tracer cpt %d' %(compteur))
            
            if compteur == compteur_max :
                print("\x1b[7;1;255mWarning : H_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {}   \x1b[0m".format(err, tol))
#                time.sleep(2.5)
#        if verbose == True :
#        plt.plot(self.line_z, T_nNext, marker="o", linestyle='none')
#        plt.legend(loc="best", ncol=4)
            if verbose==True :
                print ("Err = {} ".format(err))
                print ("Compteur = ", compteur)
        
        if verbose == True :
            print("H_beta ok")
#        time.sleep(1)
        return T_nNext 
##---------------------------------------------------
    def true_beta(self, T) : 
        # dans t1 on ne rajoute pas le bruit contrairement à l'équation 36
        t1 = np.asarray([ 1./self.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i])) *10**(-4) for i in range(self.N_discr-2)])
        t2 = np.asarray([self.h / self.eps_0*(self.T_inf[i] - T[i])/(self.T_inf[i]**4 - T[i]**4)  for i in range(self.N_discr-2)]) 
        return t1 + t2        
##---------------------------------------------------    
##----------------------------------------------------##
##----------------------------------------------------##
######                                            ######
######        Génération de données et stats      ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------##
    def obs_pri_model(self) :
        T_nNext_obs_lst, T_nNext_pri_lst, T_init = [], [], []
        
        for it, bruit in enumerate(self.lst_gauss) :
            # Obs and Prior Temperature field initializations
            # For python3.5 add list( )
            obs_filename    = '{}_obs_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            pri_filename  =   '{}_pri_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            
            obs_filename = osp.join(self.datapath, obs_filename)
            pri_filename = osp.join(self.datapath, pri_filename)
            
            tol ,err_obs, err_pri, compteur = 1e-4, 1.0, 1.0, 0                
                            
            if osp.isfile(obs_filename) and osp.isfile(pri_filename) :
                continue
                            
            T_n_obs =  list(map(lambda x : -4*self.T_inf[0]*x*(x-1), self.line_z) )
            T_n_pri =  list(map(lambda x : -4*self.T_inf[0]*x*(x-1), self.line_z) ) 
            
            T_init.append(T_n_obs)
            T_nNext_obs = T_n_obs
            T_nNext_pri = T_n_pri
        
            B_n_obs     =   np.zeros((self.N_discr-2, 1))
            B_n_pri     =   np.zeros((self.N_discr-2, 1))
            T_n_obs_tmp =   np.zeros((self.N_discr-2, 1))
            T_n_pri_tmp =   np.zeros((self.N_discr-2, 1))
            
            while (np.abs(err_obs) > tol) and (compteur < 800) and (np.abs(err_pri) > tol):
                if compteur > 0 :
                    T_n_obs = T_nNext_obs
                    T_n_pri = T_nNext_pri
                compteur += 1
                
                # B_n = np.zeros((N_discr,1))
                T_n_obs_tmp = np.dot(self.A2, T_n_obs) 
                T_n_pri_tmp = np.dot(self.A2, T_n_pri)
                 
                for i in range(self.N_discr-2) :
                    B_n_obs[i] = T_n_obs_tmp[i] + self.dt*  (
                    ( 10**(-4) * ( 1.+5.*np.sin(3.*T_n_obs[i]*np.pi/200.) + 
                    np.exp(0.02*T_n_obs[i]) + bruit[i] ) ) *( self.T_inf[i]**4 - T_n_obs[i]**4)
                     + self.h * (self.T_inf[i]-T_n_obs[i])      )   
                    
                    B_n_pri[i] = T_n_pri_tmp[i] + self.dt * (5 * 10**(-4) * (self.T_inf[i]**4-T_n_pri[i]**4) * (1 + bruit[i]))
                                                        
                T_nNext_obs = np.dot(np.linalg.inv(self.A1), B_n_obs)
                T_nNext_pri = np.dot(np.linalg.inv(self.A1), B_n_pri)
                
                T_nNext_obs_lst.append(T_nNext_obs)
                T_nNext_pri_lst.append(T_nNext_pri)
            
                err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2)
                err_pri = np.linalg.norm(T_nNext_pri - T_n_pri, 2)
                # n -> n+1
                
            # On écrit le champ convergé 
            self.pd_write_csv(obs_filename, T_nNext_obs)
            self.pd_write_csv(pri_filename, T_nNext_pri)             

        print ("Calculus with {} completed. Convergence status :".format(self.body))
        print ("Err_obs = {} ".format(err_obs))    
        print ("Err_pri = {} ".format(err_pri))
        print ("Iterations = {} ".format(compteur))
        
        self.T_init             =   T_init    
        self.T_nNext_obs_lst    =   T_nNext_obs_lst
        self.T_nNext_pri_lst    =   T_nNext_pri_lst
##---------------------------------------------------   
    def get_prior_statistics(self, verbose = False):
        cov_obs_dict    =   dict() 
        cov_pri_dict    =   dict()
        
        mean_meshgrid_values_obs = dict()  
        mean_meshgrid_values_pri = dict()
        full_cov_obs_dict   =   dict()        
        vals_obs_meshpoints =   dict()
        vals_pri_meshpoints =   dict()
        
        condi   =   dict()
        T_obs_mean  =   dict()
        T_pri_mean  =   dict() 
        
        for j in range(self.N_discr-2) :
            key = self.body+"_"+str(j)
            vals_obs_meshpoints[key] =  []
            vals_pri_meshpoints[key] =  []
            
        T_obs, T_prior = [], []
        T_sum_obs = np.zeros((self.N_discr-2))
        T_sum_pri = np.zeros((self.N_discr-2))
        
        for it in range(self.num_real) :
            obs_filename  =   '{}_obs_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            pri_filename  =   '{}_pri_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            
            obs_filename = osp.join(self.datapath, obs_filename)
            pri_filename = osp.join(self.datapath, pri_filename)
            
            # Compute observationnal covariance from data 
            T_temp_obs = self.pd_read_csv(obs_filename)
            T_temp_pri = self.pd_read_csv(pri_filename)
            
            T_sum_pri += T_temp_pri / float(self.num_real)
            T_prior.append(T_temp_pri)
            
            T_sum_obs += T_temp_obs / float(self.num_real)
            T_obs.append(T_temp_obs)
        
            # Compute prior covariance from data 
            for j in range(self.N_discr-2) :
                vals_obs_meshpoints[self.body+"_"+str(j)].append(T_temp_obs[j])
                vals_pri_meshpoints[self.body+"_"+str(j)].append(T_temp_pri[j])        
            
            # We conserve the T_disc
            if verbose == True :
                plt.plot(self.line_z, T_temp_pri, label='pri real = %d' %(it), marker='o', linestyle='none')
                plt.plot(self.line_z, T_temp_obs, label='obs real = %d' %(it))
        plt.show()  
        T_obs_mean[self.body] = T_sum_obs # Joue aussi le rôle de moyenne pour la covariance
        T_pri_mean[self.body] = T_sum_pri
        
        Sum_obs = np.zeros((self.N_discr-2, self.N_discr-2))
        Sum_pri = np.zeros((self.N_discr-2, self.N_discr-2))   

        std_meshgrid_values_obs = np.asarray([np.std(vals_obs_meshpoints[self.body+"_"+str(j)]) for j in range(self.N_discr-2)])
        
        std_meshgrid_values_pri = np.asarray([np.std(vals_pri_meshpoints[self.body+"_"+str(j)]) for j in range(self.N_discr-2)])
        
        for it in range(self.num_real) :
            obs_filename  =   '{}_obs_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            pri_filename  =   '{}_pri_{}_N{}_{}.csv'.format(self.cov_mod, self.body, self.N_discr-2, it)
            obs_filename = osp.join(self.datapath, obs_filename)
            pri_filename = osp.join(self.datapath, pri_filename)

            T_temp_obs = self.pd_read_csv(obs_filename)
            T_temp_pri = self.pd_read_csv(obs_filename)
            
            for ii in range(self.N_discr-2)  :
                for jj in range(self.N_discr-2) : 
                    Sum_obs[ii,jj] += (T_temp_obs[ii] - T_obs_mean[self.body][ii]) * (T_temp_obs[jj] - T_obs_mean[self.body][jj])/float(self.num_real)
                    if ii == jj :
                        Sum_pri[ii,jj] += (T_temp_pri[ii] - T_pri_mean[self.body][ii]) * (T_temp_pri[jj] - T_obs_mean[self.body][jj])/float(self.num_real)
        
        full_cov_obs_dict[self.body] = Sum_obs
        condi['full' + self.body] = np.linalg.norm(Sum_obs)*np.linalg.norm(np.linalg.inv(Sum_obs))
        print ("cov_obs :\n{}".format(Sum_obs))
        
        std_mean_prior = np.mean(np.asarray([np.std(T_prior[i]) for i in range(len(T_prior))]))
        cov_obs_dict[self.body] = np.diag([std_meshgrid_values_obs[j]**2 for j in range(self.N_discr-2)])
#        cov_pri_dict[self.body] = np.diag([self.prior_sigma[self.body**2] for j in range(self.N_discr-2)])
        self.cov_obs_dict   =   cov_obs_dict
        self.cov_pri_dict   =   cov_pri_dict
        self.full_cov_obs_dict      =   full_cov_obs_dict
        
        cov_pri_dict[self.body] = Sum_pri
        self.cov_prior()
        
        
        condi['diag' + self.body] = np.linalg.norm(cov_obs_dict[self.body])*np.linalg.norm(np.linalg.inv(cov_obs_dict[self.body]))
        
        self.T_obs_mean     =   T_obs_mean
        
        self.vals_obs_meshpoints    =   vals_obs_meshpoints
        
        self.bool_method["stat"] = True
##----------------------------------------------------##    
    def cov_prior(self) :
        cov_obs = self.cov_obs_dict[self.body] if self.cov_mod=='diag' else\
                  self.full_cov_obs_dict[self.body]
        sigma_obs = np.diag(cov_obs)
        cov_pri = self.cov_pri_dict[self.body]
        
        b_distrib_dict = dict()
        
        for j  in range(self.N_discr-2) :
            b_distrib_dict[self.body+"_"+str(j)] = []
            
        for it in range(self.num_real) :
            s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
            b_distrib = self.beta_prior +\
                np.dot(np.linalg.cholesky(cov_pri), s)
            
            for j in range(self.N_discr-2) :
                b_distrib_dict[self.body+"_"+str(j)].append(b_distrib[j])
        sigma_pri = np.asarray([np.std(b_distrib_dict[self.body+"_"+str(j)]) for j in range(self.N_discr-2)])
        print 2*sigma_obs - sigma_pri
        
        bool_tab = [2*sigma_pri[i] - (np.abs(sigma_obs[i])) > 0 for i in range(self.N_discr-2)]
        
        if False in bool_tab :
            print False
            sys.exit(0)
##----------------------------------------------------##    
##----------------------------------------------------##  
######                                            ######
######        Routines pour l'optimisation        ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------##    
    def optimization(self, verbose=False) :
        """
        Fonction utilisant la fonction op.minimize de scipy. La méthode utilisée est BFGS.
        La dérivée est calculée à partir de la méthode utilisant les adjoints.
        """
        print("Début de l\'optimisation scipy\n")
        if self.bool_method["stat"] == False : self.get_prior_statistics()
        
        betamap, beta_final = dict(), dict()
        hess, cholesky = dict(), dict()
        
        mins_dict, maxs_dict = dict(), dict()
        
        sigma_post_dict = dict()
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        ######################
        ##-- Optimisation --##
        ######################
        
        mins, maxs = dict(), dict()
        print ("Optimisation pour T_inf = %s" %(self.body))
        
        curr_d  =   self.T_obs_mean[self.body]
        cov_prior   =  self.cov_pri_dict[self.body]
        cov_m = self.cov_obs_dict[self.body] if self.cov_mod=='diag'\
                else self.full_cov_obs_dict[self.body]           
        
        inv_cov_pri =   np.linalg.inv(cov_prior)  
        inv_cov_obs =   np.linalg.inv(cov_m)
        
        J_1 =   lambda beta :\
        0.5*np.dot(np.dot((curr_d - self.h_beta(beta)).T, inv_cov_obs), (curr_d - self.h_beta(beta)))
        
        J_2 =   lambda beta :\
        0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                    
        J = lambda beta : J_1(beta) + J_2(beta)  ## Fonction de coût
        
        print ("J(beta_prior) = {}".format(J(self.beta_prior)))

        grad_J = lambda beta :\
        np.dot(self.PSI(beta), np.diag(self.DR_DBETA(beta)) ) + self.DJ_DBETA(beta)
        
        #################
        ##-- Routine --##
        #################
        opti_obj = op.minimize(J, self.beta_prior, jac=grad_J, method="BFGS", tol=self.tol,\
                   options={"disp" : True, "maxiter" : 500})
        
        ######################
        ##-- Post Process --##
        ######################
        
        betamap[self.body]  =   opti_obj.x
        hess[self.body]     =   opti_obj.hess_inv
        self.opti_obj       =   opti_obj
        
        cholesky[self.body] =   np.linalg.cholesky(hess[self.body])
                        
        print ("Sucess state of the optimization {}".format(self.opti_obj.success))
        
        beta_final[self.body]  =   betamap[self.body] + np.dot(cholesky[self.body], s)  
        
        beta_var = []
        sigma_post = []
        
        # Construction de la distribution beta à partir de beta map et cov_betamap
        for i in range(249):
            s = self.tab_normal(0,1,self.N_discr-2)[0]
            beta_var.append(betamap[self.body] + np.dot(cholesky[self.body], s))
        beta_var.append(beta_final[self.body])
        
        # Calcule des min maxs et std sur chaque point            
        for i in range(self.N_discr-2) :
            mins[self.body + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
            maxs[self.body + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
            sigma_post.append(np.std([j[i] for j in beta_var])) 

        sigma_post_dict[self.body] = sigma_post 

        mins_lst =  [mins[self.body + str("{:03d}".format(i)) ] for i in range(self.N_discr-2)]   
        maxs_lst =  [maxs[self.body + str("{:03d}".format(i)) ] for i in range(self.N_discr-2)]
        
        mins_dict[self.body] = mins_lst
        maxs_dict[self.body] = maxs_lst
        
        self.bool_method["opti_scipy_"+self.body] = True
        
        f = open(self.logout_title, "a")
        f.write("\nSCIPY_OPTI\n")
        f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(self.body))
        f.write("SCIPY: J(beta_last) = {}\n".format(self.opti_obj.values()[5]))
        f.write("beta_last = {}\n".format(self.opti_obj.x))
        f.write("g_last = {}\n".format(np.linalg.norm(self.opti_obj.jac, np.inf)))
        
        f.write("Message : {} \t Success = {}\n".format(self.opti_obj.message, self.opti_obj.success))
        f.write("N-Iterations:  = {}\n".format(self.opti_obj.nit))
        
        self.bool_written["opti_scipy_"+self.body] = True
        f.close()
            
        ##############################
        ##-- Passages en attribut --##
        ##############################
            
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins_dict  =   mins_dict
        self.maxs_dict  =   maxs_dict
        self.beta_var   =   beta_var
        self.sigma_post_dict = sigma_post_dict
        
        #########
        #- Fin -#
        #########
##----------------------------------------------------##        
    def adjoint_bfgs(self, inter_plot=False, verbose = False) : 
        """
        
        """
        print("Début de l\'optimisation maison\n")
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        bfgs_adj_grad,   bfgs_adj_gamma     =   dict(), dict()
        bfgs_adj_bmap,   bfgs_adj_bf        =   dict(), dict()
        bfgs_adj_hessinv,bfgs_adj_cholesky  =   dict(), dict()
        
        bfgs_adj_mins,   bfgs_adj_maxs  =   dict(),  dict()
        
        bfgs_adj_mins_dict, bfgs_adj_maxs_dict = dict(), dict()
        bfgs_adj_sigma_post  = dict()
        
        self.too_low_err_hess = dict()
        sup_g_stagne = False
        
        sigmas      =   np.sqrt(np.diag(self.cov_obs_dict[self.body]))
        curr_d      =   self.T_obs_mean[self.body]
        cov_obs     =   self.cov_obs_dict[self.body] if self.cov_mod=='diag' else\
                        self.full_cov_obs_dict[self.body]
        
        cov_pri     =   self.cov_pri_dict[self.body]
        
        inv_cov_pri =   np.linalg.inv(cov_pri)  
        inv_cov_obs =   np.linalg.inv(cov_obs)
        
        J_1 =   lambda beta :\
        0.5*np.dot(np.dot(curr_d - self.h_beta(beta).T, inv_cov_obs), (curr_d - self.h_beta(beta)))
        
        J_2 =   lambda beta :\
        0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                    
        ## Fonction de coût
        J = lambda beta : J_1(beta) + J_2(beta)  
        
        grad_J = lambda beta :\
        np.dot(self.PSI(beta), np.diag(self.DR_DBETA(beta)) ) + self.DJ_DBETA(beta)
        
        err_beta = err_hess = err_j = 1            
        cpt, cptMax =   0, self.cpt_max_adj
        
        sup_g_lst = []
        corr_chol = []
        al2_lst  =  []
        
        print("J(beta_prior) = {} \t T_inf = {}".format(J(self.beta_prior), self.body))
        ########################
        ##-- Initialisation --##
        ########################

        #dJ/dBeta 
        beta_n  =   self.beta_prior
        g_n     =   grad_J(beta_n)
        
        g_sup = np.linalg.norm(g_n, np.inf)
        sup_g_lst.append(np.linalg.norm(g_n, np.inf))
        
        print ("\x1b[1;37;44mSup grad : %f \x1b[0m" %(np.linalg.norm(g_n, np.inf)))

        H_n_inv =   np.eye(self.N_discr-2)
        self.debug["first_hess"] = H_n_inv
                        
        fig, ax = plt.subplots(1,2,figsize=(13,7))
        ax[0].plot(self.line_z, beta_n, label="beta_prior")
        ax[1].plot(self.line_z, g_n,    label="gradient prior")
        
        self.alpha_lst, err_hess_lst, err_beta_lst = [], [], []
        dir_lst =   []
        
        ######################
        ##-- Optimisation --##
        ######################
        
        while (cpt<cptMax) and g_sup > self.g_sup_max :
            if cpt > 0 :
            ########################
            ##-- Incrementation --##
            ######################## 
                beta_n  =   beta_nNext
                ax[0].plot(self.line_z, beta_n, label="beta cpt%d_%s" %(cpt, self.body))
               
                g_n     =   g_nNext
                g_sup   =   np.linalg.norm(g_n, np.inf)
                
                plt.figure("Evolution de l'erreur %s" %(self.body))
                plt.scatter(cpt, g_sup, c='black')
                if inter_plot == True :
                    plt.pause(0.05)
                sup_g_lst.append(g_sup)
                if len(sup_g_lst) > 6 :
                    lst_ = sup_g_lst[(len(sup_g_lst)-5):]
                    mat = [[abs(i - j) for i in lst_] for j in lst_]
                    sup_g_stagne = (np.linalg.norm(mat, 2) <= 1e-2)
                
                print ("cpt = %d" %(cpt))
                print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))
                print("Stagne ? {}".format(sup_g_stagne))
                
                ax[1].plot(self.line_z, g_n, label="grad cpt%d" %(cpt), marker='s')

                H_n_inv   =   H_nNext_inv
                
                if verbose == True :
                    print ("beta cpt {}:\n{}".format(cpt,beta_n))
                    print("grad n = {}".format(g_n))                            
                    print("beta_n = \n  {} ".format(beta_n))
                    print("cpt = {} \t err_beta = {} \t err_hess = {}".format(cpt, \
                                                       err_beta, err_hess) )
                                                       
            test = lambda H_n_inv :  -np.dot(g_n[np.newaxis, :],\
                                np.dot(H_n_inv, g_n[:, np.newaxis]))[0,0]                
            #################
            ##-- Routine --##
            #################
            ## Calcule de la direction 
            d_n     =   - np.dot(H_n_inv, g_n)
            test_   =   (test(H_n_inv) < 0)
            print("d_n descent direction : {}".format(test_))    
                        
            if test_  == False :
                self.positive_definite_test(H_n_inv, verbose=False)

                H_n_inv = self.cholesky_for_MPD(H_n_inv, fac = 2.)
                print("cpt = {}\t cholesky for MPD used.")
                print("new d_n descent direction : {}".format(test(H_n_inv) < 0))
                
                d_n     =   - np.dot(H_n_inv, g_n)  
                corr_chol.append(cpt)

            print("d_n :\n {}".format(d_n))
            
            ## Calcule de la longueur de pas 
            ## Peut nous faire gagner du temps de calcule
            if (sup_g_stagne == True and cpt > 20) and g_sup < 10000 :
#                if g_sup < 1e-2 and cpt > 150 :
#                    # Dans ce cas on suppose qu'on n'aura pas mieux
#                    break

                alpha = 1.
                print("\x1b[1;37;44mCompteur = {}, alpha = 1.\x1b[0m".format(cpt))
                if sup_g_stagne == True :
                    print("\x1b[1;37;44mgradient stagne : coup de pouce alpha = 1. \x1b[0m")
                
                time.sleep(0.7)                    
            
            
            else :  
                alpha, al2_cor = self.backline_search(J, grad_J, g_n, beta_n ,d_n ,cpt, g_sup, rho=1e-2,c=0.5)
                if al2_cor  :
                    al2_lst.append(cpt)
            
            ## Calcule des termes n+1
            dbeta_n =  alpha*d_n
#                print ("Pas pour cpt = {}: dbeta_n = {}".format(cpt, dbeta_n))
            beta_nNext = beta_n + dbeta_n  # beta_n - alpha*d_n              

            g_nNext =   grad_J(beta_nNext)
            s_nNext =   (beta_nNext - beta_n)
            y_nNext =   g_nNext - g_n

            ## Ça arrange un peu la hessienne
            if cpt == 0 :
                fac_H = np.dot(y_nNext[np.newaxis, :], s_nNext[:, np.newaxis])
                fac_H /= np.dot(y_nNext[np.newaxis, :], y_nNext[:, np.newaxis])
                
                H_n_inv *= fac_H
                             
            H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)
                
            self.debug["curr_hess_%s" %(str(cpt))] = H_nNext_inv
#                print("Hess:\n{}".format(H_nNext_inv))
            
            err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
            
            err_j    =   J(beta_nNext) - J(beta_n)
            
            err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
            if verbose == True :
                print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                print("err_beta = {} cpt = {}".format(err_beta, cpt))
                print ("err_hess = {}".format(err_hess))
                
            self.alpha_lst.append(alpha)
            err_hess_lst.append(err_hess) 
            err_beta_lst.append(err_beta)
            dir_lst.append(np.linalg.norm(d_n, 2))
            
            print("\n")
            cpt +=  1    
            # n --> n+1 si non convergence, sort de la boucle sinon 
            
        ######################
        ##-- Post Process --##
        ######################
        H_last  =   H_nNext_inv
        g_last  =   g_nNext
        beta_last=  beta_nNext
        d_n_last = d_n
        
        logout_last = dict()
        logout_last["cpt_last"]    =   cpt   
        logout_last["g_last"]      =   g_last
        logout_last["g_sup_max"]   =   g_sup
        logout_last["beta_last"]   =   beta_last
        logout_last["J(beta_last)"]=   J(beta_last)
        
        logout_last["Residu_hess"] =   err_hess
        logout_last["Residu_beta"] =   err_beta
        
        logout_last["Corr_chol"]   =   len(corr_chol)
        
        print ("\x1b[1;35;47mFinal Sup_g = {}\nFinal beta = {}\nFinal direction {}\x1b[0m".format(\
            g_sup, beta_last, d_n_last))
        
        ax[1].plot(self.line_z, g_last, label="gradient last")

        ax[0].plot(self.line_z, beta_last, label="beta_n last")
        
        try :
            R   =   np.linalg.cholesky(H_last)
        except np.linalg.LinAlgError :
            H_last = self.cholesky_for_MPD(H_last, fac = 5.)
            R   =   H_last
            
        bfgs_adj_bmap[self.body]   =   beta_last
        bfgs_adj_grad[self.body]   =   g_last
        
        bfgs_adj_hessinv[self.body]    =   H_last            
        bfgs_adj_cholesky[self.body]   =   R
        
        bfgs_adj_bf[self.body]     =   bfgs_adj_bmap[self.body] + np.dot(R, s)
        
        self.pd_write_csv(osp.join(osp.abspath("./data/post_cov"), "adj_post_cov_%s_%s.csv" %(self.cov_mod,self.body)), pd.DataFrame(bfgs_adj_cholesky[self.body]))
        
        beta_var = []
        sigma_post = []
        
        for i in range(249):
            s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
            beta_var.append( bfgs_adj_bmap[self.body] + np.dot(R, s) )
        
        beta_var.append(bfgs_adj_bf[self.body]) # Pour faire 250
        
        for i in range(self.N_discr-2) :
            bfgs_adj_mins[self.body + str("_{:03d}".format(i))] = (min([j[i] for j in beta_var]))
            bfgs_adj_maxs[self.body + str("_{:03d}".format(i))] = (max([j[i] for j in beta_var]))
            sigma_post.append(np.std([j[i] for j in beta_var]))
             
        bfgs_adj_sigma_post[self.body] = sigma_post 
        bfgs_adj_mins_lst =  [bfgs_adj_mins["%s_%03d" %(self.body, i) ]\
                                        for i in range(self.N_discr-2)]   
        bfgs_adj_maxs_lst =  [bfgs_adj_maxs["%s_%03d" %(self.body, i) ]\
                                        for i in range(self.N_discr-2)]
        
        bfgs_adj_mins_dict[self.body] = bfgs_adj_mins_lst
        bfgs_adj_maxs_dict[self.body] = bfgs_adj_maxs_lst
        
        plt.legend(loc="best")
        
        try :
            fiig, axxes = plt.subplots(2,2,figsize=(8,8))
            axxes[0][0].set_title("alpha vs iterations ")
            axxes[0][0].plot(range(cpt), self.alpha_lst, marker='o',\
                                        linestyle='none', markersize=8)
            axxes[0][0].set_xlabel("Iterations")
            axxes[0][0].set_ylabel("alpha")
            
            axxes[0][1].set_title("err_hess vs iterations ")
            axxes[0][1].plot(range(cpt), err_hess_lst, marker='s',\
                                        linestyle='none', markersize=8)
            axxes[0][1].set_xlabel("Iterations")
            axxes[0][1].set_ylabel("norm(H_nNext_inv - H_n_inv, 2)")
            
            axxes[1][0].set_title("err_beta vs iterations ")
            axxes[1][0].plot(range(cpt), err_beta_lst, marker='^',\
                                        linestyle='none', markersize=8)
            axxes[1][0].set_xlabel("Iterations")
            axxes[1][0].set_ylabel("beta_nNext - beta_n")            
            
            axxes[1][1].set_title("||d_n|| vs iterations")
            axxes[1][1].plot(range(cpt), dir_lst, marker='v',\
                                        linestyle='none', markersize=8)
            axxes[1][1].set_xlabel("Iteration")
            axxes[1][1].set_ylabel("Direction")            
        
        except ValueError :
            pass
            

        # On l'écrit
        f = open(self.logout_title, "a")
        f.write("\nADJ_BFGS\n")
        f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(self.body))
        for item in logout_last.iteritems() :
            f.write("{} = {} \n".format(item[0], item[1])) # Voir adjoint_bfgs dans la section Post Process
        self.bool_written["adj_bfgs_"+self.body] == True
        f.close()
        
        # On écrit les erreurs 
        f = open(self.err_title, "a")
        f.write("\nADJ_BFGS\n")
        f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(self.body))
        f.write("\nErreur grad\tErreur Hess\tErreur Beta\n")
        for g,h,j in zip(sup_g_lst, err_hess_lst, err_beta_lst) :
            f.write("{:.7f} \t {:.7f} \t {:.7f}\n".format(g, h, j))
        f.close()

        ## Fin boucle sur température
        self.bool_method["adj_bfgs_"+self.body] = True
        
        #self.Hess = np.dot(g_n.T, g_n)
        self.bfgs_adj_bf     =   bfgs_adj_bf
        self.bfgs_adj_bmap   =   bfgs_adj_bmap
        self.bfgs_adj_grad   =   bfgs_adj_grad
        self.bfgs_adj_gamma  =   bfgs_adj_gamma
        
        self.bfgs_adj_hessinv=  bfgs_adj_hessinv
        self.bfgs_adj_cholesky= bfgs_adj_cholesky
        
        self.bfgs_adj_mins_dict   =   bfgs_adj_mins_dict
        self.bfgs_adj_maxs_dict   =   bfgs_adj_maxs_dict
        
        self.bfgs_adj_sigma_post     =   bfgs_adj_sigma_post
        
        self.al2_lst    =    al2_lst
        self.corr_chol  =   corr_chol
        
###---------------------------------------------------##   
######                                            ######
######    Fonctions pour calculer la Hessienne    ######
######                                            ######
##----------------------------------------------------##
##----------------------------------------------------## 
    def Next_hess(self, prev_hess_inv, y_nN, s_nN) :
        """
        Procedure close to the scipy's one 
        """
        # Comme Scipy
        rho_nN  =   1./np.dot(y_nN.T, s_nN) if np.dot(y_nN.T, s_nN) > 1e-12 else 1000.0
        self.store_rho.append(rho_nN)
        print("\x1b[1;35;47mIn Next Hess, check rho_nN = {}\x1b[0m".format(rho_nN))
        
        Id      =   np.eye(self.N_discr-2)
        
        A1 = Id - rho_nN * np.dot(s_nN[:, np.newaxis] , y_nN[np.newaxis, :])
        A2 = Id - rho_nN * np.dot(y_nN[:, np.newaxis] , s_nN[np.newaxis, :])
        
        return np.dot(A1, np.dot(prev_hess_inv, A2)) + (rho_nN* np.dot(s_nN[:, np.newaxis] ,s_nN[np.newaxis, :]))
##----------------------------------------------------##
##----------------------------------------------------##
######                                            ######
######        Routines pour le Line Search        ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------## 
    def backline_search(self, J, g_J, djk, xk, dk, cpt_ext, g_sup, rho=1., c=0.5) :
        alpha = alpha_lo = alpha_hi = 1.
        correction = False
        bool_curv  = False
        
        self.warn = "go on"
        
        # Condition d'Armijo
        armi  = lambda alpha : (J(xk + alpha*dk)) <=\
                (J(xk) + c * alpha * np.dot(djk.T, dk)) 
        # Strong Wolf Condition
        curv  = lambda alpha : (np.linalg.norm(g_J(xk + alpha*dk))) <=\
                (0.9*np.linalg.norm(djk,2))  
        
        cpt, cptmax = 0, 15
        
        # On conmmence avec le backline classique qui consiste à chercher la alpha vérifiant condition d'Armijo
        # Algo inspiré (si ce n'est calqué) de l'algo 3.1 de Nocedal and Wright
        while (armi(alpha) == False) and cpt< cptmax and self.warn=="go on" :
            alpha_lo =  alpha
            alpha   *=  rho
            cpt     +=  1
            print (alpha,  armi(alpha))
            alpha_hi =  alpha
            if alpha <= 1.e-14 :
                self.warn = "out"
                break
                # inutile de continuer dans ces cas là
        print("alpha = {}\t cpt = {}".format(alpha, cpt))
        print("Armijo = {}\t Curvature = {}".format(armi(alpha), curv(alpha)))
        
        if (alpha <= 1e-7 and cpt_ext > 10) :# and self.warn == "go on":
            print("\x1b[1;37;44mCompteur = {}\alpha from {} to 1.\x1b[0m".format(cpt_ext, alpha))
            time.sleep(0.3)
            alpha = 1.
            self.warn = "go on"
        
        else :
            print ("alpha_l = {}\t alpha hi = {}".format(alpha_lo, alpha_hi))
            bool_curv = curv(alpha)
            it = 0
            
            if cpt > 0 and bool_curv == False: # la condition cpt > 0 équivaut à alpha != 1
                # On va parcourir des alpha entre alpha_lo et alpha_hi (autour du alpha qui a vérifié armijo)
                # Pour voir si on peut trouver un alpha qui vérifie Strong Wolf 
                alpha_2 = alpha_lo 
                bool_curv = curv(alpha_2)
                
                while bool_curv == False and (alpha_2 - alpha_hi)>0 :
                    alpha_2 *= 0.7  # l'incrément peut être plus soft ou plus aigüe      
                    it  +=  1       # Pour le fun
                    bool_curv = curv(alpha_2)
                            
                if bool_curv == True :  # Si on a finalement trouvé le bon alpha
                    alpha = alpha_2     # Alors on prend celui qui vérifie les deux conditions
                    correction = True
                    print ("\x1b[1;37;43malpha_2 = {}\t alpha = {}, it = {}\x1b[0m".format(alpha_2, alpha, it))

            # On considère un cas qui n'arrive quasiment jamais
            if bool_curv == False and armi(alpha) == False :
                alpha = max(alpha, 1e-11)
                # Car en général dans ce cas la alpha environ 1e-20
                # Mettre alpha = 1 aurait été trop radical (mon avis)

        if self.warn == "out" and armi(alpha) == False :
            alpha = 1e-8 ## Au pire on recentrera avec l'itération suivante mais on veut éviter l'explosion
            print("Alpha = 1e-8 previously under 1e-14 ")
          
        if armi(alpha) == True and curv(alpha) == True :
            print("\x1b[1;37;43mArmijo = True \t Curvature = True \x1b[0m") 
            # On sait qu'ils sont True, on gagne du temps en ne recalculant pas armi(alpha) et curv(alpha)
        
        return alpha, correction
##----------------------------------------------------##   
##----------------------------------------------------##              
######                                            ######
######     Fonctions auxillaires de la classe     ######
######                                            ######
##----------------------------------------------------##
##----------------------------------------------------##
    def DR_DT(self, beta) :
        M1 = np.diag([(self.N_discr-1)**2 for i in range(self.N_discr-3)], -1) # Extra inférieure
        P1 = np.diag([(self.N_discr-1)**2 for i in range(self.N_discr-3)], +1)  # Extra supérieure
        A_diag1 = -4* np.diag((self.h_beta(beta, self.T_inf[i])**3 * beta * self.eps_0))-np.diag([2*(self.N_discr-1)**2 for i in range(self.N_discr-2) ]) # Diagonale
        result = A_diag1 + M1 + P1
#        print ("Temperature in function DR_DT = \ {}".format(self.h_beta(beta, T_inf)))
#        print("DR_DT =\n {} ".format(result))
        return  result
##----------------------------------------------------##
    def DJ_DT(self, beta) :
        curr_d = self.T_obs_mean[self.body]
        cov_m = self.cov_obs_dict[self.body] if self.cov_mod=='diag' else self.full_cov_obs_dict[self.body]        
#        print(" Cov = \n {} ".format(cov_m))        
#        return (self.h_beta(beta, T_inf) - curr_d) / (cov_m[0,0])
        result = np.dot(np.linalg.inv(cov_m), self.h_beta(beta) - curr_d) 
#        print("inv_cov \ {} ".format(np.linalg.inv(cov_m)))
#        print("DJ_DT =\n {} ".format(result))
#        return ( np.dot(np.linalg.inv(cov_m),self.h_beta(beta, T_inf) - curr_d)  )      
        return result     
##----------------------------------------------------##
    def DJ_DBETA(self, beta):
        cov_prior   =   self.cov_pri_dict[self.body]
        result = np.dot( np.linalg.inv(cov_prior), beta - self.beta_prior )
        return result
    
##----------------------------------------------------##
    def DR_DBETA(self, beta):
#        print ("DR_DBETA in function -- h_beta**4 =\n {} \n T_inf**4 = {}".format((T_inf**4 - self.h_beta(beta, T_inf)**4) * T.eps_0, T_inf**4))
        h_beta = self.h_beta(beta)
        return np.asarray([self.T_inf[i]**4 - h_beta[i]**4 for i in range(self.N_discr-2)]) * self.eps_0
##----------------------------------------------------##
    def PSI(self,beta) :
#        result = -np.dot(np.linalg.inv(self.DR_DT(beta, T_inf)), self.DJ_DT(beta, T_inf))
        result = -np.dot(self.DJ_DT(beta) , np.linalg.inv(self.DR_DT(beta)).T )
#        print("PSI in function = \n{}".format( result))
#        return -np.dot(np.linalg.inv(self.DR_DT(beta, T_inf)), self.DJ_DT(beta, T_inf))
        return result
##----------------------------------------------------##
    def positive_definite_test(self, matrix, matrix_name="H_inv", verbose = False):
        test = True
        mat_det =   np.linalg.det(matrix)     
        mat_eigvalue    =   np.linalg.eig(matrix)[0]
        
        if False in [k<0 for k in mat_eigvalue] or mat_det < 0 :
            print ("{} is not positive definite".format(matrix_name))
            
            if verbose == True :
                print("{} eigenvalues : \n{}".format(matrix_name, mat_eigvalue))
                print("{} determinant : {}".format(matrix_name, mat_det))
            test = False
        
        return test
##----------------------------------------------------##
    def cholesky_for_MPD(self, matrix, rho=10**-3, fac=2.) :
        diag = np.diag(matrix)
        mindiag = min(diag)
        if  mindiag > 0 :
            tau = 0.
        else :
            tau = - mindiag + rho
        while True :
            try :
                L = np.linalg.cholesky(matrix + tau * np.eye(matrix.shape[0]))
                break
            except np.linalg.LinAlgError :
                tau = max(fac * tau, rho)
        return L
##----------------------------------------------------## 
if __name__ == "__main__" :
    import class_functions_aux as cfa #Pour les tracés post-process
    
    parser = cfa.parser()
    print (parser)
    
    temp = Temperature(parser)
    temp.obs_pri_model()
    temp.get_prior_statistics()
#    
#    temp.adjoint_bfgs(inter_plot=True)
    
########
#              
##----------------------------------------------------##
##----------------------------------------------------##
######                                            ######
######      Cas testés avec meilleurs outputs     ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------##

#run temp_class_temp_ML.py -T_inf_lst 50 -kappa 1 -tol 1e-5 -beta_prior 1. -num_real 100 -cov_mod 'full' -N 33 -dt 1e-4 -cptmax 500
#run temp_class_temp_ML.py -T_inf_lst 50 -g_sup 1e-4 -tol 1e-5 -beta_prior 1. -num_real 100 -cov_mod 'full' -N 33 -dt 1e-4 -cptmax 150 
