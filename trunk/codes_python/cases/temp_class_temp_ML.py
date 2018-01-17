#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os.path as os

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

## Import de la classe TF ##
nnc_folder = os.abspath(os.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)
import NN_class_try as NNC
##------------------------##  

def parser() :
    parser=argparse.ArgumentParser(description='You can initialize a case you want to study')
    #lists
    parser.add_argument('--T_inf_lst', '-T_inf_lst', nargs='+', action='store', type=int, default=['all'],dest='T_inf_lst', 
                        help='List of different T_inf. Default : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]\n' )
    
    #digits
    parser.add_argument('--N_discr', '-N', action='store', type=int, default=33, dest='N_discr', 
                        help='Define the number of discretization points : default %(default)d \n' )
    parser.add_argument('--compteur_max_adjoint', '-cptmax', action='store', type=int, default=50, dest='cpt_max_adj', 
                        help='Define compteur_max (-cptmax) for adjoint method : default %(default)d \n' )

    parser.add_argument('--H', '-H', action='store', type=float, default=0.5, dest='h', 
                        help='Define the convection coefficient h \n' )
    parser.add_argument('--delta_t', '-dt', action='store', type=float, default=0.005, dest='dt', 
                        help='Define the time step disctretization. Default to %(default).5f \n' )
    parser.add_argument('--kappa', '-kappa', action='store', type=float, default=1.0, dest='kappa', 
                        help='Define the diffusivity number kappa. Default to %(default).2f\n' )
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=10, dest='num_real', 
                        help='Define the number of realization of epsilon(T) you want to pick up. Default to %(default)d\n' )
    parser.add_argument('--tolerance', '-tol', action='store', type=float, default=1e-5, dest='tol', 
                        help='Define the tolerance on the optimization error. Default to %(default).8f \n' )
    parser.add_argument('--g_sup_max', '-g_sup', action='store', type=float, default=0.1, dest='g_sup_max', 
                        help='Define the criteria on grad_J to stop the optimization. Default to %(default).5f \n' )
    
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', 
        default=1 ,dest='beta_prior', help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    
    #strings
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data', dest='datapath', 
                        help='Define the directory where the data will be stored and read. Default to %(default)s \n')
    parser.add_argument('--covariance_model', '-cov_mod', action='store', type=str, default='diag', dest='cov_mod', 
                        help='Define the covariance model. Default to %(default)s \n')
    
    return parser.parse_args()
##---------------------------------------------------------------
class Temperature() :
    def __init__ (self, parser):
        """
        This object has been made to solve optimization problem. Several methods are available and tested : _for_adjoint, optimization and minimization_with_first_guess. 
        It contains several attributes and initialiazed with a parser. 
        
        See parser 
        """
        np.random.seed(1000) ; plt.ion()
        
        if parser.cov_mod not in ['full', 'diag'] :
            raise AttributeError("\x1b[7;1;255mcov_mod must be either diag or full\x1b[0m")
        
        T_inf_lst = parser.T_inf_lst

        N_discr,    kappa   =   parser.N_discr, parser.kappa,    
        dt,         h       =   parser.dt,      parser.h
        datapath            =   os.abspath(parser.datapath)
        num_real,   tol     =   parser.num_real,parser.tol
        cpt_max_adj         =   parser.cpt_max_adj
        cov_mod,    g_sup_max  =   parser.cov_mod, parser.g_sup_max
        
        self.beta_prior = np.asarray([parser.beta_prior for i in range(parser.N_discr-2)]) 
        
        try :
            self.len_T_lst = len(T_inf_lst) 
        except TypeError :
            T_inf_lst = [T_inf_lst]
            self.len_T_lst = len(T_inf_lst) 
        
        if T_inf_lst == ['all'] :
            T_inf_lst = [i*5 for i in range(1, 11)]
        
        self.T_inf_lst = T_inf_lst
        
        z_init, z_final =   0.0, 1.0
        dz = np.abs(z_final - z_init) / float(N_discr)
        
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
        
        self.prior_sigma = dict()
        prior_sigma_lst = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]
        
        for i, t in enumerate([i*5 for i in range(1, 11)]) :
            self.prior_sigma["T_inf_%d" %(t)] = prior_sigma_lst[i]
        
        self.line_z  = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]
        self.cpt_max_adj = cpt_max_adj
        self.g_sup_max = g_sup_max  
        self.num_real = num_real
        self.eps_0 = 5.*10**(-4)
        self.cov_mod = cov_mod
        self.N_discr = N_discr
        self.tol = tol
        self.dt = dt        
        self.h = h
        
        bool_method = dict()
        for s in {"opti_scipy", "adj_bfgs", "stat", "sr1", "dump_bfgs"} :
            bool_method[s] = False
        self.bool_method = bool_method
        
        if os.exists(datapath) == False :
            os.mkdir(datapath)
        
        self.datapath = datapath
##---------------------------------------------------
    def set_T_inf (self, T_inf) :
        """
        Descr :
        ----------
        Method designed to change the T_inf_lst without running back the whole program and 
        losing information (and time).
        
        Argument:
        ----------
        T_inf : list, tuple or integer with new value(s) of T_inf you want to test. 
        
        Action(s)/Return(s):
        ---------------------
        This method will modify the class attribute T_inf_lst. The new T_inf_lst value is   
        displayed. 
        """
        try :
            self.len_T_lst = len(T_inf) 
        except TypeError :
            T_inf = [T_inf]
            self.len_T_lst = len(T_inf) 
        
        self.T_inf_lst = T_inf
        
        print("T_inf_lst is now \n{}".format(self.T_inf_lst))
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

    def pd_read_csv(self, filename) :
        if os.splitext(filename)[-1] is not ".csv" :
            filename = os.splitext(filename)[0] + ".csv"
        path = os.join(self.datapath, filename)
        data = pd.read_csv(path).get_values()            
        return data.reshape(data.shape[0])
##---------------------------------------------------
    def pd_write_csv(self, filename, data) :
        path = os.join(self.datapath, filename)
        pd.DataFrame(data).to_csv(path, index=False, header= True)
##---------------------------------------------------
    def tab_normal(self, mu, sigma, length) :
        return ( sigma * np.random.randn(length) + mu, 
                (sigma * np.random.randn(length) + mu).mean(), 
                (sigma * np.random.randn(length) + mu).std()
               ) 
##---------------------------------------------------   
    def h_beta(self, beta, T_inf, verbose=False) :
#        T_n = list(map(lambda x : -4*T_inf*x*(x-1), self.line_z))
#   Initial condition
        
        sT_inf  =   "T_inf_" + str(T_inf)
        T_n= self.T_obs_mean[sT_inf]

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
                    B_n[i] = T_n_tmp[i] + self.dt*(beta[i])*self.eps_0*(T_inf**4 - T_n[i]**4)
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
                warnings.warn("\x1b[7;1;255mH_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {} \t \x1b[0m".format(err, tol))
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
    def true_beta(self, T, T_inf) : 
        return np.asarray (
        [ 1./self.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i])) *10**(-4) + self.h / self.eps_0*(T_inf - T[i])/(T_inf**4 - T[i]**4)  for i in range(self.N_discr-2)]        
                          )
##---------------------------------------------------    
    def obs_pri_model(self) :
        T_nNext_obs_lst, T_nNext_pri_lst, T_init = [], [], []
        
        for T_inf in self.T_inf_lst :
            for it, bruit in enumerate(self.lst_gauss) :
                # Obs and Prior Temperature field initializations
                # For python3.5 add list( )
                T_n_obs =  list(map(lambda x : -4*T_inf*x*(x-1), self.line_z) )
                T_n_pri =  list(map(lambda x : -4*T_inf*x*(x-1), self.line_z) ) 
                
                T_init.append(T_n_obs)
                T_nNext_obs = T_n_obs
                T_nNext_pri = T_n_pri
            
                tol ,err_obs, err_pri, compteur = 1e-4, 1.0, 1.0, 0
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
                        np.exp(0.02*T_n_obs[i]) + bruit[i] ) ) *( T_inf**4 - T_n_obs[i]**4)
                         + self.h * (T_inf-T_n_obs[i])      )   
                        
                        B_n_pri[i] = T_n_pri_tmp[i] + self.dt * (5 * 10**(-4) * (T_inf**4-T_n_pri[i]**4) * (1 + bruit[i]))
                                                            
                    T_nNext_obs = np.dot(np.linalg.inv(self.A1), B_n_obs)
                    T_nNext_pri = np.dot(np.linalg.inv(self.A1), B_n_pri)
                    
                    T_nNext_obs_lst.append(T_nNext_obs)
                    T_nNext_pri_lst.append(T_nNext_pri)
                
                    obs_filename    =   'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                    pri_filename  =   'prior_T_inf_{}_{}.csv'.format(T_inf, it)
                    
                    self.pd_write_csv(obs_filename, T_nNext_obs)
                    self.pd_write_csv(pri_filename, T_nNext_pri)        
            
                    err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2)
                    err_pri = np.linalg.norm(T_nNext_pri - T_n_pri, 2)
            
            print ("Calculus with T_inf={} completed. Convergence status :".format(T_inf))
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
        
        if self.num_real > 50 :
            verbose == False
        
        mean_meshgrid_values=   dict()  
        full_cov_obs_dict   =   dict()        
        vals_obs_meshpoints =   dict()
        
        T_obs_mean, condi  = dict(),    dict()
        self.J_los = dict()
        
        if verbose == True :
            plt.figure("Check comparaison pri-obs-bruit")
        
        for t in self.T_inf_lst :
            for j in range(self.N_discr-2) :
                key = "T_inf_{}_{}".format(t, j)
                vals_obs_meshpoints[key] =  []
        
        for i, T_inf in enumerate(self.T_inf_lst) :
            T_obs, T_prior = [], []     
            T_sum = np.zeros((self.N_discr-2))
            sT_inf = "T_inf_" + str(T_inf)
            
            for it in range(self.num_real) :
                obs_filename  =  'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                pri_filename  =  'prior_T_inf_{}_{}.csv'.format(T_inf, it)
                
                # Compute covariance from data 
                T_temp = self.pd_read_csv(obs_filename)
                T_sum += T_temp / float(self.num_real)
                T_obs.append(T_temp)
                
                for j in range(self.N_discr-2) :
                    vals_obs_meshpoints[sT_inf+"_"+str(j)].append(T_temp[j])
                
                # We conserve the T_disc
                T_disc = self.pd_read_csv(pri_filename)
                T_prior.append(T_disc)
                if verbose == True :
                    plt.plot(self.line_z, T_disc, label='pri real = %d' %(it), marker='o', linestyle='none')
                    plt.plot(self.line_z, T_temp, label='obs real = %d' %(it))

            T_obs_mean[sT_inf] = T_sum # Joue aussi le rôle de moyenne pour la covariance
            
            Sum    =    np.zeros((self.N_discr-2, self.N_discr-2))   
            std_meshgrid_values     =   np.asarray([np.std(vals_obs_meshpoints[sT_inf+"_"+str(j)])  for j   in  range(self.N_discr-2)])
            
            for it in range(self.num_real) :
                obs_filename  =  'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                T_temp = self.pd_read_csv(obs_filename)
                
                for ii in range(self.N_discr-2)  :
                    for jj in range(self.N_discr-2) : 
                        Sum[ii,jj] += (T_temp[ii] - T_obs_mean[sT_inf][ii]) * (T_temp[jj] - T_obs_mean[sT_inf][jj])/float(self.num_real)
            
            full_cov_obs_dict[sT_inf] = Sum 
            condi['full' + sT_inf] = np.linalg.norm(Sum)*np.linalg.norm(np.linalg.inv(Sum))
            print ("cov_obs :\n{}".format(Sum))
            
            std_mean_prior          =   np.mean(np.asarray([np.std(T_prior[i]) for i in range(len(T_prior))]))
            cov_obs_dict[sT_inf]    =   np.diag([std_meshgrid_values[j]**2 for j in range(self.N_discr-2)])
            cov_pri_dict[sT_inf]    =   np.diag([self.prior_sigma[sT_inf]**2 for j in range(self.N_discr-2)])
            
            condi['diag' + sT_inf]  = np.linalg.norm(cov_obs_dict[sT_inf])*np.linalg.norm(np.linalg.inv(cov_obs_dict[sT_inf]))
            
            self.J_los[sT_inf]      =   lambda beta : 0.5 * np.sum( [((self.h_beta(beta, T_inf)[i] - T_obs_mean[sT_inf][i]))**2 for i in range(self.N_discr-2)] ) /cov_obs_dict[sT_inf][0,0]
            print (cov_pri_dict[sT_inf][0,0])
        
            self.cov_obs_dict   =   cov_obs_dict
            self.cov_pri_dict   =   cov_pri_dict
            self.T_obs_mean     =   T_obs_mean
            
            self.vals_obs_meshpoints    =   vals_obs_meshpoints
            self.full_cov_obs_dict      =   full_cov_obs_dict
            
            self.bool_method["stat"] = True
##--------------------------------------------------- -##
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
        
        if self.bool_method["stat"] == False : self.get_prior_statistics()
        
        betamap, beta_final = dict(), dict()
        hess, cholesky = dict(), dict()
        
        mins, maxs = dict(), dict()
        sigma_post_dict = dict()
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        beta_var = []
        
        
        ######################
        ##-- Optimisation --##
        ######################
        
        for T_inf in self.T_inf_lst :
            print ("Calcule pour T_inf = %d" %(T_inf))
            sT_inf  =   "T_inf_" + str(T_inf)
            
            curr_d  =   self.T_obs_mean[sT_inf]
            cov_prior   =  self.cov_pri_dict[sT_inf]
            cov_m = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]           
            
            inv_cov_pri =   np.linalg.inv(cov_prior)  
            inv_cov_obs =   np.linalg.inv(cov_m)
            
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            J = lambda beta : J_1(beta) + J_2(beta)  ## Fonction de coût
            
            print ("J(beta_prior) = {}".format(J(self.beta_prior)))

            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            #################
            ##-- Routine --##
            #################
            opti_obj = op.minimize(J, self.beta_prior, jac=grad_J, method="BFGS", tol=self.tol,\
                       options={"disp" : True, "maxiter" : 500})
            
            ######################
            ##-- Post Process --##
            ######################
            
            betamap[sT_inf] =   opti_obj.x
            hess[sT_inf]    =   opti_obj.hess_inv
            self.opti_obj   =   opti_obj
            cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf])
                            
            print ("Sucess state of the optimization {}".format(self.opti_obj.success))
            
            beta_final[sT_inf]  =   betamap[sT_inf] + np.dot(cholesky[sT_inf], s)  
            
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append(betamap[sT_inf] + np.dot(cholesky[sT_inf], s))
            beta_var.append(beta_final[sT_inf])
            sigma_post = []
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var])) 
            sigma_post_dict[sT_inf] = sigma_post 
            mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
            maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
        
        
        ##############################
        ##-- Passages en attribut --##
        ##############################
            
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins_lst   =   mins_lst
        self.maxs_lst   =   maxs_lst
        self.beta_var   =   beta_var
        self.sigma_post_dict = sigma_post_dict
        
        self.bool_method["opti_scipy"]   =   True
        
        #########
        #- Fin -#
        #########
##----------------------------------------------------##
    def optimization_2(self, verbose=False) :
        """
        Fonction utilisant la fonction op.minimize de scipy. La méthode utilisée est BFGS.
        La dérivée est calculée à partir de la méthode utilisant les adjoints.
        """
        if self.bool_method["stat"] == False : self.get_prior_statistics()
        
        betamap, beta_final = dict(), dict()
        hess, cholesky = dict(), dict()
        
        mins, maxs = dict(), dict()
        sigma_post_dict = dict()
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        beta_var = []
        
        for T_inf in self.T_inf_lst :
            print ("Calculus for T_inf = %d" %(T_inf))
            sT_inf  =   "T_inf_" + str(T_inf)
            
            curr_d  =   self.T_obs_mean[sT_inf]
            cov_prior   =  self.cov_pri_dict[sT_inf]
            cov_m = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]           
            
            if verbose == True : 
                print("shapes to debug :")
                print("shape of cov_m = {}".format(cov_m.shape) )
                print("shape of cov_p = {}".format(cov_prior.shape) )
                print("shape of curr_d = {}".format(curr_d.shape) )
                
            J = lambda beta : 0.5 * np.dot((self.h_beta(beta, T_inf) - curr_d).T, np.dot(np.linalg.inv(cov_m),(self.h_beta(beta, T_inf) - curr_d)))

            print ("J = {}".format(J(self.beta_prior)))

            grad_J = lambda beta : np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) )
            
            opti_obj = op.minimize(J, self.beta_prior, jac=grad_J, method="BFGS", tol=self.tol,\
                       options={"disp" : True})
            
            print("grad_optimization:\n{}".format(opti_obj.jac))
            betamap[sT_inf] =   opti_obj.x
#            hbeta =  self.h_beta(betamap[sT_inf], T_inf, verbose=False)
            
#            f = np.diag([( hbeta[p] - curr_d[p] ) / sigmas[p] for p in range(self.N_discr-2)])
#            df_I_dT =   1. / sigmas[p]

#            phi_t = - np.dot( df_I_dT, np.linalg.inv(self.DR_DT(beta_n, T_inf)) )
#            
#            grad_j  = np.dot( phi_t, self.DR_DBETA(beta_n, T_inf) )
#            jac_j   =   np.diag(grad_j)
#            hess[sT_inf] = np.linalg.inv( 2.*np.eye(self.N_discr-2)  )
            self.opti_obj   =   opti_obj
            try :
                hess[sT_inf] = self.H_formule_2(betamap[sT_inf], cov_prior, T_inf)
                cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf])
            except np.linalg.LinAlgError  :
                sigmas = np.sqrt(np.diag(self.cov_obs_dict[sT_inf]))
                print ("Cholesky impossible avec H_formule. Plan B : adjoint de la solution")
                hbeta   =   self.h_beta(betamap[sT_inf], T_inf, verbose=False)
                f       =   np.diag([( hbeta[p] - curr_d[p] ) / sigmas[p] for p in range(self.N_discr-2)])
                df_I_dT =   -1. / sigmas[p]

                phi_t =    -np.dot( df_I_dT, np.linalg.inv(self.DR_DT(betamap[sT_inf], T_inf)) )
                
                grad_j  =   np.dot( phi_t, self.DR_DBETA(betamap[sT_inf], T_inf) )
                jac_j   =   np.diag(grad_j)
            
                h = np.dot(jac_j.T, jac_j)
            
                hess[sT_inf] = np.linalg.inv(h)
                cholesky[sT_inf] = np.linalg.cholesky(hess[sT_inf])
                
#            hess[sT_inf]    =   opti_obj.hess_inv
#            hessienne = np.dot(np.linalg.inv(cov_m), np.diag([2 for i in range(T.N_discr-2)]))
#            hess[sT_inf] = np.linalg.inv(hessienne) 
            print ("Sucess state of the optimization {}".format(self.opti_obj.success))
            
            beta_final[sT_inf]  =   betamap[sT_inf] + np.dot(cholesky[sT_inf], s)  
            
            for i in range(99):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append(betamap[sT_inf] + np.dot(cholesky[sT_inf], s))
            beta_var.append(beta_final[sT_inf])
            sigma_post = []
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var])) 
            sigma_post_dict[sT_inf] = sigma_post 
            mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
            maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
            
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins_lst   =   mins_lst
        self.maxs_lst   =   maxs_lst
        self.beta_var   =   beta_var
        self.sigma_post_dict = sigma_post_dict
        
        self.bool_method["opti_scipy"] = True
##----------------------------------------------------##
##----------------------------------------------------##        
    def adjoint_bfgs(self, inter_plot=False) : 
        """
        
        """
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        bfgs_adj_grad,   bfgs_adj_gamma     =   dict(), dict()
        bfgs_adj_bmap,   bfgs_adj_bf        =   dict(), dict()
        bfgs_adj_hessinv,bfgs_adj_cholesky  =   dict(), dict()
        
        bfgs_adj_mins,   bfgs_adj_maxs  =   dict(),  dict()
        
        bfgs_adj_sigma_post  = dict()
        beta_var = []
        
        frozen = 0
        self.too_low_err_hess = dict()
        
        for T_inf in self.T_inf_lst :
            sT_inf      =   "T_inf_%d" %(T_inf)
            sigmas = np.sqrt(np.diag(self.cov_obs_dict[sT_inf]))
            curr_d      =   self.T_obs_mean[sT_inf]
            cov_obs     =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                            self.full_cov_obs_dict[sT_inf]
            
            cov_pri     =   self.cov_pri_dict[sT_inf]
            
            inv_cov_pri =   np.linalg.inv(cov_pri)  
            inv_cov_obs =   np.linalg.inv(cov_obs)
            
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            ## Fonction de coût
            J = lambda beta : J_1(beta) + J_2(beta)  
            
            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            err_beta = err_hess = err_j = 1            
            cpt, cptMax =   0, self.cpt_max_adj
            
            sup_g_lst = []
            corr_chol = []
            al2_lst  =  []
                        
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
                    beta_nPrev  =   beta_n             
                    beta_n  =   beta_nNext
                    ax[0].plot(self.line_z, beta_n, label="beta cpt%d" %(cpt))
                    print ("beta cpt {}:\n{}".format(cpt,beta_n))
                   
                    g_nPrev =   g_n
                    g_n     =   g_nNext
                    g_sup   =   np.linalg.norm(g_n, np.inf)
                    
                    plt.figure("Evolution de l'erreur")
                    plt.scatter(cpt, g_sup, c='black')
                    if inter_plot == True :
                        plt.pause(0.05)
                    sup_g_lst.append(g_sup)
                    
                    print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))

                    ax[1].plot(self.line_z, g_n, label="grad cpt%d" %(cpt), marker='s')

                    H_n_inv   =   H_nNext_inv
                    
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
                
                alpha, al2_cor =  self.backline_search(J, grad_J, g_n, beta_n, d_n, rho=1e-2, c=0.5)
                if al2_cor  :
                    al2_lst.append(cpt)
                
                if (alpha <= 1e-7 and cpt > 70) and g_sup > self.g_sup_max :
                    print("\x1b[1;37;44mCompteur = {}, alpha = 1.\x1b[0m".format(cpt))
                    time.sleep(1.)
                    alpha = 1.
                
                
#                print("alpha for cpt {}: {}".format(cpt, alpha))
#                if self.warn == "out" :
#                    print("Alpha lower than 5e-12. Convergence issue ?\n")
#                    break
#                      
#                if cpt < cpt_inter :
###                    dbeta = dbeta/10000000
#                    dbeta_n /= 1000000
                
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
                                 
#                if np.linalg.norm(s_nNext,2) < 1e-8: 
#                    frozen +=1
#                    if frozen <=6 :
#                        print("H_nNext = H_n, cpt = {} \t frozen = {}".format(cpt, frozen))
#                        H_nNext_inv = H_n_inv ## Cad le determinant rho est trop petit, une mise à jour n'est pas nécessaire                    
#                    else :
#                        H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)
##                    break
##                else :
#                else :
#                    H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)

#                if frozen == 7 : print("s_nN petit, convergence a priori"); break
##                H_nNext_inv = self.H_formule(beta_n, self.cov_pri_dict["T_inf_%s"%(str(T_inf))], T_inf)
                
#                if np.abs(err_hess) <= 1e-7 :
#                    H_nNext_inv = H_n_inv
#                    print ("err_hess < 1e-7, H_nNext = H_n, cpt = {}".format(cpt))
#                    self.too_low_err_hess["cpt_" + str(cpt)] = err_hess
#                else :
                H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)
                    
                self.debug["curr_hess_%s" %(str(cpt))] = H_nNext_inv
#                print("Hess:\n{}".format(H_nNext_inv))
                
                err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
                print("err_beta = {} cpt = {}".format(err_beta, cpt))
                
                err_j    =   J(beta_nNext) - J(beta_n)
                print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                
                err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
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
            
            print ("Final Sup_g = {}\nFinal beta = {}\nFinal direction {}".format(g_sup,\
                                      beta_last,       d_n_last  ))
            
            ax[1].plot(self.line_z, g_last, label="gradient last")

            ax[0].plot(self.line_z, beta_last, label="beta_n last")
            
            try :
                R   =   np.linalg.cholesky(H_last)
            except np.linalg.LinAlgError :
                H_last = self.cholesky_for_MPD(H_last, fac = 5.)
                R   =   H_last
#                print("HH eign: \n{}".format(np.linalg.eig(H_last)[0]))
#                print('det(HH) ={}'.format(np.linalg.det(H_last)))
                
            bfgs_adj_bmap[sT_inf]   =   beta_last
            bfgs_adj_grad[sT_inf]   =   g_last
            
            bfgs_adj_hessinv[sT_inf]    =   H_last            
            bfgs_adj_cholesky[sT_inf]   =   R
            
            bfgs_adj_bf[sT_inf]     =   bfgs_adj_bmap[sT_inf] + np.dot(R, s)
            
            
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append( bfgs_adj_bmap[sT_inf] + np.dot(R, s) )
            
            beta_var.append(bfgs_adj_bf[sT_inf]) # Pour faire 250
            sigma_post = []
            
            for i in range(self.N_discr-2) :
                bfgs_adj_mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                bfgs_adj_maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var]))
                 
            bfgs_adj_sigma_post[sT_inf] = sigma_post 
            bfgs_mins_lst =  [bfgs_adj_mins["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]   
            bfgs_maxs_lst =  [bfgs_adj_maxs["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]
            
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
                break
        #self.Hess = np.dot(g_n.T, g_n)
        self.bfgs_adj_bf     =   bfgs_adj_bf
        self.bfgs_adj_bmap   =   bfgs_adj_bmap
        self.bfgs_adj_grad   =   bfgs_adj_grad
        self.bfgs_adj_gamma  =   bfgs_adj_gamma
        
        self.bfgs_adj_hessinv=  bfgs_adj_hessinv
        self.bfgs_adj_cholesky= bfgs_adj_cholesky
        
        
        self.bfgs_adj_maxs   =   bfgs_maxs_lst
        self.bfgs_adj_mins   =   bfgs_mins_lst
        
        self.bfgs_adj_sigma_post     =   bfgs_adj_sigma_post
        
        self.al2_lst    =    al2_lst
        self.corr_chol  =   corr_chol
        
        self.bool_method["adj_bfgs"] = True
###---------------------------------------------------##   
##----------------------------------------------------##        
    def dumped_bfgs(self, inter_plot=False) : 
        """
        
        """
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        sigmas = np.sqrt(np.diag(self.cov_obs_dict[sT_inf]))
        
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        dump_bfgs_grad,   dump_bfgs_gamma     =   dict(), dict()
        dump_bfgs_bmap,   dump_bfgs_bf        =   dict(), dict()
        dump_bfgs_hessinv,dump_bfgs_cholesky  =   dict(), dict()
        
        dump_bfgs_mins,   dump_bfgs_maxs  =   dict(),  dict()
        
        dump_bfgs_sigma_post  = dict()
        beta_var = []
                
        for T_inf in self.T_inf_lst :
            sT_inf      =   "T_inf_%d" %(T_inf)
            curr_d      =   self.T_obs_mean[sT_inf]
            cov_obs     =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                            self.full_cov_obs_dict[sT_inf]
            
            cov_pri     =   self.cov_pri_dict[sT_inf]
            
            inv_cov_pri =   np.linalg.inv(cov_pri)  
            inv_cov_obs =   np.linalg.inv(cov_obs)
            
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            ## Fonction de coût
            J = lambda beta : J_1(beta) + J_2(beta)  
            
            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            err_beta = err_hess = err_j = 1            
            cpt, cptMax =   0, self.cpt_max_adj
            
            sup_g_lst = []
            corr_chol = []
            al2_lst  =   []
                        
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
                    beta_nPrev  =   beta_n             
                    beta_n  =   beta_nNext
                    ax[0].plot(self.line_z, beta_n, label="beta cpt%d" %(cpt))
                    print ("beta cpt {}:\n{}".format(cpt,beta_n))
                   
                    g_nPrev =   g_n
                    g_n     =   g_nNext
                    g_sup   =   np.linalg.norm(g_n, np.inf)
                    
                    plt.figure("Evolution de l'erreur")
                    plt.scatter(cpt, g_sup, c='black')
                    if inter_plot == True :
                        plt.pause(0.05)
                    sup_g_lst.append(g_sup)
                    
                    print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))

                    ax[1].plot(self.line_z, g_n, label="grad cpt%d" %(cpt), marker='s')

                    H_n_inv   =   H_nNext_inv
                    
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
                
                alpha, al2_cor =  self.backline_search(J, grad_J, g_n, beta_n, d_n, rho=0.1, c=0.5)
                if al2_cor == True :
                    al2_lst.append(cpt)
                
                print("alpha for cpt {}: {}".format(cpt, alpha))
                              
#                if cpt < cpt_inter :
###                    dbeta = dbeta/10000000
#                    dbeta_n /= 1000000
                
                ## Calcule des termes n+1
                dbeta_n =  alpha*d_n
                print ("Pas pour cpt = {}: dbeta_n = {}".format(cpt, dbeta_n))
                beta_nNext = beta_n + dbeta_n  # beta_n - alpha*d_n              

                g_nNext =   grad_J(beta_nNext)
                s_nNext =   (beta_nNext - beta_n)
                y_nNext =   g_nNext - g_n

                ## Ça arrange un peu la hessienne
                if cpt == 0 :
                    fac_H = np.dot(y_nNext[np.newaxis, :], s_nNext[:, np.newaxis])
                    fac_H /= np.dot(y_nNext[np.newaxis, :], y_nNext[:, np.newaxis])
                    
                    H_n_inv *= fac_H
                
                H_nNext_inv = self.Next_Dumped_BFGS(H_n_inv, y_nNext, s_nNext)
                
                self.debug["curr_hess"] = H_nNext_inv
#                print("Hess:\n{}".format(H_nNext_inv))
                
                err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
                print("err_beta = {} cpt = {}".format(err_beta, cpt))
                
                err_j    =   J(beta_nNext) - J(beta_n)
                print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                
                err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
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
            
            print ("Final Sup_g = {}\nFinal beta = {}\nFinal direction {}".format(g_sup,\
                                      beta_last,       d_n_last  ))
            
            ax[1].plot(self.line_z, g_last, label="gradient last")

            ax[0].plot(self.line_z, beta_last, label="beta_n last")
            
            try :
                R   =   np.linalg.cholesky(H_last)
            except np.linalg.LinAlgError :
                H_last = self.cholesky_for_MPD(H_last, fac = 5.)
                R   =   H_last
#                print("HH eign: \n{}".format(np.linalg.eig(H_last)[0]))
#                print('det(HH) ={}'.format(np.linalg.det(H_last)))
                
            dump_bfgs_bmap[sT_inf]   =   beta_last
            dump_bfgs_grad[sT_inf]   =   g_last
                        
            dump_bfgs_bf[sT_inf]     =   dump_bfgs_bmap[sT_inf] + np.dot(R, s)
            
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append( dump_bfgs_bmap[sT_inf] + np.dot(R, s) )
            
            beta_var.append(dump_bfgs_bf[sT_inf]) # Pour faire 250
            sigma_post = []
            
            for i in range(self.N_discr-2) :
                dump_bfgs_mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                dump_bfgs_maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var]))
                 
            dump_bfgs_sigma_post[sT_inf] = sigma_post 
            dump_bfgs_mins_lst =  [dump_bfgs_mins["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]   
            dump_bfgs_maxs_lst =  [dump_bfgs_maxs["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]
            
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
                break
        #self.Hess = np.dot(g_n.T, g_n)
        self.dump_bfgs_bf     =   dump_bfgs_bf
        self.dump_bfgs_bmap   =   dump_bfgs_bmap
        self.dump_bfgs_grad   =   dump_bfgs_grad
        self.dump_bfgs_gamma  =   dump_bfgs_gamma
        
        self.dump_bfgs_maxs   =   dump_bfgs_maxs_lst
        self.dump_bfgs_mins   =   dump_bfgs_mins_lst
        
        self.dump_bfgs_sigma_post     =   bfgs_adj_sigma_post
        
        self.dump_bfgs_al2_lst    =    al2_lst
        self.dump_bfgs_corr_chol  =   corr_chol
        
        self.bool_method["dump_bfgs"] = True
###---------------------------------------------------##  
    def adj_SR1(self, inter_plot=False): 
        """
        
        """
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        sigmas = np.sqrt(np.diag(self.cov_obs_dict[sT_inf]))
        
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        sr1_grad,   sr1_gamma     =   dict(), dict()
        sr1_bmap,   sr1_bf        =   dict(), dict()
        sr1_hessinv,sr1_cholesky  =   dict(), dict()
        
        sr1_mins,   sr1_maxs  =   dict(),  dict()
        
        sr1_sigma_post  = dict()
        beta_var = []
                
        for T_inf in self.T_inf_lst :
            sT_inf      =   "T_inf_%d" %(T_inf)
            curr_d      =   self.T_obs_mean[sT_inf]
            cov_obs     =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                            self.full_cov_obs_dict[sT_inf]
            
            cov_pri     =   self.cov_pri_dict[sT_inf]
            
            inv_cov_pri =   np.linalg.inv(cov_pri)  
            inv_cov_obs =   np.linalg.inv(cov_obs)
            
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            ## Fonction de coût
            J = lambda beta : J_1(beta) + J_2(beta)  
            
            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            err_beta = err_hess = err_j = 1            
            cpt, cptMax =   0, self.cpt_max_adj
            
            sup_g_lst   =   []
            corr_chol   =   []
            al2_lst     =   []
                        
            ########################
            ##-- Initialisation --##
            ########################

            beta_n  =   self.beta_prior
            g_n     =   grad_J(beta_n)
            
            g_sup = np.linalg.norm(g_n, np.inf)
            sup_g_lst.append(np.linalg.norm(g_n, np.inf))
            
            print ("\x1b[1;37;44mFirst Sup grad : %f \x1b[0m" %(np.linalg.norm(g_n, np.inf)))

            H_n_inv =   np.eye(self.N_discr-2)
            self.debug["first_hess"] = H_n_inv
                            
            fig, ax = plt.subplots(1,2,figsize=(13,7))
            ax[0].plot(self.line_z, beta_n, label="beta_prior")
            ax[1].plot(self.line_z, g_n,    label="gradient prior")
            
            self.alpha_lst, err_hess_lst, err_beta_lst = [], [], []
            dir_lst =   []
            itt = 0
            ######################
            ##-- Optimisation --##
            ######################
            
            while (cpt<cptMax) and g_sup > self.g_sup_max :
                if cpt > 0 :
                ########################
                ##-- Incrementation --##
                ######################## 
                    beta_n  =   beta_nNext
                    #Traće des différents beta au cours de la boucle
                    ax[0].plot(self.line_z, beta_n, label="beta cpt%d" %(cpt))
                    print ("beta cpt {}:\n{}".format(cpt,beta_n))
                   
                    g_n     =   g_nNext
                    g_sup   =   np.linalg.norm(g_n, np.inf)
                    
                    # Tracé de l'erreur sur le gradient
                    plt.figure("Evolution de l'erreur")
                    plt.scatter(cpt, g_sup, c='black')
                    if inter_plot == True :
                        plt.pause(0.05)
                    sup_g_lst.append(g_sup)
                    
                    print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))
                    
                    #Tracé du gradient 
                    ax[1].plot(self.line_z, g_n, label="grad cpt%d" %(cpt), marker='s')

                    H_n_inv   =   H_nNext_inv
                    
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
                
                alpha, al2_cor =  self.backline_search(J, grad_J, g_n, beta_n, d_n, rho=0.1, c=0.5)
                if al2_cor == True :
                    al2_lst.append(cpt)
                
                print("alpha for cpt {}: {}".format(cpt, alpha))
                              
                ## Calcule des termes n+1
                dbeta_n =  alpha*d_n
                print ("Pas pour cpt = {} \ndbeta_n = {}".format(cpt, dbeta_n))
                beta_nNext = beta_n + dbeta_n  # beta_n - alpha*d_n              

                g_nNext =   grad_J(beta_nNext)
                s_nNext =   (beta_nNext - beta_n)
                
                # SR1 
                r = 1.e-8
                y_nNext =   g_nNext - g_n

                bksk = np.dot(H_n_inv, s_nNext)
                yk_bksk= y_nNext - bksk
                
                t1 = abs( np.dot( s_nNext[np.newaxis, :], (y_nNext[:, np.newaxis] - np.dot(H_n_inv, s_nNext[:, np.newaxis])) )[0,0] )
                t2 = r * np.linalg.norm(s_nNext, 2) * np.linalg.norm(yk_bksk, 2)
                
                ## Pour éviter de diviser par zéro
                test_sr1 = (t1 >= t2) 
                print test_sr1

                if cpt == 0 :
                # Ça arrange un peu la hessienne
                    fac_H =  np.dot(y_nNext[np.newaxis, :], s_nNext[:, np.newaxis])
                    fac_H /= np.dot(y_nNext[np.newaxis, :], y_nNext[:, np.newaxis])
                    
                    H_n_inv *= fac_H
                                
                if test_sr1 == True :
                    H_nNext_inv = self.sr1_Next_hess(H_n_inv, yk_bksk, s_nNext)
                
                else :
                    if np.linalg.norm(s_nNext,2) < 1e-8 :
                        print("SR1 non adapté np.linalg.norm(s_nNext,2) = {}".format(np.linalg.norm(s_nNext,2)))
                        itt+=1
                        print("erreur vu {} fois".format(itt))
                        
                    H_nNext_inv = H_n_inv
                
                self.debug["curr_hess"] = H_nNext_inv
#                print("Hess:\n{}".format(H_nNext_inv))
                
                err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
                print("err_beta = {} cpt = {}".format(err_beta, cpt))
                
                err_j    =   J(beta_nNext) - J(beta_n)
                print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                
                err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
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
            
            print ("Final Sup_g = {}\nFinal beta = {}\nFinal direction {}".format(g_sup,\
                                      beta_last,       d_n_last  ))
            
            ax[1].plot(self.line_z, g_last, label="gradient last")

            ax[0].plot(self.line_z, beta_last, label="beta_n last")
            
            try :
                R   =   np.linalg.cholesky(H_last)
            except np.linalg.LinAlgError :
                H_last = self.cholesky_for_MPD(H_last, fac = 5.)
                R   =   H_last
#                print("HH eign: \n{}".format(np.linalg.eig(H_last)[0]))
#                print('det(HH) ={}'.format(np.linalg.det(H_last)))
                
            sr1_bmap[sT_inf]   =   beta_last
            sr1_grad[sT_inf]   =   g_last
                        
            sr1_bf[sT_inf]     =   sr1_bmap[sT_inf] + np.dot(R, s)
            
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append( sr1_bmap[sT_inf] + np.dot(R, s) )
            
            beta_var.append(sr1_bf[sT_inf]) # Pour faire 250
            sigma_post = []
            
            for i in range(self.N_discr-2) :
                sr1_mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                sr1_maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var]))
                 
            sr1_sigma_post[sT_inf] = sigma_post 
            sr1_mins_lst =  [sr1_mins["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]   
            sr1_maxs_lst =  [sr1_maxs["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]
            
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
                break
        #self.Hess = np.dot(g_n.T, g_n)
        self.sr1_bf     =   sr1_bf
        self.sr1_bmap   =   sr1_bmap
        self.sr1_grad   =   sr1_grad
        self.sr1_gamma  =   sr1_gamma
        
        self.sr1_maxs   =   sr1_maxs_lst
        self.sr1_mins   =   sr1_mins_lst
        
        self.sr1_sigma_post     =   sr1_sigma_post
        
        self.al2_lst    =    al2_lst
        self.corr_chol  =   corr_chol
        
        self.bool_method["sr1"] = True 
###---------------------------------------------------##
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
    def Next_hess_bis(self, Bkm1, yk, sk) :
        """
        Procedure Wikipédia 
        """
        skL = sk[np.newaxis, :] #skLigne
        skC = sk[:, np.newaxis] #skColone
        
        ykL = yk[np.newaxis, :] #ykLigne
        ykC = yk[:, np.newaxis] #ykColone
        
        
        fac_1   =   np.dot(skL, ykC)[0,0] #sk.T yk
        fac_2   =   np.dot(ykL, np.dot(Bkm1, ykC))[0,0] #yk.T B yk
        
        skskT   =   skL * skC
        bkykskT =   np.dot(Bkm1, np.dot(ykC, skL))
        
        skykTbk =   np.dot(np.dot(skC, ykL), Bkm1)
        
        t2  =   (fac_1 + fac_2) / (fac_1**2) * skskT
        t3 = bkykskT + skykTbk / fac_1 
        
        return Bkm1 + t2 - t3
##----------------------------------------------------##
    def sr1_Next_hess(self, Bk, yk_bksk, s_nNext) : 
        """
        Formule d'update de rang 1 appelé SR1 voir Nocedal and Wright
        """
        deno=   np.dot(yk_bksk.reshape(1,-1), s_nNext)[0]
        print deno
        mat =   np.dot(yk_bksk.reshape(-1,1), yk_bksk.reshape(1,-1))
        
        mat /=  deno
        
#        print("mat sr1 = {}".format(mat))
        
        return np.linalg.inv(Bk + mat)
##----------------------------------------------------##   
    def Next_Dumped_BFGS(self, Bk, yk, sk): 
        """
        Formule de Dumped BFGS.
        """
        fac_1   =   np.dot(sk[np.newaxis, :], yk[:, np.newaxis])[0,0]
        fac_2   =   np.dot(sk[np.newaxis, :], np.dot(Bk, sk[:, np.newaxis]))[0,0]

        t1 = fac_1
        t2 = 0.2*fac_2
        
        # Construction de theta_k abrégé tk. C'est un scalaire
        if t1 >= t2 : 
            tk = 1.
        else :
            tk = 0.8*fac_2 / (fac_2 - fac_1) 
        
        rk = tk * yk + (1-tk)*np.dot(Bk, sk)
        
        mat1 = np.dot(np.dot(Bk, sk)[:, np.newaxis], np.dot(sk[np.newaxis, :], Bk))
        mat1 /= fac_2
        
        mat2 = np.dot(rk[:, np.newaxis], rk[np.newaxis, :])
        mat2 /= np.dot(sk[np.newaxis ,:], rk)[0]
        
        return  np.linalg.inv(Bk - mat1 + mat2)
##----------------------------------------------------##   
    def Next_hess_further(self, prev_hess_inv, y_nN, s_nN):
        ## Trop lente
        """
        Formula from https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
        Didn't find it anywhere else. Must check it out
        """
        
        d = self.N_discr-2
        scal_1  =   np.dot(s_nN.T, y_nN)
        scal_2  =   np.dot(np.dot(y_nN.T, prev_hess_inv), y_nN)

        M_1 =   np.dot(s_nN.reshape(d,-1), s_nN.reshape(1,-1))
        M_2 =   np.dot( (np.dot(prev_hess_inv, y_nN)).reshape(d,-1),s_nN.reshape(1,-1) )
        M_3 =   np.dot( np.dot(s_nN.reshape(d,-1), y_nN.reshape(1,-1)), prev_hess_inv )
        
        T_1 =   prev_hess_inv
        T_2 =   (scal_1 + scal_2) * M_1 / scal_1**2
        T_3 =   (M_2 + M_3) / scal_1
                
        return T_1 + T_2 - T_3
##----------------------------------------------------##
    def H_formule(self, beta, cov_pri, T_inf):
        dR_dT   =   self.DR_DT(beta, T_inf)
        dR_dBeta=   self.DR_DBETA(beta, T_inf)
        h_beta  =   self.h_beta(beta, T_inf)
        
        nu      =   - np.dot( np.diag(dR_dBeta), np.linalg.inv(dR_dT) )
        psi     =   self.PSI(beta, T_inf)

        nu_diag = np.diag(np.diag(nu))

        M_1 =   np.diag( [- 4*psi[m]*self.eps_0*h_beta[m]**3 for m in range(self.N_discr-2)] )

        M_4 =   np.linalg.inv(dR_dT) 

        Mu  =   np.dot( -M_1 , M_4 )
        
        H_1 =   np.linalg.inv(cov_pri.T)
        H_2 =   np.dot(Mu, np.diag( [self.eps_0*(T_inf**4 - h_beta[j]**4) for j in range(self.N_discr-2)] ) )
        H_3 =   - 4 * np.dot(nu, np.diag([self.eps_0*psi[j]*h_beta[j]**3 for j in range(self.N_discr-2)])) 

        H   =   H_1 + H_2 + H_3 
        
        return np.linalg.inv(H)
##----------------------------------------------------##
    def H_formule_2(self, beta, cov_pri, T_inf):
        dR_dT   =   self.DR_DT(beta, T_inf)
        dR_dBeta=   self.DR_DBETA(beta, T_inf)
        h_beta  =   self.h_beta(beta, T_inf)
        
        psi     =   self.PSI(beta, T_inf)
        
        H_1 =   np.linalg.inv(cov_pri.T) ## d2J/dbidbj
        H_2 =   - 4 * np.diag([self.eps_0*psi[j]*h_beta[j]**3 for j in range(self.N_discr-2)]) 
        
        H_2 =   np.dot(H_2, np.linalg.inv(dR_dT))
        H_2 =   np.dot(H_2, np.diag(dR_dBeta))
        

        H   =   H_1 - H_2 
        
        return np.linalg.inv(H)
##----------------------------------------------------##
##----------------------------------------------------##
######                                            ######
######        Routines pour le Line Search        ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------##
    def search_alpha(self, func, func_prime, curr_beta, err_grad, alpha=1.) :
#        https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080 Corollaire 2
#        alpha_m = lambda m : float(alpha) / (2**(m-1))
#       Deuxieme test avec https://en.wikipedia.org/wiki/Backtracking_line_search 
        
        cptmax = 18 if err_grad > 1e3 else 50
        mm = 1
        while ((func(curr_beta - alpha*func_prime)) <= \
                 func(curr_beta) - 0.5*alpha * np.linalg.norm(func_prime)**2 ) == False and mm < cptmax:
            alpha *=0.25
            mm += 1
        
        return alpha
##----------------------------------------------------## 
    def backline_search(self, J, g_J, djk, xk, dk, rho=1., c=0.5) :
        alpha = alpha_lo = alpha_hi = 1.
        correction = False
        bool_curv  = False
        
        self.warn = "go on"
        
        armi  = lambda alpha : (J(xk + alpha*dk)) <=\
                (J(xk) + c * alpha * np.dot(djk.T, dk)) 
        curv  = lambda alpha : (np.linalg.norm(g_J(xk + alpha*dk))) <=\
                (0.9*np.linalg.norm(djk,2))  
        
        cpt, cptmax = 0, 15
        while (armi(alpha) == False) and cpt< cptmax and self.warn=="go on" :
            alpha_lo =  alpha
            alpha   *=  rho
            cpt     +=  1
            print (alpha,  armi(alpha))
            alpha_hi =  alpha            
        print("alpha = {}\t cpt = {}".format(alpha, cpt))
        print("Armijo = {}\t Curvature = {}".format(armi(alpha), curv(alpha)))
        
        bool_curv = curv(alpha)
        it = 0
        
        print ("alpha_l = {}\t alpha hi = {}".format(alpha_lo, alpha_hi))
        
        if alpha <= 1.e-14 :
            self.warn = "out"
            
        if cpt > 0 and bool_curv == False:
            alpha_2 = alpha_lo
            
            bool_curv = curv(alpha_2)
            while bool_curv == False and (alpha_2 - alpha_hi)>0 :
                alpha_2 *= 0.9        
                it  +=  1
                bool_curv = curv(alpha_2)
#                if it > 50 :
#                    alpha_2 = alpha
#                    break
                        
            if bool_curv == True :
                alpha = alpha_2
                correction = True
                print ("\x1b[1;37;43malpha_2 = {}\t alpha = {}, iteration = {}\x1b[0m".format(alpha_2, alpha, it))
                                   
        if bool_curv == False and armi(alpha) == False :
            alpha = max(alpha, 1e-11)
        
        if armi(alpha) == True and curv(alpha) == True :
            print("it = {}".format(it))
            print("\x1b[1;37;43mArmijo = {}\t Curvature = {}\x1b[0m".format(armi(alpha), curv(alpha)))
            
        return alpha, correction
##----------------------------------------------------##   
#    def goldstein(self, )   
##----------------------------------------------------## 
    def wolf_conditions(self, f, df, xk, dk, it, strong=True, verbose = False,c1 = 1e-4, c2 = 0.9 , alpha=1.):
        fk  =   f(xk)
        dk_c=   dk
        
        if (np.dot(dk, df(xk)) < 0) == False :
            warnings.warn("La direction doit etre negative \"Gradient descent\"\nErreur apparue cpt = %d"%(it))
        
        dfk =   df(xk)
        dfk = np.asarray(dfk).reshape(1,-1)[0]
        
        dfpk=   lambda a_n : df(xk + a_n*dk)
        
        #Armijo condition
        t1  =   lambda a_n : (f(xk + a_n * dk_c)) <= (fk + c1 * a_n * np.dot(dfk, dk_c)) 
        
        #Curvature condition    
        if strong == True :
            t2 = lambda a_n : (np.linalg.norm(np.dot(np.asarray(dfpk(a_n)).reshape(1,-1), dk_c))) <= \
                                (c2 * np.linalg.norm(np.dot(dfk, dk_c)))  
        else :
            t2 = lambda a_n : (np.dot(np.asarray(dfpk(a_n)).reshape(1,-1), dk_c)) >= (c2 * np.dot(dfk, dk_c))
        
        cpt = 0
#        if it < 10 :
#            cptmax = 150
#        if (it >=10 and it<200) :
#            cptmax = 80
#        if (it >=200 and it<1000)  :
#            cptmax = 50 
        cptmax = 100
        if strong== True :  
            while (t1(alpha) == False or t2(alpha) == False ) and cpt < cptmax :
                cpt +=  1
                if cpt % 10 ==0 and verbose == True:
                    print("cpt : {}\nt1 = {} \t t2 = {}".format(cpt, t1(alpha), t2(alpha)))
                alpha *= 0.85
        
        else :
            while (t1(alpha) == False or t2(alpha) == False) and cpt < cptmax :
                cpt +=  1
                if cpt % 10 ==0 and verbose== True :
                    print("\x1b[1;37;43mcpt : {}\nt1 = {} \t t2 = {}\x1b[0m".format(cpt, t1(alpha), t2(alpha)))
                alpha *= 0.85
        if verbose == True : print  ("t1 = {} \t t2 = {}".format(t1(alpha), t2(alpha)))
        return alpha
##----------------------------------------------------##   
##----------------------------------------------------##              
######                                            ######
######     Fonctions auxillaires de la classe     ######
######                                            ######
##----------------------------------------------------##
##----------------------------------------------------##
    def DR_DT(self, beta, T_inf) :
        M1 = np.diag([(self.N_discr-1)**2 for i in range(self.N_discr-3)], -1) # Extra inférieure
        P1 = np.diag([(self.N_discr-1)**2 for i in range(self.N_discr-3)], +1)  # Extra supérieure
        A_diag1 = -4* np.diag((self.h_beta(beta, T_inf)**3 * beta * self.eps_0))-np.diag([2*(self.N_discr-1)**2 for i in range(self.N_discr-2) ]) # Diagonale
        result = A_diag1 + M1 + P1
#        print ("Temperature in function DR_DT = \ {}".format(self.h_beta(beta, T_inf)))
#        print("DR_DT =\n {} ".format(result))
        return  result
##----------------------------------------------------##
    def DJ_DT(self, beta, T_inf) :
        sT_inf = "T_inf_%d" %(T_inf)
        curr_d = self.T_obs_mean[sT_inf]
        cov_m = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]        
#        print(" Cov = \n {} ".format(cov_m))        
#        return (self.h_beta(beta, T_inf) - curr_d) / (cov_m[0,0])
        result = np.dot(np.linalg.inv(cov_m), self.h_beta(beta, T_inf) - curr_d) 
#        print("inv_cov \ {} ".format(np.linalg.inv(cov_m)))
#        print("DJ_DT =\n {} ".format(result))
#        return ( np.dot(np.linalg.inv(cov_m),self.h_beta(beta, T_inf) - curr_d)  )      
        return result     
##----------------------------------------------------##
    def DJ_DBETA(self, beta, T_inf):
        sT_inf = "T_inf_%d" %(T_inf)
        cov_prior   =   self.cov_pri_dict[sT_inf]
        result = np.dot( np.linalg.inv(cov_prior), beta - self.beta_prior )
        return result
    
##----------------------------------------------------##
    def DR_DBETA(self, beta, T_inf):
#        print ("DR_DBETA in function -- h_beta**4 =\n {} \n T_inf**4 = {}".format((T_inf**4 - self.h_beta(beta, T_inf)**4) * T.eps_0, T_inf**4))
        return (T_inf**4 - self.h_beta(beta, T_inf)**4) * self.eps_0
##----------------------------------------------------##
    def PSI(self,beta, T_inf) :
#        result = -np.dot(np.linalg.inv(self.DR_DT(beta, T_inf)), self.DJ_DT(beta, T_inf))
        result = -np.dot(self.DJ_DT(beta, T_inf) , np.linalg.inv(self.DR_DT(beta, T_inf)).T )
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
## Autres fonctions en dehors de la classe ##
def subplot(T, method='adj_bfgs', save = False) : 
    """
    Fonction pour comparer les beta. 
    Afficher les max et min des différents tirages 
    Comparer l'approximation de la température avec beta_final et la température exacte
    """
    if method == "sr1"    :
        dico_beta_map   =   T.sr1_bmap
        dico_beta_fin   =   T.sr1_bf
        
        mins    =   T.sr1_mins
        maxs    =   T.sr1_maxs
        
        titles = ["SR1: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "SR1: Temperature fields"]
    if method in {"optimization", "Optimization", "opti"}:
        dico_beta_map   =   T.betamap
        dico_beta_fin   =   T.beta_final

        mins    =   T.mins_lst
        maxs    =   T.maxs_lst
        titles = ["Opti: Beta comparaison (bp = {},  cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]

    if method in {"",  "adjoint" }:
        dico_beta_map   =   T.adj_bmap
        dico_beta_fin   =   T.adj_bf

        mins    =   T.adj_mins
        maxs    =   T.adj_maxs
        
        titles = ["Adjoint: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]
    
    if method=="adj_bfgs":
        dico_beta_map   =   T.bfgs_adj_bmap
        dico_beta_fin   =   T.bfgs_adj_bf

        mins    =   T.bfgs_adj_mins
        maxs    =   T.bfgs_adj_maxs
        
        titles = ["BFGS_ADJ: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]
    
    for T_inf in T.T_inf_lst :
        sT_inf = "T_inf_" + str(T_inf)
        curr_d = T.T_obs_mean[sT_inf]
        fig, axes = plt.subplots(1,2,figsize=(20,10))
        colors = 'green', 'orange'
                
        axes[0].plot(T.line_z, dico_beta_fin[sT_inf], label = "Beta for {}".format(sT_inf), 
            marker = 'o', linestyle = 'None', color = colors[0])
            
        axes[0].plot(T.line_z, dico_beta_map[sT_inf], label = 'Betamap for {}'.format(sT_inf),      
             marker = 'o', linestyle = 'None', color = colors[1])
        
        axes[0].plot(T.line_z, T.true_beta(curr_d, T_inf), label = "True beta profil {}".format(sT_inf))
        
        axes[1].plot(T.line_z, T.h_beta(dico_beta_fin[sT_inf], T_inf), 
            label = "h_beta {}".format(sT_inf), marker = 'o',\
                                                    linestyle = 'None', color = colors[0])
        axes[1].plot(T.line_z, T.h_beta(dico_beta_map[sT_inf], T_inf), 
            label = "h_betamap {}".format(sT_inf), marker = 'o',\
                                                    linestyle = 'None', color = colors[1])
        axes[1].plot(T.line_z, curr_d, label= "curr_d {}".format(sT_inf))
        
        axes[0].plot(T.line_z, mins, label='Valeurs minimums', marker='s',\
                                                    linestyle='none', color='magenta')
        axes[0].plot(T.line_z, maxs, label='Valeurs maximums', marker='s',\
                                                    linestyle='none', color='black')

        axes[0].fill_between(T.line_z, mins, maxs, facecolor= "0.2", alpha=0.4, interpolate=True)                
#            for m,M in zip(self.mins_list, self.maxs_list) :
#                axes[0].axvspan(m, M, facecolor="0.2", alpha = 0.5 )
        
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])

        axes[0].legend(loc='best', fontsize = 10, ncol=2)
        axes[1].legend(loc='best', fontsize = 10, ncol=2)
        
        plt.show()
        
        if T.bool_method["adj_bfgs"] == True and T.bool_method["opti_scipy"] == True :
            bmaps=  [T.betamap[sT_inf], T.bfgs_adj_bmap[sT_inf]]
            mins =  [T.mins_lst, T.bfgs_adj_mins]
            maxs =  [T.maxs_lst, T.bfgs_adj_maxs]
            keys =  ["Scipy - optimization", "BFGS Adjoint Opti"]
                     
            comparaison(T, bmaps, mins, maxs, keys, T_inf)
        
        if T.bool_method["adj_bfgs"] == True and T.bool_method["sr1"] == True :
            bmaps=  [T.bfgs_adj_bmap[sT_inf], T.sr1_bmap[sT_inf]]
            mins =  [T.bfgs_adj_mins, T.sr1_mins]
            maxs =  [T.bfgs_adj_maxs, T.sr1_maxs]
            keys =  ["BFGS Adjoint Opti", "SR1 Update"]
                     
            comparaison(T, bmaps, mins, maxs, keys, T_inf)
        
        if T.bool_method["opti_scipy"] == True and T.bool_method["sr1"] == True :
            bmaps=  [T.betamap[sT_inf], T.sr1_bmap[sT_inf]]
            mins =  [T.mins_lst, T.sr1_mins]
            maxs =  [T.maxs_lst, T.sr1_maxs]
            keys =  ["Scipy - optimization", "SR1 Update"]
            
            comparaison(T, bmaps, mins, maxs, keys, T_inf)
            
    return axes
##---------------------------------------------------##
def comparaison(T, betamaps, minslsts, maxslsts, keys, T_inf, save = False) :
    betamap1    =   betamaps[0]
    betamap2    =   betamaps[1]
    
    labelbmap1  =   keys[0] + " bmap" 
    labelbmap2  =   keys[1] + " bmap"
    
    mins_lst1   =   minslsts[0]
    mins_lst2   =   minslsts[1]

    maxs_lst1   =   maxslsts[0]
    maxs_lst2   =   maxslsts[1]
    
    labelmm1    =   keys[0] + " uncertainty" 
    labelmm2    =   keys[1] + " uncertainty"
    sT_inf      =   "T_inf_"+str(T_inf)
    curr_d = T.T_obs_mean[sT_inf]
    
    # Main plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], axisbg="#f5f5f5")
    ax.plot(T.line_z, betamap1, label=labelbmap1 + " for {}".format(sT_inf), linestyle='none', marker='o', color='magenta')
    ax.plot(T.line_z, betamap2, label=labelbmap2 + " for {}".format(sT_inf), linestyle='none', marker='+', color='yellow')
    ax.plot(T.line_z, T.true_beta(curr_d, T_inf), label = "True beta profil {}".format(sT_inf), color='orange')
    
    ax.fill_between(T.line_z, mins_lst1, maxs_lst1, facecolor= "1", alpha=0.4, interpolate=True, hatch='\\', color="cyan", label=labelmm1)
    ax.fill_between(T.line_z, mins_lst2, maxs_lst2, facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="black", label=labelmm2)
    
    plt.legend(loc='best')
#    import matplotlib as mpl
#    x0, x1 = 0.3, 0.7
#    dz = 1./(T.N_discr-1)
#    ind0, ind1 = int(x0/dz), int(x1/dz)
#    ## Les deux lignes pointillees
#    ax.axvline(x0, ymin = 1, ymax=1.6, color="black", linestyle=":")  
#    ax.axvline(x1, ymin = 1, ymax=1.6, color="black", linestyle=":")

#    #Ajout de la figure
#    ax1 = fig.add_axes([0.05, 0.05, 0.4, 0.32], axisbg='#f8f8f8') 
#    ax1.set_ylim(1.2,1.6)
#    #[beg_horz, beg_vertical, end_horiz, end_vertical]
#    ##
#    x = np.linspace(x0, x1, len(T.line_z[ind0:ind1]))
#    ax1.plot(x, betamap1[ind0:ind1], linestyle='none', marker='o', color='magenta')
#    ax1.plot(x, betamap1[ind0:ind1], linestyle='none', marker='+', color='yellow')
#    ax1.plot(x, T.true_beta(curr_d, T_inf)[ind0:ind1], color='orange')
#    
#    ax1.fill_between(x, mins_lst1[ind0:ind1], maxs_lst1[ind0:ind1], facecolor= "1", alpha=0.2, interpolate=True, hatch='\\', color="cyan", label=labelmm1)
#    ax1.fill_between(x, mins_lst2[ind0:ind1], maxs_lst2[ind0:ind1], facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="black", label=labelmm2)
#        
##---------------------------------------------------##
##---------------------------------------------------##
def sigma_plot(T, method='adj_bfgs', exp = [0.02], base = [0.8], save=False) :
    """
    Fonction pour comparer les sigma posterior
    """
    import os.path as os
    if method in {"optimization", "Optimization", "opti"}:
        sigma_post = T.sigma_post_dict
        title = "Optimization sigma posterior comparison "

    if method in {"",  "adjoint" }:
        sigma_post = T.adj_sigma_post
        
        title = "Adjoint (Steepest D) sigma posterior comparison "
    
    if method=="adj_bfgs":
        sigma_post = T.bfgs_adj_sigma_post
        
        title = "Adjoint (BFGS) sigma posterior comparison "        
    
    if method=="sr1" :
        sigma_post = T.sr1_sigma_post
        
        title = "SR1 (1-rank BFGS) sigma posterior comparison "        
            
    
    print ("Title = %s" %(title))
    title_to_save = os.join(os.abspath("./res_all_T_inf"),title.replace(" ", "_")[:-1]+".png")
    dual = True if T.bool_method["opti_scipy"] == True and T.bool_method["adj_bfgs"] == True\
                else False
        
    for i, t in enumerate(T.T_inf_lst) :
        sT_inf = "T_inf_"+str(t)
        
        base_sigma  =   np.asarray([base[i] for j in range(T.N_discr-2)])
        exp_sigma   =   np.asarray([exp[i] for j in range(T.N_discr-2)])
        
        plt.figure()
        plt.title(title + sT_inf)
        plt.semilogy(T.line_z, exp_sigma, label='Expected Sigma', marker = 's', linestyle='none')
        plt.semilogy(T.line_z, sigma_post[sT_inf], label="Sigma Posterior %s" %(title[:3]))
        plt.semilogy(T.line_z, base_sigma, label="Sigma for beta = beta_prior (base)")
        plt.legend(loc='best')
        if save == True :
            plt.savefig(title_to_save)##Pour ne pas prendre le dernier espace
        plt.show()
        
    if dual == True :
        title_dual = "Scipy_opti and Adj_bfgs sigma post comparison %s (%s)" %(sT_inf, T.cov_mod)
        title_to_save = os.join(os.split(title_to_save)[0],title_dual.replace(" ", "_")+".png")
        opti_sigma_post = T.sigma_post_dict[sT_inf]
        adj_bfgs_sigma_post = T.bfgs_adj_sigma_post[sT_inf]
        
        plt.figure()
        plt.title(title_dual)
        plt.semilogy(T.line_z, exp_sigma, label='Expected Sigma', marker = '^', linestyle='none')
        plt.semilogy(T.line_z, opti_sigma_post, label="Opti Sigma posterior")
        plt.semilogy(T.line_z, adj_bfgs_sigma_post, label="Adj_bfgs Sigma posterior")
        plt.semilogy(T.line_z, base_sigma, label="Sigma for beta = beta_prior (base)")
        plt.legend(loc='best')
        if save == True :
            plt.savefig(title_to_save)
        plt.show()
    
##---------------------------------------------------##
##---------------------------------------------------##     
##---------------------------------------------------##

##---------------------------------------------------##        
if __name__ == "__main__" :
    
    parser = parser()
    print (parser)
    
    temp = Temperature(parser)
    temp.obs_pri_model()
    temp.get_prior_statistics()
    
    temp.adjoint_bfgs(inter_plot=True)
    
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
