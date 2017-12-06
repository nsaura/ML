#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

## Import de la classe TF ##
nnc_folder = os.path.abspath(os.path.dirname("../TF/NN_class_try.py"))
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
    parser.add_argument('--QN_tolerance', '-QN_tol', action='store', type=float, default=1e-5, dest='QN_tol', 
                        help='Define the tolerance on the QN_BFGS error (hybrid). Default to %(default).8f \n' )
    
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
        datapath            =   os.path.abspath(parser.datapath)
        num_real,   tol     =   parser.num_real,parser.tol
        cov_mod,    QN_tol  =   parser.cov_mod, parser.QN_tol
        
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
        
        ## Matrice pour la résolution
        M1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) # Extra inférieure
        P1 = np.diag(np.transpose([-dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)  # Extra supérieure
        A_diag1 = np.diag(np.transpose([(1 + dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

        self.A1 = A_diag1 + M1 + P1 #Construction de la matrice des coefficients
        
        M2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) # Extra inférieure
        P2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)  # Extra supérieure
        A_diag2 = np.diag(np.transpose([(1 - dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

        self.A2 = A_diag2 + M2 + P2 #Construction de la matrice des coefficients
        
        self.noise = self.tab_normal(0, 0.1, N_discr-2)[0]
        self.lst_gauss = [self.tab_normal(0,0.1,N_discr-2)[0] for i in range(num_real)]
        
        self.prior_sigma = dict()
        prior_sigma_lst = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]
        
        for i, t in enumerate([i*5 for i in range(1, 11)]) :
            self.prior_sigma["T_inf_%d" %(t)] = prior_sigma_lst[i]
        
        self.line_z  = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]
        self.num_real = num_real
        self.eps_0 = 5.*10**(-4)
        self.cov_mod = cov_mod
        self.N_discr = N_discr
        self.QN_tol = QN_tol  
        self.tol = tol
        self.dt = dt        
        self.h = h
        
        self.QN_done, self.optimize, self._adj =    False,  False,  False
        
        if os.path.exists(datapath) == False :
            os.mkdir(datapath)
        
        self.datapath = datapath
        self.stat_done = False
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
        
        print("T_inf_lst is now \n {}".format(self.T_inf_lst))
##---------------------------------------------------
    def set_beta_prior(self, new_beta) :
        """
        Descr :
        ----------
        Method designed to change the beta_prior array without running back the whole program.
        """
        self.beta_prior = np.asarray([new_beta for i in range(self.N_discr-2)])
        print("Beta prior is now \n {}".format(self.beta_prior))
##---------------------------------------------------
    def pd_read_csv(self, filename) :
        if os.path.splitext(filename)[-1] is not ".csv" :
            filename = os.path.splitext(filename)[0] + ".csv"
        path = os.path.join(self.datapath, filename)
        data = pd.read_csv(path).get_values()            
        return data.reshape(data.shape[0])
##---------------------------------------------------
    def pd_write_csv(self, filename, data) :
        path = os.path.join(self.datapath, filename)
        pd.DataFrame(data).to_csv(path, index=False, header= True)
##---------------------------------------------------
    def tab_normal(self, mu, sigma, length) :
        return ( sigma * np.random.randn(length) + mu, 
                (sigma * np.random.randn(length) + mu).mean(), 
                (sigma * np.random.randn(length) + mu).std()
               ) 
##---------------------------------------------------   
    def h_beta(self, beta, T_inf, verbose=False) :
        
        T_n = map(lambda x : -4*T_inf*x*(x-1), self.line_z)
        B_n = np.zeros((self.N_discr-2))
        T_nNext = T_n
        
        err, tol, compteur, compteur_max = 1., 1e-4, 0, 1500
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
                    print "i = ", i
                    print "B_n = ", B_n
                    print "T_n = ", T_n
                    print "T_N_tmp = ", T_n_tmp
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
                T_n_obs =  map(lambda x : -4*T_inf*x*(x-1), self.line_z) 
                T_n_pri =  map(lambda x : -4*T_inf*x*(x-1), self.line_z) 
                
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
            
            condi['diag' + sT_inf] = np.linalg.norm(cov_obs_dict[sT_inf])*np.linalg.norm(np.linalg.inv(cov_obs_dict[sT_inf]))
            
            self.J_los[sT_inf]      =   lambda beta : 0.5 * np.sum( [((self.h_beta(beta, T_inf)[i] - T_obs_mean[sT_inf][i]))**2 for i in range(self.N_discr-2)] ) /cov_obs_dict[sT_inf][0,0]
            print cov_pri_dict[sT_inf][0,0]
            
            
            
            self.cov_obs_dict   =   cov_obs_dict
            self.cov_pri_dict   =   cov_pri_dict
            self.T_obs_mean     =   T_obs_mean
            
            self.vals_obs_meshpoints    =   vals_obs_meshpoints
            self.full_cov_obs_dict      =   full_cov_obs_dict
            
            self.stat_done = True
##---------------------------------------------------  
    def optimization(self, verbose=False) :
        if self.stat_done == False : self.get_prior_statistics()
        
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
                print ("shapes to debug :")
                print("shape of cov_m = {}".format(cov_m.shape) )
                print("shape of cov_p = {}".format(cov_prior.shape) )
                print("shape of curr_d = {}".format(curr_d.shape) )
                
            J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                    np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
                    np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) )   
                                        ) ## Fonction de coût
                                        
            ## BFGS : quasi Newton method that approximate the Hessian on the fly

#            print "J =" , self.J_los[sT_inf](self.beta_prior)
            print ("J =" , J(self.beta_prior))
            opti_obj = op.minimize(J, self.beta_prior, method="BFGS", tol=self.tol, options={"disp" : True})

            betamap[sT_inf] =   opti_obj.x
            hess[sT_inf]    =   opti_obj.hess_inv
            cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf])
            
            self.opti_obj   =   opti_obj
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
            
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins_lst   =   mins_lst
        self.maxs_lst   =   maxs_lst
        self.beta_var   =   beta_var
        self.optimize   =   True
        
        self.sigma_post_dict = sigma_post_dict
##--------------------------------------------------- 
    def backline_search(self, J_lambda, var_J_lambda, direction, incr=0.5) :
        t = 1.
        norm_direction = np.linalg.norm(direction, 2)**2
        while J_lambda(var_J_lambda - t*direction) > (J_lambda(var_J_lambda) - t/2.0 * norm_direction) :
            t *= incr
        return t
##--------------------------------------------------- 
    def Next_hess(self, prev_hess, y_nN, s_nN ) :
        rho_nN  =   np.dot(y_nN.T, s_nN)
        n       =   prev_hess.shape[0]
        Id      =   np.diag([1 for i in range(n)])
        
        H_nN    =   np.dot( np.dot(Id - np.dot(np.dot(rho_nN, y_nN), s_nN.T) ,prev_hess ), Id - np.dot(np.dot(rho_nN, s_nN), y_nN.T) ) + \
                        np.dot(np.dot(rho_nN, s_nN), s_nN)
        return H_nN
##--------------------------------------------------- 
    def minimization_with_first_guess(self) :
        if self.stat_done == False : self.get_prior_statistics()

        QN_BFGS_bmap,   QN_BFGS_bf        =   dict(),     dict()
        QN_BFGS_hess,   QN_BFGS_cholesky  =   dict(),     dict()

        mins,   maxs    =   dict(),     dict()
        
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        beta_QNBFGS_real =  []
        alpha_lst       =   []
        direction_lst   =   []
        
        for T_inf in self.T_inf_lst :
            sT_inf  =   "T_inf_" + str(T_inf)
            curr_d  =   self.T_obs_mean[sT_inf]
            cov_m   =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf] 
            cov_prior =  self.cov_pri_dict[sT_inf]
            
            J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                    np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
                    np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) ) 
                                    )
                                    
#            first_guess_opti =  op.minimize(self.J_los[sT_inf], self.beta_prior, method="BFGS", tol=0.1, options={"disp" : True, "maxiter" : 10}) ## Simplement pour avoir Hess_0 
            first_guess_opti =  op.minimize(J, self.beta_prior, method="BFGS", tol=0.01, options={"disp" : True, "maxiter" : 10}) ## Simplement pour avoir Hess_0 
            for item in first_guess_opti.iteritems() : print (item)
            
            beta_n  =   first_guess_opti.x
            H_n     =   first_guess_opti.hess_inv
            g_n     =   first_guess_opti.jac
            
            self.first_guess_opti   =   first_guess_opti
            
            err_hess,   err =   0., 1.0
            cpt,        cpt_max     =   0,  100
            while (np.abs(err) > self.QN_tol) and (cpt<cpt_max) :
                ## Incr
                if cpt > 0 :
                    H_n     =   H_nNext
                    g_n     =   g_nNext 
                    beta_n  =   beta_nNext      
                
                cpt += 1    
                direction   =   np.dot(H_n, g_n) ; print("direction:\n", direction)

                alpha       =   self.backline_search(J, beta_n, direction)
                print ("alpha = ", alpha)

                beta_nNext  =   beta_n - alpha*direction

                alpha_lst.append(alpha)
                direction_lst.append(direction)
                
                ## Estimation of H_nNext
                g_nNext =   nd.Gradient(J)(beta_nNext) # From numdifftools
                print ("compteur = {}, Gradient dans la boucle minimization: \n {}".format(cpt, g_nNext))
                y_nNext =   g_nNext - g_n
                s_nNext =   beta_nNext - beta_n
                H_nNext =   self.Next_hess(H_n, y_nNext, s_nNext)
                
                err     =   (J(beta_nNext) - J(beta_n)) 
                
                err_hess=  np.linalg.norm(H_nNext - H_n)
                
                print ("cpt = {} \t err = {:.5}".format(cpt, err))

            print ("Compteur = %d \t err_j = %.12f \t err_Hess %.12f" % (cpt, err, err_hess))
            print ("beta_nNext = ", beta_nNext)
            print ("J(beta_nNext) = ", J(beta_nNext) )
            
            QN_BFGS_bmap[sT_inf]    =   beta_nNext
            QN_BFGS_hess[sT_inf]    =   H_nNext
            
            QN_BFGS_cholesky[sT_inf]=   np.linalg.cholesky(H_nNext)
            
            QN_BFGS_bf[sT_inf]      =   QN_BFGS_bmap[sT_inf] + np.dot(np.linalg.cholesky(H_nNext), s)
            
            for i in range(100):
                s   =   np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_QNBFGS_real.append(QN_BFGS_bmap[sT_inf] + np.dot(QN_BFGS_cholesky[sT_inf], s))
            
            beta_QNBFGS_real.append(QN_BFGS_bf[sT_inf])
            
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_QNBFGS_real]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_QNBFGS_real]))
            
            mins_lst    =   [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
            maxs_lst    =   [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
            
        self.QN_BFGS_bmap   =   QN_BFGS_bmap
        self.QN_BFGS_bf     =   QN_BFGS_bf
        self.QN_BFGS_hess   =   QN_BFGS_hess
        
        self.QN_BFGS_cholesky   =   QN_BFGS_cholesky
        self.QN_BFGS_mins_lst   =   mins_lst
        self.QN_BFGS_maxs_lst   =   maxs_lst
        
        self.alpha_lst      =   np.asarray(alpha_lst)
        self.direction_lst  =   np.asarray(direction_lst)
        
        self.QN_done = True
##---------------------------------------------------    
    def DR_DT(self, beta, T_inf) :
        return  -4* np.diag((self.h_beta(beta, T_inf)**3 * beta * T.eps_0))
##---------------------------------------------------
    def DJ_DT(self, beta, T_inf) :
        sT_inf = "T_inf_%d" %(T_inf)
        curr_d = self.T_obs_mean[sT_inf]
        cov_m = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]        
        return (self.h_beta(beta, T_inf) - curr_d) / (cov_m[0,0])
##---------------------------------------------------
    def DR_DBETA(self, beta, T_inf):
        print ("DR_DBETA -- h_beta**4 =\n {} \n T_inf**4 = {}".format(self.h_beta(beta, T_inf)**4, T_inf**4))
        return (T_inf**4 - self.h_beta(beta, T_inf)**4) * T.eps_0
##---------------------------------------------------
    def PSI(self,beta, T_inf) :
        return -np.dot(np.linalg.inv(self.DR_DT(beta, T_inf)), self.DJ_DT(beta, T_inf))
                
##---------------------------------------------------
    def _for_adjoint(self, T_inf) : 
        if self.stat_done == False : self.get_prior_statistics()
        
        sT_inf      =   "T_inf_%d" %(T_inf)
        curr_d      =   self.T_obs_mean[sT_inf]
        cov_m       =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]
        cov_prior   =   self.cov_pri_dict[sT_inf]
        
        J = lambda beta : 0.5*  ( 
                      np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                        np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                    + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
                        np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) ) 
                                 )
        
        alpha_lst,  direction_lst   =   [],     [] 
        BFGS_bmap,   BFGS_bf        =   dict(), dict()
        BFGS_hess,   BFGS_cholesky  =   dict(), dict()

        mins,   maxs    =   dict(),     dict()
        
#        curr_h_beta =   self.h_beta(self.beta_prior, T_inf)

#        dR_dT = lambda beta : np.asarray( [ - 4 * (self.h_beta(beta, T_inf)[i]**3) * beta[i] * T.eps_0 for i in range(self.N_discr - 2)])
#        dJ_dT = lambda beta : np.asarray([(self.h_beta(beta, T_inf)[i] - curr_d[i]) / cov_prior[0,0] for i in range(self.N_discr - 2) ] )
#        dR_dBeta = lambda beta : \
#                    np.asarray( [  (T_inf**4 - self.h_beta(beta, T_inf)[i]**4) * T.eps_0 for i in range(self.N_discr - 2) ] )
#                        
#        psi = lambda beta : np.asarray(- self.DJ_DT(beta, T_inf) / DR_DT(beta, T_inf)) #
        psi = self.PSI(self.beta_prior, T_inf)
        beta_nPrev = np.zeros_like(self.beta_prior) 
        beta_n = self.beta_prior
        
        g_nPrev = np.zeros((self.N_discr-2, 1))

        cpt,    err,    cptMax, tol =   0,  1., 5000,  1e-7
        print("beta_nPrev: \n {}".format(beta_n))
        
        plt.figure()
        print("psi(beta_prior) = {}) \n".format(psi))
        print("dR/dBeta(T.beta_prior) = {} \n".format(self.DR_DBETA(self.beta_prior, T_inf)))
        while np.abs(err) > tol and (cpt<cptMax) :
        
            self.g_n     =   -np.dot(self.PSI(beta_n, T_inf).T, np.diag(self.DR_DBETA(beta_n,T_inf)) )   #dJ/dBeta
#            self.g_n     =   nd.Gradient(J)(beta_n) + np.dot(self.PSI(beta_n, T_inf).T, np.diag(self.DR_DBETA(beta_n,T_inf)) )   dJ/dBeta            
#            self.g_n     =   nd.Gradient(self.J_los[sT_inf])(beta_n) + np.dot(self.PSI(beta_n, T_inf).T, np.diag(self.DR_DBETA(beta_n,T_inf)) )   ##dJ/dBeta                        
            self.N_n_nP  =   np.linalg.norm(self.g_n - g_nPrev, 2)
            self.gamma_n =   np.dot( (beta_n-beta_nPrev).T ,(self.g_n - g_nPrev) ) / self.N_n_nP
            plt.plot(self.line_z, beta_n, label="beta_n compteur %d" %(cpt))
            
            beta_nNext = beta_n - self.gamma_n* self.g_n
            err = np.abs(self.J_los[sT_inf](beta_nNext) - self.J_los[sT_inf](beta_n))
#            err = np.abs(J(beta_nNext) - J(beta_n))
            g_nPrev     =   self.g_n         # n-1   --> n
            beta_nPrev  =   beta_n      # n-1   --> n
            beta_n      =   beta_nNext  # n     --> n+1  
            
            print("cpt = {} \t err = {}".format(cpt, err))
            print("beta_n: \n {}".format(beta_n))
            print ("grad n : -np.dot(psi(beta_n).T, np.diag(dR_dBeta(beta_n)) ) = {}".format(self.g_n))
            print ("Norme grad_n - grad_nPrev = {} \n gamma_n = {}".format(self.N_n_nP, self.gamma_n))
            print("\n")
            print("psi(beta_nNext = {} \n".format(self.PSI(beta_nNext, T_inf)))
            print("dR/dBeta(beta_nNext) = {}".format(self.DR_DBETA(beta_nNext, T_inf)))
            cpt +=1
        plt.legend(loc="best")
            
            
            # n --> n+1 si non convergence, sort de la boucle sinon 
        
        #self.Hess = np.dot(g_n.T, g_n)
        self.sd_beta = beta_n
        self.sd_grad = self.g_n
###---------------------------------------------------
def subplot(T, method='QN_BFGS') : 
    if method == "QN_BFGS"    :
        dico_beta_map   =   T.QN_BFGS_bmap
        dico_beta_fin   =   T.QN_BFGS_bf
        
        mins    =   T.QN_BFGS_mins_lst
        maxs    =   T.QN_BFGS_maxs_lst
        
        titles = ["QNBFGS: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "QNBFGS: Temperature fields"]
    else :
        dico_beta_map   =   T.betamap
        dico_beta_fin   =   T.beta_final

        mins    =   T.mins_lst
        maxs    =   T.maxs_lst
        
        titles = ["Beta comparaison (bp = {}, , cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]
        
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
            label = "h_beta {}".format(sT_inf), marker = 'o', linestyle = 'None', color = colors[0])
        axes[1].plot(T.line_z, T.h_beta(dico_beta_map[sT_inf], T_inf), 
            label = "h_betamap {}".format(sT_inf), marker = 'o', linestyle = 'None', color = colors[1])
        axes[1].plot(T.line_z, curr_d, label= "curr_d {}".format(sT_inf))
        
        axes[0].plot(T.line_z, mins, label='Valeurs minimums', marker='s', linestyle='none', color='magenta')
        axes[0].plot(T.line_z, maxs, label='Valeurs maximums', marker='s', linestyle='none', color='black')

        axes[0].fill_between(T.line_z, mins, maxs, facecolor= "0.2", alpha=0.4, interpolate=True)                
#            for m,M in zip(self.mins_list, self.maxs_list) :
#                axes[0].axvspan(m, M, facecolor="0.2", alpha = 0.5 )
        
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])

        axes[0].legend(loc='best', fontsize = 10, ncol=2)
        axes[1].legend(loc='best', fontsize = 10, ncol=2)
        
        plt.show()
    
    if T.QN_done == True and T.optimize == True :
        # Main plot
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], axisbg="#f5f5f5")
        ax.plot(T.line_z, T.betamap[sT_inf], label="Optimization Betamap for {}".format(sT_inf), linestyle='none', marker='o', color='magenta')
        ax.plot(T.line_z, T.QN_BFGS_bmap[sT_inf], label="QN_BFGS Betamap for {}".format(sT_inf), linestyle='none', marker='+', color='yellow')
        ax.plot(T.line_z, T.true_beta(curr_d, T_inf), label = "True beta profil {}".format(sT_inf), color='orange')
        
        ax.fill_between(T.line_z, T.QN_BFGS_mins_lst, T.QN_BFGS_maxs_lst, facecolor= "1", alpha=0.4, interpolate=True, hatch='\\', color="cyan", label="QN_BFGS uncertainty")
        ax.fill_between(T.line_z,  T.mins_lst, T.maxs_lst, facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="black", label="Optimization uncertainty")
        
        import matplotlib as mpl
        x0, x1 = 0.3, 0.7
        dz = 1./(N_discr-1)
        ind0, ind1 = int(x0/dz), int(x1/dz)
        ## Les deux lignes pointillees
        ax.axvline(x0, ymin = 1, ymax=1.6, color="black", linestyle=":")  
        ax.axvline(x1, ymin = 1, ymax=1.6, color="black", linestyle=":")

        #Ajout de la figure
        ax1 = fig.add_axes([0.05, 0.05, 0.4, 0.32], axisbg='#f8f8f8') 
        ax1.set_ylim(1.2,1.6)
        #[beg_horz, beg_vertical, end_horiz, end_vertical]
        ##
        x = np.linspace(x0, x1, len(T.line_z[ind0:ind1]))
        ax1.plot(x, T.betamap[sT_inf][ind0:ind1], linestyle='none', marker='o', color='magenta')
        ax1.plot(x, T.QN_BFGS_bmap[sT_inf][ind0:ind1], linestyle='none', marker='+', color='yellow')
        ax1.plot(x, T.true_beta(curr_d, T_inf)[ind0:ind1], color='orange')
        
#        ax1.fill_between(x, T.QN_BFGS_mins_lst[ind0:ind1], T.QN_BFGS_maxs_lst[ind0:ind1], facecolor= "1", alpha=0.2, interpolate=True, hatch='\\', color="cyan")
        ax1.fill_between(x,  T.mins_lst[ind0:ind1], T.maxs_lst[ind0:ind1], facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="black")
        
        ax.set_title("Beta comparison between Optimisation and QN_BFGS (hybrid) method")
        ax.set_xlabel("z")
        ax.set_ylabel("beta\'s")
        ax.legend(loc="best", fontsize = 13, ncol=2)
        
        return axes
        
##---------------------------------------------------##
##---------------------------------------------------##
##---------------------------------------------------##        
if __name__ == "__main__" :
#    __init__ (self, T_inf_lst, N_discr, dt, h, kappa, datapath, num_real = )
    parser = parser()
    print (parser)
    T = Temperature(parser)
    T.obs_pri_model()
    T.get_prior_statistics()
#    T.optimization(verbose=True)

## run class_temp_ML.py -T_inf_lst 50 -kappa 1 -tol 1e-5 -beta_prior 1. -num_real 100 -cov_mod 'diag' -N 50 -dt 1e-4

#plt.figure()
#plt.semilogy(T.line_z, [0.02 for i in range(T.N_discr-2)], label='true', marker = 's' linestyle='none')
#plt.semilogy(T.line_z, [0.02 for i in range(T.N_discr-2)], label='true', marker = 's', linestyle='none')
#plt.semilogy(T.line_z, T.sigma_post_dict["T_inf_50"], label="Post")
#plt.semilogy(T.line_z, [0.8 for i in range(T.N_discr-2)], label="base")
#plt.legend()

#run class_temp_ML.py -T_inf_lst 50 -kappa 10 -tol 1e-5 -beta_prior 1.5 -num_real 100 -cov_mod 'full' -N 50 -dt 1e-4
#run class_temp_ML.py -T_inf_lst 50 -kappa 10 -tol 1e-5 -beta_prior 1.3 -num_real 100 -cov_mod 'full' -N 50 -dt 1e-4
#run class_temp_ML.py -T_inf_lst 50 -kappa 1 -tol 1e-5 -beta_prior 1.7 -num_real 100 -cov_mod 'full' -N 50 -dt 1e-4


