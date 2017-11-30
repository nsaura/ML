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
        np.random.seed(1000) ; plt.ion()
        
        if parser.cov_mod not in ['full', 'diag'] :
            raise AttributeError("\x1b[7;1;255mcov_mod must be either diag or full\x1b[0m")
        
        T_inf_lst = parser.T_inf_lst

        N_discr,    kappa   =   parser.N_discr, parser.kappa,    
        dt,         h       =   parser.dt,      parser.h
        datapath            =   os.path.abspath(parser.datapath)
        num_real,   tol     =   parser.num_real,parser.tol
        cov_mod             =   parser.cov_mod
        
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
        A_diag1 = np.diag(np.transpose([(1+ dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

        self.A1 = A_diag1 + M1 + P1 #Construction de la matrice des coefficients
        
        
        M2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), -1) # Extra inférieure
        P2 = np.diag(np.transpose([dt/dz**2*kappa/2 for i in range(N_discr-3)]), 1)  # Extra supérieure
        A_diag2 = np.diag(np.transpose([(1-dt/dz**2*kappa) for i in range(N_discr-2)])) # Diagonale

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
        self.tol = tol
        self.dt = dt        
        self.h = h
        
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
        
        print ("T_inf_lst is now {}".format(self.T_inf_lst))
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
    def h_beta(self, beta, T_inf, verbose=False,noise= 'none') :
        
        T_n = map(lambda x : -4*T_inf*x*(x-1), self.line_z)
        B_n = np.zeros((self.N_discr-2))
        T_nNext = T_n
        
        err, tol, compteur, compteur_max = 1., 1e-4, 0, 5000
        if verbose == True :
            plt.figure()
            
        while (np.abs(err) > tol) and (compteur <= compteur_max) :
            if compteur > 0 :
                T_n = T_nNext
            compteur +=1 
            
            T_n_tmp = np.dot(self.A2, T_n)
            
            for i in range(self.N_discr-2) :
                B_n[i] = T_n_tmp[i] + self.dt*(beta[i])*self.eps_0*(T_inf**4 - T_n[i]**4)
            
            T_nNext = np.dot(np.linalg.inv(self.A1), B_n)
            err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
            
            if verbose == True and compteur % 5 == 0 :
                print err
                plt.plot(self.line_z, T_nNext, label='tracer cpt %d' %(compteur))
            
            if compteur == compteur_max :
                warnings.warn("\x1b[7;1;255mH_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {} \x1b[0m".format(err, tol))
        
        if verbose == True :
            plt.plot(self.line_z, T_nNext, marker="o", linestyle='none')
            plt.legend(loc="best", ncol=4)
        
        print ("Err = {} ".format(err))
        
        return T_nNext 
##---------------------------------------------------
    def true_beta(self, T, T_inf) : 
        return np.asarray (
        [ 1./self.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i]) + self.noise[i] ) *10**(-4) + self.h / self.eps_0*(T_inf - T[i])/(T_inf**4 - T[i]**4)  for i in range(self.N_discr-2)]        
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
                
                while (np.abs(err_obs) > tol) and (compteur <800) and (np.abs(err_pri) > tol):
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
    def get_prior_statistics(self):
        cov_obs_dict    =   dict() 
        cov_pri_dict    =   dict()
        
        mean_meshgrid_values=   dict()  
        full_cov_obs_dict   =   dict()        
        vals_obs_meshpoints =   dict()
        
        T_obs_mean  = dict()
        
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

            T_obs_mean[sT_inf] = T_sum # Joue aussi le rôle de moyenne pour la covariance
            
            Sum    =    np.zeros((self.N_discr-2, self.N_discr-2))   
            std_meshgrid_values     =   np.asarray([np.std(vals_obs_meshpoints[sT_inf+"_"+str(j)])  for j   in  range(self.N_discr-2)])
#            mean_meshgrid_values[sT_inf]    =   np.asarray([np.mean(vals_obs_meshpoints[sT_inf+"_"+str(j)]) for j   in  range(self.N_discr-2)])
            
#            print mean_meshgrid_values
            
            for it in range(self.num_real) :
                obs_filename  =  'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                T_temp = self.pd_read_csv(obs_filename)
                
                for ii in range(self.N_discr-2)  :
                    for jj in range(self.N_discr-2) : 
                        Sum[ii,jj] += (T_temp[ii] - T_obs_mean[sT_inf][ii]) * (T_temp[jj] - T_obs_mean[sT_inf][jj])/float(self.num_real)
            
            full_cov_obs_dict[sT_inf] = Sum            
            print ("cov_obs :\n{}".format(Sum))
            
            std_mean_prior          =   np.mean(np.asarray([np.std(T_prior[i]) for i in range(len(T_prior))]))
            cov_obs_dict[sT_inf]    =   np.diag([std_meshgrid_values[j]**2 for j in range(self.N_discr-2)])
            cov_pri_dict[sT_inf]    =   np.diag([self.prior_sigma[sT_inf]**2 for j in range(self.N_discr-2)])
            
#            # Construction of full_cov_obs_dict
#            full_cov = np.zeros((self.N_discr-2, self.N_discr-2))
#            
#            for ii in range(self.N_discr) :
#                for jj in range(self.N_discr) :
#                    full_cov[i,j] = np.mean((T_obs_mean[sT_inf][i] - mean_meshgrid_values[i])*(T_obs_mean[sT_inf][j] - mean_meshgrid_values[j]))
#            
#            full_cov_obs_dict[sT_inf] = full_cov
            #### Until Here

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
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        beta_var = []
        
        for T_inf in self.T_inf_lst :
            sT_inf  =   "T_inf_" + str(T_inf)
            curr_d  =   self.T_obs_mean[sT_inf]
            cov_prior   =   self.full_cov_obs_dict[sT_inf], self.cov_pri_dict[sT_inf]
            cov_m = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf]           
            J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                    np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
                    np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) )   
                            ) ## Fonction de coût
                            
            # BFGS : quasi Newton method that approximate the Hessian on the fly
            opti = op.minimize(J, self.beta_prior, method="BFGS", tol=self.tol)

            betamap[sT_inf] =   opti.x
            hess[sT_inf]    =   opti.hess_inv
            cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf])
            
            self.opti = opti
            print ("Sucess state of the optimization {}".format(self.opti.success))
            
            beta_final[sT_inf]  =   betamap[sT_inf] + np.dot(cholesky[sT_inf], s)  
            
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append(betamap[sT_inf] + np.dot(cholesky[sT_inf], s))
            beta_var.append(beta_final[sT_inf])
            
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
            
            mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
            maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
            
            if verbose == True :
                fig, axes = plt.subplots(1,2,figsize=(20,10))
                colors = 'green', 'orange'
                
                axes[0].plot(self.line_z, beta_final[sT_inf], label="Beta for {}".format(sT_inf), marker='o',
                                 linestyle='None', color=colors[0])
                axes[0].plot(self.line_z, betamap[sT_inf], label='Betamap for {}'.format(sT_inf),      
                             marker='o', linestyle='None', color=colors[1])
                axes[0].plot(self.line_z, self.true_beta(curr_d, T_inf), label = "True beta profil {}".format(sT_inf))
                axes[1].plot(self.line_z, self.h_beta(beta_final[sT_inf], T_inf), label= "h_beta {}".format 
                                (sT_inf), marker='o', linestyle='None', color=colors[0])
                axes[1].plot(self.line_z, self.h_beta(betamap[sT_inf], T_inf), label= "h_betamap {}".format
                                (sT_inf), marker='o', linestyle ='None', color=colors[1])
                axes[1].plot(self.line_z, curr_d, label= "curr_d {}".format(sT_inf))

                axes[0].plot(self.line_z, mins_lst, label='Valeurs minimums', marker='s', linestyle='none', color='magenta')
                axes[0].plot(self.line_z, maxs_lst, label='Valeurs maximums', marker='s', linestyle='none', color='black')
                
                axes[0].fill_between(T.line_z, T.mins, T.maxs, facecolor= "0.2", alpha=0.4, interpolate=True)
                
                axes[0].set_title("Optimized beta and Duraisamy beta; tolerance={}".format(self.tol))
                axes[1].set_title("Temperature field with optimized betas and true solution")

                axes[0].legend(loc='best', fontsize = 10, ncol=2)
                axes[1].legend(loc='best', fontsize = 10, ncol=2)
                
                plt.show()
                
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins_lst   =   mins_lst
        self.maxs_lst   =   maxs_lst
        self.beta_var   =   beta_var
        
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
        beta_QNBFGS_real = []
        alpha_lst = []
        direction_lst =[]
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
                                    
            first_guess_opti =  op.minimize(J, self.beta_prior, method="BFGS", tol=0.1) ## Simplement pour avoir Hess_0 
            for item in first_guess_opti.iteritems() : print (item)
            
            beta_n  =   first_guess_opti.x
            H_n     =   first_guess_opti.hess_inv
            g_n     =   first_guess_opti.jac
            
            self.first_guess_opti = first_guess_opti
#            beta_n =    self.beta_prior
#            H_n    =    np.eye(self.N_discr-2)
#            g_n    =    np.ones((self.N_discr-2,))
            
            err, tol = 1.0, 1e-7
            cpt, cpt_max    =   0,  5000
            err_hess = 0.
            while (np.abs(err) > tol) and (cpt<cpt_max) :
                ## Incr
                if cpt > 0 :
                    H_n     =   H_nNext
                    g_n     =   g_nNext 
                    beta_n  =   beta_nNext      
                
                cpt += 1    
                direction   =   np.dot(H_n, g_n)
                alpha       =   self.backline_search(J, beta_n, direction)
                print ("alpha = ", alpha)
#                alpha  = 1e-2
                beta_nNext  =   beta_n - alpha*direction
                #self.beta_nNext = beta_nNext
                alpha_lst.append(alpha)
                direction_lst.append(direction)
                
                ## Estimation of H_nNext
                g_nNext =   nd.Gradient(J)(beta_nNext) # From numdifftools
                print "compteur = {}, Gradient dans la boucle minimization \n {}:".format(cpt, g_nNext)
                y_nNext =   g_nNext - g_n
                s_nNext =   beta_nNext - beta_n
                H_nNext =   self.Next_hess(H_n, y_nNext, s_nNext)
                
                err = (J(beta_nNext) - J(beta_n)) 
                
                err_hess = np.linalg.norm(H_nNext - H_n)
                
                print ("cpt = {} \t err = {:.5}".format(cpt, err))

            print ("Compteur = %d \t err_j = %.12f \t err_Hess %.12f" % (cpt, err, err_hess))
            print ("beta_nNext = ", beta_nNext)
            print ("J(beta_nNext) = ", J(beta_nNext) )
            
            QN_BFGS_bmap[sT_inf]    =   beta_nNext
            QN_BFGS_hess[sT_inf]    =   H_nNext
            
            QN_BFGS_cholesky[sT_inf]=   np.linalg.cholesky(H_nNext)
            
            QN_BFGS_bf[sT_inf]      =   QN_BFGS_bmap[sT_inf] + np.dot(np.linalg.cholesky(H_nNext), s)
            
            for i in range(100):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_QNBFGS_real.append(QN_BFGS_bmap[sT_inf] + np.dot(QN_BFGS_cholesky[sT_inf], s))
            beta_QNBFGS_real.append(QN_BFGS_bf[sT_inf])
            
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_QNBFGS_real]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_QNBFGS_real]))
            
            mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
            maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
            
        self.QN_BFGS_bmap   =   QN_BFGS_bmap
        self.QN_BFGS_bf     =   QN_BFGS_bf
        self.QN_BFGS_hess   =   QN_BFGS_hess
        
        self.QN_BFGS_cholesky   =   QN_BFGS_cholesky
        self.QN_BFGS_mins_lst   =   mins_lst
        self.QN_BFGS_maxs_lst   =   maxs_lst
        
        self.alpha_lst      =   np.asarray(alpha_lst)
        self.direction_lst  =   np.asarray(direction_lst)
##---------------------------------------------------
    def _for_adjoint(self, T_inf=50) : 
        if self.stat_done == False : self.get_prior_statistics()
        
        sT_inf      =   "T_inf_%d" %(T_inf)
        curr_d      =   self.T_obs_mean[sT_inf]
        cov_m       =   self.cov_obs_dict[sT_inf]
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

        dR_dT = lambda beta : np.asarray( [ 4 * (self.h_beta(beta, T_inf)[i]**3) * beta[i] * T.eps_0 for i in range(self.N_discr - 2)])
        dJ_dT = lambda beta : np.asarray([(self.h_beta(beta, T_inf)[i] - curr_d[i]) / cov_prior[0,0] for i in range(self.N_discr - 2) ] )
        dR_dBeta = lambda beta : \
                    np.asarray( [  - (T_inf**4 - self.h_beta(beta, T_inf)[i]**4) * T.eps_0 for i in range(self.N_discr - 2) ] )
                        
        psi = lambda beta : np.asarray(- dJ_dT(beta) / dR_dT(beta)) #
        
        beta_nPrev = np.zeros_like(self.beta_prior) 
        beta_n = self.beta_prior
        
        g_nPrev = np.zeros((self.N_discr-2, self.N_discr-2))

        cpt,    err,    cptMax, tol =   0,  1., 5000,  1e-7
        print("beta_nPrev: \n {}".format(beta_n))
        
        while np.abs(err) > tol and (cpt<cptMax) :
        
            self.g_n     =   -np.dot(psi(beta_n).T, np.diag(dR_dBeta(beta_n)) )   #dJ/dBeta
            
            self.N_n_nP  =   np.linalg.norm(self.g_n - g_nPrev, 2)
            
            self.gamma_n =   np.dot( (beta_n-beta_nPrev).T ,(self.g_n - g_nPrev) ) / self.N_n_nP
            
            beta_nNext = beta_n - self.gamma_n* self.g_n
            
            err = np.abs(J(beta_nNext) - J(beta_n))
            
            g_nPrev     =   self.g_n         # n-1   --> n
            beta_nPrev  =   beta_n      # n-1   --> n
            beta_n      =   beta_nNext  # n     --> n+1  
            
            print("cpt = {} \t err = {}".format(cpt, err))
            print("beta_n: \n {}".format(beta_n))
            
            cpt +=1
            
            # n --> n+1 si non convergence, sort de la boucle sinon 
        
        #self.Hess = np.dot(g_n.T, g_n)
        self.sd_beta = beta_n
        self.sd_grad = self.g_n
##---------------------------------------------------
    def adjoint_optimization(self, T_inf) :
        if self.stat_done == False : self.get_prior_statistics()
        
        alpha_lst,  direction_lst   =   [],     [] 
        BFGS_bmap,   BFGS_bf        =   dict(), dict()
        BFGS_hess,   BFGS_cholesky  =   dict(), dict()

        mins,   maxs    =   dict(),     dict()
        
        sT_inf      =   "T_inf_%d" %(T_inf)
        curr_d      =   self.T_obs_mean[sT_inf]
        cov_m       =   self.cov_obs_dict[sT_inf]
        cov_prior   =   self.cov_pri_dict[sT_inf]
        
        J = lambda beta : 0.5*  ( 
                      np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                        np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                    + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
                        np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) ) 
                                )
        cpt,    err,    cptMax, tol =   0,  1., 5000,  1e-3
        dz  =   1. / (self.N_discr-1)
        
        s   =   np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        beta_n =   self.beta_prior
        
        g_n =   self._for_adjoint(beta_n, T_inf, curr_d, cov_prior)
#        H_n =   np.linalg.inv(np.abs(np.random.rand(31,31)))
        first_guess_opti =  op.minimize(J, self.beta_prior, method="BFGS", tol=0.1, options={"maxiter" :100} ) ## Simplement pour avoir Hess_0 
        
        print ("comparaison des gradients : \n Adjoint : g_n = {} \t \t op.minimize : = first_guess_opti.jac {}".format(g_n, first_guess_opti.jac))
        
        aaa = str(input("Any Key : \n")) 
        
        while (np.abs(err) > tol) and (cpt < cptMax) :
        # Intialisation : 
            if cpt > 0 :
                beta_n  =   beta_nNext 
                H_n     =   H_nNext
                g_n     =   g_nNext
            
            cpt += 1
            direction   =   np.dot(H_n, g_n)
            alpha       =   self.backline_search(J, beta_n, direction)
            
            alpha_lst.append(alpha)
            direction_lst.append(direction)
        #    print "alpha = ", alpha
            
            beta_nNext      =   beta_n - alpha*direction
            self.beta_nNext =   beta_nNext
            
            g_nNext = self._for_adjoint(beta_nNext, T_inf, curr_d, cov_prior)
            print "compteur = {}, \t Gradient dans la boucle minimization \n {}:".format(cpt, g_nNext)
            y_nNext =   g_nNext - g_n
            s_nNext =   beta_nNext - beta_n
            H_nNext =   self.Next_hess(H_n, y_nNext, s_nNext)
            
            err = (J(beta_nNext) - J(beta_n)) 
            err_hess = np.linalg.norm(H_nNext - H_n)
            
            print ("cpt = {} \t err = {:.5}".format(cpt, err))
            
        print ("Compteur     = %d \t err_j = %.12f \t err_Hess %.12f" % (cpt, err, err_hess))
        print ("Last Beta    = ", beta_nNext)
        print ("Last J(beta) = ", J(beta_nNext)) 
        
        BFGS_bmap[sT_inf]    =   beta_nNext
        BFGS_hess[sT_inf]    =   H_nNext
        
        BFGS_cholesky[sT_inf]=   np.linalg.cholesky(H_nNext)
        
        BFGS_bf[sT_inf]      =   BFGS_bmap[sT_inf] + np.dot(BFGS_cholesky[sT_inf], s)
        
        for i in range(100):
            s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
            beta_QNBFGS_real.append(QN_BFGS_bmap[sT_inf] + np.dot(QN_BFGS_cholesky[sT_inf], s))
        beta_QNBFGS_real.append(QN_BFGS_bf[sT_inf])
        
        for i in range(self.N_discr-2) :
            mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_QNBFGS_real]))
            maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_QNBFGS_real]))
        
        mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]   
        maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in range(self.N_discr-2)]
            
        BFGS_bmap   =   BFGS_bmap
        BFGS_bf     =   BFGS_bf
        BFGS_hess   =   BFGS_hess
        
        self.BFGS_cholesky   =   BFGS_cholesky
        self.BFGS_mins_lst   =   mins_lst
        self.BFGS_maxs_lst   =   maxs_lst
        
        self.alpha_lst      =   np.asarray(alpha_lst)
        self.direction_lst  =   np.asarray(direction_lst)
##---------------------------------------------------
def subplot(T, method='QN_BFGS') : 
    if method == "QN_BFGS"    :
        dico_beta_map   =   T.QN_BFGS_bmap
        dico_beta_fin   =   T.QN_BFGS_bf
        
        mins    =   T.QN_BFGS_mins_lst
        maxs    =   T.QN_BFGS_maxs_lst
        
        titles = ["Hybrid optimization with QNBFGS: Beta comparaison", "Hybrid optimization with QNBFGS: Temperature fields"]
    else :
        dico_beta_map   =   T.betamap
        dico_beta_fin   =   T.beta_final

        mins    =   T.mins_lst
        maxs    =   T.maxs_lst
        
        titles = ["Beta comparaison", "Temperature fields"]
        
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
        
        return axes
##---------------------------------------------------##
##---------------------------------------------------##
##---------------------------------------------------##        
if __name__ == "__main__" :
#    __init__ (self, T_inf_lst, N_discr, dt, h, kappa, datapath, num_real = )
    parser = parser()
    print parser
    T = Temperature(parser)
    T.obs_pri_model()
    T.get_prior_statistics()
    
    
     
