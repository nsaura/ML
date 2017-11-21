#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm


def parser() :
    parser=argparse.ArgumentParser(description='You can initialize a case you want to study')
    #lists
    parser.add_argument('--T_inf_lst', '-T_inf_lst', nargs='+', action='store', type=int, default=['all'],dest='T_inf_lst', help='List of different T_inf\n' )
    parser.add_argument('--beta_prior', '-beta_prior', nargs='+', action='store', type=float, default=['init'],dest='beta_prior', help='beta_prior: first guess on the optimization solution\n')
    
    #digits
    parser.add_argument('--N_discr', '-N_discr', action='store', type=int, default=33, dest='N_discr', help='Define the number of discretization points \n' )
    parser.add_argument('--H', '-H', action='store', type=float, default=0.5, dest='h', help='Define the convection coefficient h \n' )
    parser.add_argument('--delta_t', '-dt', action='store', type=float, default=0.001, dest='dt', help='Define the time step disctretization \n' )
    parser.add_argument('--kappa', '-kappa', action='store', type=float, default=1.0, dest='kappa', help='Define the diffusivity number kappa \n' )
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=10, dest='num_real', help='Define the number of realization of epsilon(T) you want to pick up \n' )
    
    #strings
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data', dest='datapath', help='Define the directory where the data will be stored and read')
#    parser.add_argument('--data', '-df', action='store', type=str, default='./SSO_Inj_AVF_UDFTHEO-data', dest='datafile', help='Define your datafile\'s path and name, initialized at %(default)s')
#    parser.add_argument('--kind', '-k', action='store', type=str, default='cubic', dest='kind', help='Define the type of your interpolation : linear, cubic or quintic')
#    parser.add_argument('--init_data', '-id', action='store', type=str, default='./init_simu.csv', dest='init_simudata', help='Define the simulation datafile\'s path and name, initialized at %(default)s')
    return parser.parse_args()

##---------------------------------------------------------------
class Temperature() :
    def __init__ (self, T_inf_lst, N_discr, dt, H, kappa, datapath, num_real):
        np.random.seed(1000)
        plt.ion()
        
        try :
            self.len_T_lst = len(T_inf_lst) 
        except TypeError :
            T_inf_lst = [T_inf_lst]
            self.len_T_lst = len(T_inf_lst) 
        
        if T_inf_lst == ['all'] :
            T_inf_lst = [i*5 for i in xrange(1, 11)]
        
        self.T_inf_lst = T_inf_lst
        
        
        z_init, z_final =   0.0, 1.0
        dz = np.abs(z_final - z_init) / float(N_discr)
        ## Matrice pour la résolution
        A_diag = np.diag(np.transpose([(1+( 2.0)*dt/dz**2*kappa) for i in range(N_discr-2)])) 
        M1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), -1) # Inferieure
        P1 = np.diag(np.transpose([-dt/dz**2*kappa for i in range(N_discr-3)]), 1)  # Superieure 
        self.A = A_diag + M1 + P1
        
        self.noise = self.tab_normal(0, 0.1, N_discr-2)[0]
        
        self.line_z  = np.linspace(z_init, z_final, N_discr)[1:N_discr-1]
        self.num_real = num_real
        self.eps_0 = 5.*10**(-4)
        self.N_discr = N_discr
        self.kappa = kappa
        self.dt = dt        
        self.h = H
        
        datapath = os.path.abspath(datapath)
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
        
        print "T_inf_lst is now {}".format(self.T_inf_lst)
##---------------------------------------------------
    def pd_read_csv(self, filename) :
        path = os.path.join(self.datapath, filename)
        data = pd.read_csv(path).get_values()            
        return data.reshape(data.shape[0])
##---------------------------------------------------
    def pd_write_csv(self, filename, data) :
        path = os.path.join(self.datapath, filename)
        pd.DataFrame(data).to_csv(path, index=False, header= True)
##---------------------------------------------------
    def tab_normal(self, mu, sigma, length) :
        return (sigma * np.random.randn(length) + mu, 
                (sigma * np.random.randn(length) + mu).mean(), 
                (sigma * np.random.randn(length) + mu).std()
               ) 
##---------------------------------------------------   
    def true_model(self) :
        lst_gauss = [self.tab_normal(0,0.1,self.N_discr-2)[0] for i in xrange(self.num_real)]
        T_nNext_obs_lst, T_nNext_pri_lst, T_init = [], [], []
        
        for T_inf in self.T_inf_lst :
            for it, bruit in enumerate(lst_gauss) :
                # Obs and Prior Temperature field initializations
                T_n_obs =  map(lambda x : -4*T_inf*x*(x-1), self.line_z) 
                T_n_pri =  map(lambda x : -4*T_inf*x*(x-1), self.line_z) 
                
                T_init.append(T_n_obs)
                T_nNext_obs = T_n_obs
                T_nNext_pri = T_n_pri
            
                tol ,err_obs, err_pri, compteur = 1e-2, 1.0, 1.0, 0

                while (np.abs(err_obs) > tol) and (compteur <800) and (np.abs(err_pri) > tol):
                    if compteur > 0 :
                        T_n_obs = T_nNext_obs
                        T_n_pri = T_nNext_pri
                    compteur += 1
                    
                    # B_n = np.zeros((N_discr,1))
                    B_n_obs = T_n_obs
                    B_n_pri = T_n_pri
                     
                    for i in xrange(1,self.N_discr-2) :
                        B_n_obs[i] = T_n_obs[i] + self.dt*  (
                        ( 10**(-4) * ( 1.+5.*np.sin(3.*T_n_obs[i]*np.pi/200.) + 
                        np.exp(0.02*T_n_obs[i]) + bruit[i] ) ) *( T_inf**4 - T_n_obs[i]**4)
                         + self.h * (T_inf-T_n_obs[i])      )   
                        
                        B_n_pri[i] = T_n_pri[i] + self.dt * (
                        5 * 10**(-4) * (T_inf**4-T_n_pri[i]**4) * (1 + bruit[i])
                                                            )
                                                            
                    T_nNext_obs = np.dot(np.linalg.inv(self.A), np.transpose(B_n_obs))
                    T_nNext_pri = np.dot(np.linalg.inv(self.A), np.transpose(B_n_pri))
                    
                    T_nNext_obs_lst.append(T_nNext_obs)
                    T_nNext_pri_lst.append(T_nNext_pri)
                
                    obs_filename    =   'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                    pri_filename  =   'prior_T_inf_{}_{}.csv'.format(T_inf, it)
                    
                    self.pd_write_csv(obs_filename, T_nNext_obs)
                    self.pd_write_csv(pri_filename, T_nNext_pri)        
            
                    err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2)
                    err_pri = np.linalg.norm(T_nNext_pri - T_n_pri, 2)
            print "calculus with T_inf {} completed".format(T_inf)
        
        self.T_init             =   T_init    
        self.T_nNext_obs_lst    =   T_nNext_obs_lst
        self.T_nNext_pri_lst    =   T_nNext_pri_lst
##---------------------------------------------------   
    def get_prior_statistics(self, prior_sigma = [20, 2, 1, 1, 0.5, 1, 1, 1, 1, 0.8]):
        ## dict of covariances for different temperatures
        cov_obs_dict, cov_pri_dict = dict(), dict() 
        cov_pri_pri = dict()
        
        T_obs_mean  = dict()
        vals_obs_meshpoints = dict()
        for t in self.T_inf_lst :
            for j in xrange(self.N_discr-2) :
                key = "T_inf_{}_{}".format(t, j) 
                vals_obs_meshpoints[key] = []
        
        for i, (T_inf, prior_s) in enumerate(zip(self.T_inf_lst, prior_sigma)) :
            T_obs, T_prior = [], []     
            T_sum = np.zeros((self.N_discr-2))
            sT_inf = "T_inf_" + str(T_inf)
            
            for it in xrange(self.num_real) :
                obs_filename  =  'obs_T_inf_{}_{}.csv'.format(T_inf, it)
                pri_filename  =  'prior_T_inf_{}_{}.csv'.format(T_inf, it)
                
                # Compute covariance from data 
                T_temp = self.pd_read_csv(obs_filename)
                T_sum += T_temp / float(self.num_real)
                T_obs.append(T_temp)
                
                for j in xrange(self.N_discr-2) :
                    vals_obs_meshpoints[sT_inf+"_"+str(j)].append(T_temp[j])
                
                # We conserve the T_disc
                T_disc = self.pd_read_csv(pri_filename)
                T_prior.append(T_disc)
            
                # Compute prior cov for each prior sigma given in the article
        #    TT = np.asarray(np.sum([T_obs[i]/float(len(T_obs)) for i in xrange(len(T_obs))]))
            T_obs_mean[sT_inf] = T_sum
            std_mean_obs = np.asarray([np.std(vals_obs_meshpoints[sT_inf+"_"+str(j)]) for j in xrange(self.N_discr-2)])
            std_mean_prior  =   np.mean(np.asarray([np.std(T_prior[i]) for i in xrange(len(T_prior))]))
                
            cov_obs_dict[sT_inf]  =   np.diag([std_mean_obs[j]**2 for j in xrange(self.N_discr-2)])
            cov_pri_dict[sT_inf]  =   np.diag([prior_s**2 for j in xrange(self.N_discr-2)])
            cov_pri_pri[sT_inf]   =   np.diag([std_mean_prior**2 for j in xrange(self.N_discr-2)])
            
            self.vals_obs_meshpoints = vals_obs_meshpoints
            self.cov_obs_dict = cov_obs_dict
            self.cov_pri_dict = cov_pri_dict
            self.cov_pri_pri  = cov_pri_pri
            
            self.T_obs_mean = T_obs_mean
            
            self.stat_done = True
##---------------------------------------------------  
    def optimization(self, beta_prior, verbose=False) :
        if self.stat_done == False : self.get_prior_statistics()
        
        betamap, beta_final = dict(), dict()
        hess, cholesky = dict(), dict()
        
        mins, maxs = dict(), dict()
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        beta_var = []
        for T_inf in self.T_inf_lst :
            sT_inf  =   "T_inf_" + str(T_inf)
            curr_d  =   self.T_obs_mean[sT_inf]
            cov_m,  cov_prior   =   self.cov_obs_dict[sT_inf],    self.cov_pri_dict[sT_inf]
            
            J = lambda beta : 0.5*  ( 
                  np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
                    np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
                + np.dot( np.dot(np.transpose(beta - beta_prior), 
                    np.linalg.inv(cov_prior) ) , (beta - beta_prior) )   
                            ) ## Fonction de coût
                            
            # BFGS : quasi Newton method that approximate the Hessian on the fly
            opti = op.minimize(J, beta_prior, method="BFGS", tol=1e-10)

            betamap[sT_inf] =   opti.x
            hess[sT_inf]    =   opti.hess_inv
            cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf])
            
            self.opti = opti
            
            beta_final[sT_inf]  =   betamap[sT_inf] + np.dot(cholesky[sT_inf].T, s)  
            
            for i in xrange(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append(betamap[sT_inf] + np.dot(cholesky[sT_inf], s))
            beta_var.append(beta_final[sT_inf])
            
            for i in xrange(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
            
            mins_list =  [mins["T_inf_" + str(50000+i)] for i in xrange(self.N_discr-2)]   
            maxs_list =  [maxs["T_inf_" + str(50000+i)] for i in xrange(self.N_discr-2)]
            
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

                axes[0].plot(self.line_z, mins_list, label='Valeurs minimums', marker='s', linestyle='none', color='magenta')
                axes[0].plot(self.line_z, maxs_list, label='Valeurs maximums', marker='s', linestyle='none', color='black')
                axes[0].set_title("Optimized beta and Duraisamy beta")
                axes[1].set_title("Temperature field with optimized betas and true solution")

                axes[0].legend(loc='best', fontsize = 10, ncol=2)
                axes[1].legend(loc='best', fontsize = 10, ncol=2)
                
                
                
                plt.show()
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholseky   =   cholesky
        self.beta_final =   beta_final
        self.mins       =   mins_list
        self.maxs       =   maxs_list
        self.beta_var   =   beta_var
##--------------------------------------------------- 
    def h_beta(self, beta, T_inf, noise= 'none') :
        err, tol, compteur, compteur_max = 1., 1e-3, 0, 1000
        T_n = map(lambda x : -4*T_inf*x*(x-1), self.line_z)
        T_nNext = T_n
        B_n = np.zeros((self.N_discr-2))
        
        while (np.abs(err) > tol) and (compteur <= compteur_max) :
            if compteur >= 1 :
                T_n = T_nNext
            compteur +=1 
            
            for i in xrange(self.N_discr-2) :
                B_n[i] = T_n[i] + self.dt*(beta[i])*self.eps_0*(T_inf**4 - T_n[i]**4)
            
            T_nNext = np.dot(np.linalg.inv(self.A), np.transpose(B_n))
            err = np.linalg.norm(T_nNext - T_n, 2) # Norme euclidienne
            if compteur == compteur_max :
                warnings.warn("\x1b[7;1;255mH_BETA function's compteur has reached its maximum value, still, the erreur is {} whereas the tolerance is {} \x1b[0m".format(err, tol))
        
        return T_nNext 
##---------------------------------------------------
    def true_beta(self, T, T_inf) : 
        return np.asarray(
        [ 1./self.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i]) + self.noise[i] ) *10**(-4) + self.h / self.eps_0*(T_inf - T[i])/(T_inf**4 - T[i]**4)  for i in xrange(self.N_discr-2)]
                          )
##---------------------------------------------------    
    def subplot(self) : 
        for T_inf in self.T_inf_lst :
            sT_inf = "T_inf_" + str(T_inf)
            curr_d = self.T_obs_mean[sT_inf]
            fig, axes = plt.subplots(1,2,figsize=(20,10))
            colors = 'green', 'orange'
                    
            axes[0].plot(self.line_z, self.beta_final[sT_inf], label = "Beta for {}".format(sT_inf), 
                marker = 'o', linestyle = 'None', color = colors[0])
                
            axes[0].plot(self.line_z, self.betamap[sT_inf], label = 'Betamap for {}'.format(sT_inf),      
                 marker = 'o', linestyle = 'None', color = colors[1])
            
            axes[0].plot(self.line_z, self.true_beta(curr_d, T_inf), label = "True beta profil {}".format(sT_inf))
            
            axes[1].plot(self.line_z, self.h_beta(self.beta_final[sT_inf], T_inf), 
                label = "h_beta {}".format(sT_inf), marker = 'o', linestyle = 'None', color = colors[0])
            axes[1].plot(self.line_z, self.h_beta(self.betamap[sT_inf], T_inf), 
                label = "h_betamap {}".format(sT_inf), marker = 'o', linestyle = 'None', color = colors[1])
            axes[1].plot(self.line_z, curr_d, label= "curr_d {}".format(sT_inf))

            axes[0].set_title("Optimized beta and Duraisamy beta")
            axes[1].set_title("Temperature field with optimized betas and true solution")

            axes[0].legend(loc='best', fontsize = 10, ncol=2)
            axes[1].legend(loc='best', fontsize = 10, ncol=2)
            
            plt.show()
##---------------------------------------------------    




if __name__ == "__main__" :
#    __init__ (self, T_inf_lst, N_discr, dt, h, kappa, datapath, num_real = )
    parser = parser()
    T = Temperature(parser.T_inf_lst, parser.N_discr, parser.dt, parser.h, parser.kappa, parser.datapath, parser.num_real)
    beta_prior = np.asarray([1 for i in xrange(parser.N_discr-2)])

#    print T.noise
    
    
    
    
    
