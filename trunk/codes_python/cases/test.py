#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import csv, os, sys, warnings, argparse

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

from numdifftools import Gradient, Jacobian

from scipy.stats import norm as norm 

import class_temp_ML as ctml
ctml = reload(ctml)
 
def gaussienne (x, mu, sigma) :
    fac = 1./np.sqrt(2*np.pi*sigma**2)
    
    return fac * np.exp(-(x-mu)**2/2.0/sigma**2)
    
p = ctml.parser()
T = ctml.Temperature(p)
T.obs_pri_model()
T.get_prior_statistics()

print "cov_obs inversee \n \n  {} ".format(np.linalg.inv(T.full_cov_obs_dict["T_inf_50"]))

#str(input("Pause ... enter any key")) 

dict_discr = dict()
for t_inf in T.T_inf_lst:
    for i in xrange(T.N_discr-2) :
        key = "%d_%d" %(t_inf, i)
        dict_discr[key] = []

for t_inf in T.T_inf_lst :
    for it in xrange(T.num_real) :
        filename    =   'obs_T_inf_{}_{}.csv'.format(t_inf, it)
        T_obs_temp  =   T.pd_read_csv(filename)
        for n in xrange(T.N_discr-2) :
              key = "%d_%d" %(t_inf, n)
              dict_discr[key].append(T_obs_temp[n])

for k in dict_discr.keys() :
    dict_discr[k] = np.asarray(dict_discr[k])

plt.hist(dict_discr["%d_27" %(t_inf)]-T.T_obs_mean['T_inf_50'][27], 30, label='Histogramme des temperature au point 27 shifte par la moyenne de toutes ces valeurs' )

plt.figure("pdf au point %.2f" %(T.line_z[10]))
plt.plot()

T.minimization_with_first_guess()

#x = T.tab_normal(0,0.1,1000)
#T.get_prior_statistics()
#norm.pdf(x)

#    def minimization_with_first_guess(self) :
#        if self.stat_done == False : self.get_prior_statistics()

#        QN_BFGS_bmap,   QN_BFGS_bf        =   dict(),     dict()
#        QN_BFGS_hess,   QN_BFGS_cholesky  =   dict(),     dict()

#        mins,   maxs    =   dict(),     dict()
#        
#        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
#        beta_QNBFGS_real = []
#        alpha_lst = []
#        direction_lst =[]
#        for T_inf in self.T_inf_lst :
#            sT_inf  =   "T_inf_" + str(T_inf)
#            curr_d  =   self.T_obs_mean[sT_inf]
#            cov_m   =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else self.full_cov_obs_dict[sT_inf] 
#            cov_prior =  self.cov_pri_dict[sT_inf]
#            J = lambda beta : 0.5*  ( 
#                  np.dot( np.dot(np.transpose(curr_d - self.h_beta(beta, T_inf)),
#                    np.linalg.inv(cov_m)) , (curr_d - self.h_beta(beta, T_inf) )  )  
#                + np.dot( np.dot(np.transpose(beta - self.beta_prior), 
#                    np.linalg.inv(cov_prior) ) , (beta - self.beta_prior) ) 
#                                    )
                                    
#            first_guess_opti =  op.minimize(J, self.beta_prior, method="BFGS", tol=0.1) ## Simplement pour avoir Hess_0 
#            for item in first_guess_opti.iteritems() : print item
#            
#            beta_n  =   first_guess_opti.x
#            H_n     =   first_guess_opti.hess_inv
#            g_n     =   first_guess_opti.jac
#            
#            self.first_guess_opti = first_guess_opti
##            beta_n =    self.beta_prior
##            H_n    =    np.eye(self.N_discr-2)
##            g_n    =    np.ones((self.N_discr-2,))
#            
#            err, tol = 1.0, 1e-9
#            cpt, cpt_max    =   0,  5000
#            err_hess = 0.
#            while (np.abs(err) > tol) and (cpt<cpt_max) :
#                ## Incr
#                if cpt > 0 :
#                    H_n     =   H_nNext
#                    g_n     =   g_nNext 
#                    beta_n  =   beta_nNext      
#                
#                cpt += 1    
#                direction   =   np.dot(H_n, g_n)
#                alpha       =   self.backline_search(J, beta_n, direction)
#                print "alpha = ", alpha
##                alpha  = 1e-2
#                beta_nNext  =   beta_n - alpha*direction
#                self.beta_nNext = beta_nNext
#                alpha_lst.append(alpha)
#                direction_lst.append(direction)
#                
#                ## Estimation of H_nNext
#                g_nNext =   Gradient(J)(beta_nNext) # From numdifftools
#                print "compteur = {}, Gradient dans la boucle minimization \n {}:".format(cpt, g_nNext)
#                y_nNext =   g_nNext - g_n
#                s_nNext =   beta_nNext - beta_n
#                H_nNext =   self.Next_hess(H_n, y_nNext, s_nNext)
#                
#                err = (J(beta_nNext) - J(beta_n)) 
#                
#                err_hess = np.linalg.norm(H_nNext - H_n)
#                
#                print "cpt = {} \t err = {:.5}".format(cpt, err) 

#            print "Compteur = %d \t err_j = %.12f \t err_Hess %.12f" % (cpt, err, err_hess)
#            print "beta_nNext = ", beta_nNext
#            print "J(beta_nNext) = ", J(beta_nNext) 
#            
#            QN_BFGS_bmap[sT_inf]    =   beta_nNext
#            QN_BFGS_hess[sT_inf]    =   H_nNext
#            
#            QN_BFGS_cholesky[sT_inf]=   np.linalg.cholesky(H_nNext)
#            
#            QN_BFGS_bf[sT_inf]      =   QN_BFGS_bmap[sT_inf] + np.dot(np.linalg.cholesky(H_nNext), s)
#            
#            for i in xrange(100):
#                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
#                beta_QNBFGS_real.append(QN_BFGS_bmap[sT_inf] + np.dot(QN_BFGS_cholesky[sT_inf], s))
#            beta_QNBFGS_real.append(QN_BFGS_bf[sT_inf])
#            
#            for i in xrange(self.N_discr-2) :
#                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_QNBFGS_real]))
#                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_QNBFGS_real]))
#            
#            mins_lst =  [mins["T_inf_%d%03d" %(T_inf, i) ] for i in xrange(self.N_discr-2)]   
#            maxs_lst =  [maxs["T_inf_%d%03d" %(T_inf, i) ] for i in xrange(self.N_discr-2)]
#            
#        self.QN_BFGS_bmap   =   QN_BFGS_bmap
#        self.QN_BFGS_bf     =   QN_BFGS_bf
#        self.QN_BFGS_hess   =   QN_BFGS_hess
#        
#        self.QN_BFGS_cholesky   =   QN_BFGS_cholesky
#        self.QN_BFGS_mins_lst   =   mins_lst
#        self.QN_BFGS_maxs_lst   =   maxs_lst
#        
#        self.alpha_lst      =   np.asarray(alpha_lst)
#        self.direction_lst  =   np.asarray(direction_lst)


