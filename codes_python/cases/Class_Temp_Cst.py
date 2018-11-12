#!/usr/bin/python2.7
# -*- coding: utf-8-*-

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import argparse

import os
import os.path as osp

from scipy import optimize as op
from itertools import cycle

import numdifftools as nd

import time

#rc('text', usetex=False)

class Temperature_cst() :
##---------------------------------------------------------------
    def __init__ (self, parser):
        """
        This object has been made to solve optimization problem.
        """
#        np.random.seed(1000) ; #plt.ion()
        
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
        
        for t in T_inf_lst :
            key = "T_inf_%d" %(t) not in self.prior_sigma.keys()
            self.prior_sigma["T_inf_%d" %(t)] = 1.
        
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
        bool_written= dict()
        
        runs = set()
        runs.add("stat")
        
        for t in self.T_inf_lst :
            sT_inf = "T_inf_%s" %(str(t))
            runs.add("opti_scipy_%s" %(sT_inf))
            runs.add("adj_bfgs_%s" %(sT_inf))
            
        for r in runs :
            bool_method[r] = False
            bool_written[r]= False
            
        self.bool_method = bool_method
        self.bool_written = bool_written
        

        #Création des différents fichiers
        self.date = time.strftime("%m_%d_%Hh%M", time.localtime())
        
        # On test si les dossiers existent, sinon on les créé
        # Pour écrire la valeur de cov_post. Utile pour le ML 
        if osp.exists(osp.abspath("./data/post_cov")) == False :
            os.mkdir(osp.abspath("./data/post_cov"))
        
        if osp.exists(osp.abspath("./data/matrices")) == False :
            os.mkdir(osp.abspath("./data/matrices"))
        self.path_fields = osp.abspath("./data/matrices")        
                
        if osp.exists(datapath) == False :
            os.mkdir(datapath)
        
        if osp.exists(parser.logbook_path) == False :
            os.mkdir(parser.logbook_path)
        
        if osp.exists("./err_check") == False :
            os.mkdir("./err_check")
            
        self.err_title = osp.join("err_check", "%s_err_check.csv" %(self.date))
        self.logout_title = osp.join(parser.logbook_path, "%s_logbook.csv" %(self.date))
        
        # On intialise les fichiers en rappelant la teneur de la simulation
        for f in {open(self.logout_title, "w"), open(self.err_title, "w")} :
            f.write("\n#######################################################\n")
            f.write("## Logbook: simulation launched %s ## \n" %(time.strftime("%Y_%m_%d_%Hh%Mm%Ss", time.localtime())))
            f.write("#######################################################\n")
            f.write("Simulation\'s features :\n{}\n".format(parser))
            f.close()

        self.datapath   =   datapath
        self.parser     =   parser
##----------------------------------------------------##        
##----------------------------------------------------##
######                                            ######
######           Modifieurs et built-in           ######
######                                            ######
##----------------------------------------------------## 
##----------------------------------------------------##
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
#    def pd_read_csv(self, filename) :
#        """
#        Argument :
#        ----------
#        filename : the file's path with or without the extension which is csv in any way. 
#        """
#        if osp.splitext(filename)[-1] is not ".csv" :
#            filename = osp.splitext(filename)[0] + ".csv"
#        data = pd.read_csv(filename).get_values()
#        data = pd.read_csv(filename).get_values()
##        print data
##        print data.shape
#        if np.shape(data)[1] == 1 : 
#            data = data.reshape(data.shape[0])
#        return data
###---------------------------------------------------
#    def pd_write_csv(self, filename, data) :
#        """
#        Argument :
#        ----------
#        filename:   the path where creates the csv file;
#        data    :   the data to write in the file
#        """
#        path = osp.join(self.datapath, filename)
#        pd.DataFrame(data).to_csv(path, index=False, header= True)
###---------------------------------------------------
    def tab_normal(self, mu, sigma, length) :
        return ( sigma * np.random.randn(length) + mu, 
                (sigma * np.random.randn(length) + mu).mean(), 
                (sigma * np.random.randn(length) + mu).std()
               ) 
##---------------------------------------------------
    def write_fields(self) :
        for t in self.T_inf_lst :
            sT_inf = "T_inf_%d" %(t)
            if self.bool_method["opti_scipy_%s" %(sT_inf)] :
                for f, s in zip([self.betamap, self.cholesky],["beta", "cholesky"]) :
                    path_tosave = "opti_scipy_%s_%s_N%d_cov%s.npy" %(s, sT_inf, self.N_discr-2, self.cov_mod)
                    path_tosave = osp.join(self.path_fields, path_tosave)
                    np.save(path_tosave, f[sT_inf])
                    
            
            if self.bool_method["adj_bfgs_%s" %(sT_inf)] :
                for f, s in zip([self.bfgs_adj_bmap, self.bfgs_adj_cholesky],["beta", "cholesky"]) :
                    path_tosave = "adj_bfgs_%s_%s_N%d_cov%s.npy" %(s, sT_inf, self.N_discr-2, self.cov_mod)
                    path_tosave = osp.join(self.path_fields, path_tosave)
                    np.save(path_tosave, f[sT_inf])

        print("Fields written see {}".format(self.path_fields))
##---------------------------------------------------   
    def h_beta(self, beta, T_inf, verbose=False) :
#        T_n = list(map(lambda x : -4*T_inf*x*(x-1), self.line_z))
#   Initial condition
        
        sT_inf  =   "T_inf_" + str(T_inf)
        T_n= self.T_obs_mean[sT_inf]

        B_n = np.zeros((self.N_discr-2))
        T_nNext = T_n
        
        err, tol, compteur, compteur_max = 1., 1e-7, 0, 5000
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
                print("\x1b[7;1;255mH_BETA function's counteur reached max value, err is {} not below tol {} \x1b[0m".format(err, tol))

            if verbose==True :
                print ("Err = {} ".format(err))
                print ("Compteur = ", compteur)
        
        if verbose == True :
            print("H_beta ok")
#        time.sleep(1)
        return T_nNext 
##---------------------------------------------------
    def true_beta(self, T, T_inf) : 
        """
        Calcul de beta théorique pour un cas considéré
        
        Args : 
        T : champ de température obs ou inferré  
        T_inf : valeur de T_inf 
        ## Attention : cette version ne marche que lorsque T_inf est un scalaire ! ##
            On utilisera la version GPC pour traiter le cas T_inf variable 
        """
        # dans t1 on ne rajoute pas le bruit contrairement à l'équation 36
        t1 = np.asarray([ 1./self.eps_0*(1. + 5.*np.sin(3.*np.pi/200. * T[i]) + np.exp(0.02*T[i])) *10**(-4) for i in range(self.N_discr-2)])
        t2 = np.asarray([self.h / self.eps_0*(T_inf - T[i])/(T_inf**4 - T[i]**4)  for i in range(self.N_discr-2)]) 
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
        
        for T_inf in self.T_inf_lst :
            for it, bruit in enumerate(self.lst_gauss) :
                # Obs and Prior Temperature field initializations
                # For python3.5 add list( )
                
                # Titre qui prend en compte la forme de la covariance, de la température en cours, et le nombre de points de discrétisation
                obs_filename  =  '{}_obs_T_inf_{}_N{}_{}.npy'.format(self.cov_mod, T_inf, self.N_discr-2, it)
                pri_filename  =  '{}_pri_T_inf_{}_N{}_{}.npy'.format(self.cov_mod, T_inf, self.N_discr-2, it)
                ## Il faudra peut être zipper tout ces fichiers, les commité puis possiblement les unziper
                ## On utilisera alors cette procédure
#                import zipfile
#                zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
#                zip_ref.extractall(directory_to_extract_to)
#                zip_ref.close()
                
                # écriture en chemin absolue pour éviter conflits
                obs_filename = osp.join(self.datapath, obs_filename) 
                pri_filename = osp.join(self.datapath, pri_filename)
                
#                print obs_filename
#                print pri_filename
                tol ,err_obs, err_pri, compteur = 1e-8, 1.0, 1.0, 0                
                
                # L'utilité de faire attention aux titres des fichiers c'est qu'on n'est plus obligé de les recalculer à toutes les SIMULATIONS (et non itération)
                # Une fois calculée les champs de températures sont enregistrés
                if osp.isfile(obs_filename) and osp.isfile(pri_filename) :
#                    print "exists"
                    continue
                
                # Initialisation des champs T que l'on va ensuite faire converger 
#                T_n_obs =  list(map(lambda x : -4*T_inf*x*(x-1), self.line_z) ) # On va utiliser le modèle complet -> obs
#                T_n_pri =  list(map(lambda x : -4*T_inf*x*(x-1), self.line_z) ) # On va utiliser le modèle incomplet -> pri
                
                T_n_obs =  list(map(lambda x : 0, self.line_z) ) # On va utiliser le modèle complet -> obs
                T_n_pri =  list(map(lambda x : 0, self.line_z) ) # On va utiliser le modèle incomplet -> pri                
                
                T_init.append(T_n_obs)
                T_nNext_obs = T_n_obs
                T_nNext_pri = T_n_pri
            
                B_n_obs     =   np.zeros((self.N_discr-2, 1))
                B_n_pri     =   np.zeros((self.N_discr-2, 1))
                T_n_obs_tmp =   np.zeros((self.N_discr-2, 1))
                T_n_pri_tmp =   np.zeros((self.N_discr-2, 1))
                
                while (np.abs(err_obs) > tol) and (compteur < 6000) and (np.abs(err_pri) > tol):
                    if compteur > 0 :
                        # Initialisation pour itération suivante
                        T_n_obs = T_nNext_obs
                        T_n_pri = T_nNext_pri
                    compteur += 1
                    
                    # B_n = np.zeros((N_discr,1))
                    T_n_obs_tmp = np.dot(self.A2, T_n_obs) # Voir les mails
                    T_n_pri_tmp = np.dot(self.A2, T_n_pri) # Voir les mails
                     
                    for i in range(self.N_discr-2) :
                        B_n_obs[i] = T_n_obs_tmp[i] + self.dt*  (
                        ( 10**(-4) * ( 1.+5.*np.sin(3.*T_n_obs[i]*np.pi/200.) + 
                        np.exp(0.02*T_n_obs[i]) + bruit[i] ) ) *( T_inf**4 - T_n_obs[i]**4)
                         + self.h * (T_inf-T_n_obs[i])          )   
                        
                        B_n_pri[i] = T_n_pri_tmp[i] + self.dt * (5 * 10**(-4) * (T_inf**4-T_n_pri[i]**4) * (1 + bruit[i]))
                                                            
                    T_nNext_obs = np.dot(np.linalg.inv(self.A1), B_n_obs)
                    T_nNext_pri = np.dot(np.linalg.inv(self.A1), B_n_pri)
                    
                    T_nNext_obs_lst.append(T_nNext_obs)
                    T_nNext_pri_lst.append(T_nNext_pri)
                
                    err_obs = np.linalg.norm(T_nNext_obs - T_n_obs, 2) 
                    err_pri = np.linalg.norm(T_nNext_pri - T_n_pri, 2)
                
                # On écrit les champs convergés 
                np.save(obs_filename, T_nNext_obs)
                np.save(pri_filename, T_nNext_pri)             
            
            if compteur == 0:
                print("T_inf = {}, files loaded.".format(T_inf))
            else : 
                print ("Calculus with T_inf={} completed. Convergence status :".format(T_inf))
                print ("Err_obs = {} ".format(err_obs))    
                print ("Err_pri = {} ".format(err_pri))
                print ("Iterations = {} ".format(compteur))
        
        self.T_init             =   T_init    
        self.T_nNext_obs_lst    =   T_nNext_obs_lst
        self.T_nNext_pri_lst    =   T_nNext_pri_lst
##---------------------------------------------------   
    def get_prior_statistics(self, verbose = False):
       # Le code a été pensé pour être lancé avec plusieurs valeurs de T_inf dans la liste T_inf_lst.
        # On fonctionne donc en dictionnaire pour stocker les valeurs importantes relatives à la température en 
        # cours. De cette façon, on peut passer d'une température à une autre, et préparer les covariances pour l'optimisation
        # ceux pour des T_inf différentes, sans craindre de perdre ou de mélanger les statistiques        
        cov_obs_dict    =   dict() 
        cov_pri_dict    =   dict()
        
        mean_meshgrid_values=   dict()  
        full_cov_obs_dict   =   dict()        
        vals_obs_meshpoints =   dict()
        
        condi = dict()
        T_obs_mean  = dict()
        
        for t in self.T_inf_lst :
            for j in range(self.N_discr-2) :
                key = "T_inf_{}_{}".format(t, j)    # Clés prenant la température en cours et le point de discrétisation
                vals_obs_meshpoints[key] =  []      # On va calculer les écarts types dans ce tableau
        
        for i, T_inf in enumerate(self.T_inf_lst) :
            
            T_obs, T_prior = [], []     
            T_sum = np.zeros((self.N_discr-2))
            sT_inf = "T_inf_" + str(T_inf)
            
            for it in range(self.num_real) :
                # num_real : nombre de tirage effectués lors de la fonction obs_pri_modele. Bruit gaussien de moyenne nulle et d'écart type 0.1
                
                # Titre qui prend en compte la forme de la covariance, de la température en cours, et le nombre de points de discrétisation
                # Correspondent aux fichiers enregistrés dans la fonction précédente
                obs_filename  =  '{}_obs_T_inf_{}_N{}_{}.npy'.format(self.cov_mod, T_inf, self.N_discr-2, it)
                pri_filename  =  '{}_pri_T_inf_{}_N{}_{}.npy'.format(self.cov_mod, T_inf, self.N_discr-2, it)
                
                # Pour éviter les conflits d'une machine à l'autre
                obs_filename = osp.join(self.datapath, obs_filename)
                pri_filename = osp.join(self.datapath, pri_filename)
                
                # Calcule des covariances
                T_temp = np.load(obs_filename).ravel()
                T_sum += T_temp / float(self.num_real) # T_sum est la température moyenne sur le nombre de tirages. Espérance pour les variances
                
                for j in range(self.N_discr-2) :
                    vals_obs_meshpoints[sT_inf+"_"+str(j)].append(T_temp[j]) # On enregistre les différentes valeurs de température en chaque point
                
                # On récupère les températures du modèle incomplet
                T_pri = np.load(pri_filename).ravel()
                T_prior.append(T_pri)
                
                if verbose == True :
                    plt.plot(self.line_z, T_pri, label='pri real = %d' %(it), marker='o', linestyle='none')
                    plt.plot(self.line_z, T_temp, label='obs real = %d' %(it))

            T_obs_mean[sT_inf] = T_sum # Voir commentaire plus haut sut T_sum
            
            # À partir des valeurs de température en chaque point, on calcule la covariance diagonale des observables
            std_meshgrid_values = np.asarray([np.std(vals_obs_meshpoints[sT_inf+"_"+str(j)]) for j in range(self.N_discr-2)]) 
            cov_obs_dict[sT_inf] = np.diag([std_meshgrid_values[j]**2 for j in range(self.N_discr-2)])
            self.cov_obs_dict = cov_obs_dict

            # On veut aussi calculer la covariance full
            full_cov = np.zeros((self.N_discr-2, self.N_discr-2)) 
            
            for it in range(self.num_real) :
                obs_filename  =  '{}_obs_T_inf_{}_N{}_{}.npy'.format(self.cov_mod, T_inf, self.N_discr-2, it)
                obs_filename = osp.join(self.datapath, obs_filename)
                T_temp = np.load(obs_filename).ravel()
                
                # Façon non Pythonique mais ça marche quand même
                for ii in range(self.N_discr-2)  :
                    for jj in range(self.N_discr-2) : 
                        full_cov[ii,jj] += (T_temp[ii] - T_obs_mean[sT_inf][ii]) * (T_temp[jj] - T_obs_mean[sT_inf][jj])/float(self.num_real)
            
            full_cov_obs_dict[sT_inf] = full_cov 
            self.full_cov_obs_dict  =   full_cov_obs_dict
            
            if verbose == True:  print ("cov_obs :\n{}".format(full_cov))
            
            # On voudrait peut être regarder le conditionnement de la matrice
            condi['full' + sT_inf] = np.linalg.norm(full_cov)*np.linalg.norm(np.linalg.inv(full_cov))
            condi['diag' + sT_inf] = np.linalg.norm(self.cov_obs_dict[sT_inf])*np.linalg.norm(np.linalg.inv(self.cov_obs_dict[sT_inf]))
            
            # Covariance prior 
            cov_pri_dict[sT_inf] = np.diag([self.prior_sigma[sT_inf]**2 for j in range(self.N_discr-2)])
            self.cov_pri_dict = cov_pri_dict
            
            # On teste si la covariance prior ainsi calculée vérifie la condition : les sigma prior doivent être compris dans l'intervalle +- sigma_obs
            self.cov_prior(T_inf)            
            
            # Quelques attributs utilises pour la suite
            self.T_obs_mean = T_obs_mean
            self.vals_obs_meshpoints = vals_obs_meshpoints
            
            # On précise que les statistiques ont été calculées
            self.bool_method["stat"] = True
##----------------------------------------------------##    
    def cov_prior(self, T_inf) :
        sT_inf = "T_inf_" + str(T_inf)
        cov_obs = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                  self.full_cov_obs_dict[sT_inf]
        sigma_obs = np.diag(cov_obs)
        
        cov_pri = self.cov_pri_dict[sT_inf]
        
        b_distrib_dict = dict()
        
        for j  in range(self.N_discr-2) :
            b_distrib_dict[sT_inf+"_"+str(j)] = []
            
        for it in range(self.num_real) :
            s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
            b_distrib = self.beta_prior +\
                np.dot(np.linalg.cholesky(cov_pri), s)
            
            for j in range(self.N_discr-2) :
                b_distrib_dict[sT_inf+"_"+str(j)].append(b_distrib[j])
                
        sigma_pri = np.asarray([np.std(b_distrib_dict[sT_inf+"_"+str(j)]) for j in range(self.N_discr-2)])
        
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
        print("-- The scipy OP began --\n")
        
        # Mesure de sécurité        
        if self.bool_method["stat"] == False : self.get_prior_statistics()

        # Le code a été pensé pour être lancé avec plusieurs valeurs de T_inf dans la liste T_inf_lst.
        # On fonctionne donc en dictionnaire pour stocker les valeurs importantes relatives à la température en 
        # cours. De cette façon, on peut passer d'une température à une autre, et donc recommencer une optimisation 
        # pour T_inf différente, sans craindre de perdre les résultats de la T_inf précédente        
        
        betamap, beta_final = dict(), dict()
        hess, cholesky = dict(), dict()
        
        mins_dict, maxs_dict = dict(), dict()
        
        sigma_post_dict = dict()
        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
        
        ######################
        ##-- Optimisation --##
        ######################
        
        for T_inf in self.T_inf_lst :
            mins, maxs = dict(), dict()
            print ("Optimisation pour T_inf = %d" %(T_inf))
            sT_inf = "T_inf_" + str(T_inf) # clé en cours pour les dictionnaires 
            
            # On récupère  les covariances 
            curr_d = self.T_obs_mean[sT_inf] # Temperature moyenne sur les self.num_real tirages
            cov_m  = self.cov_obs_dict[sT_inf] if self.cov_mod=='diag'\
                        else self.full_cov_obs_dict[sT_inf]           
            cov_prior = self.cov_pri_dict[sT_inf]
            
            inv_cov_pri = np.linalg.inv(cov_prior)  
            inv_cov_obs = np.linalg.inv(cov_m)
            
            # On construit la fonction de coût en trois temps
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot((curr_d - self.h_beta(beta, T_inf)).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                      
            ## Fonction de coût                  
            J = lambda beta : J_1(beta) + J_2(beta)  ## Fonction de coût
            
            print ("J(beta_prior) = {}".format(J(self.beta_prior)))
            
            # Calcule de dJ/dbeta avec méthode adjoint equation (13) avec (12)
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
            self.opti_obj   =   opti_obj # object that contains all the scipy optimizations information
            
            # Cbmap
            cholesky[sT_inf]=   np.linalg.cholesky(hess[sT_inf]) 
                            
            print ("Sucess state of the optimization {}".format(self.opti_obj.success))
           
            beta_final[sT_inf]  =   betamap[sT_inf] + np.dot(cholesky[sT_inf], s)  
            
            beta_var = []
            sigma_post = []
            
            # On va à présent faire des tirages (sample) à partir de beta_map et de Cholesky
            for i in range(249):
                s = self.tab_normal(0,1,self.N_discr-2)[0]
                beta_var.append(betamap[sT_inf] + np.dot(cholesky[sT_inf], s))
            beta_var.append(beta_final[sT_inf])
            # Beta var est une liste de listes. Chacune de ces liste mesure self.N_discr - 2 et contient la valeur
            # les samples tirées plus haut.
            # Pour les tracés, on calcule les minimum et maximum de toutes ces listes pour chaque point.
            
            # Calcule des min maxs et std sur chaque point            
            for i in range(self.N_discr-2) :
                mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                sigma_post.append(np.std([j[i] for j in beta_var])) 
                # Pour une itération, i (point de discrétisation) est fixé et on parcours les liste contenues dans
                # beta_var. De cette façon on raisonne sur le i-ème élément de tous les samples.
                # On calcule également l'écart type de chaque point.  

            sigma_post_dict[sT_inf] = sigma_post 

            # On rassemble les valeurs des mins relatives à tous les points de discrétisation dans une seule liste
            mins_lst =  [mins[sT_inf + str("{:03d}".format(i)) ] for i in range(self.N_discr-2)]   
            maxs_lst =  [maxs[sT_inf + str("{:03d}".format(i)) ] for i in range(self.N_discr-2)]

            # Puis on les garde pour la température en cours
            mins_dict[sT_inf] = mins_lst
            maxs_dict[sT_inf] = maxs_lst
            
            # On précise que le cas en cours a été traité. On pourra alors le tracé (voir class_functions_aux.py subplot_cst) et l'écrire
            self.bool_method["opti_scipy_"+sT_inf] = True
            # On l'écrit
            f = open(self.logout_title, "a")
            f.write("\nSCIPY_OPTI\n")
            f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(sT_inf))
            f.write("SCIPY: J(beta_last) = {}\n".format(self.opti_obj.values()[5]))
            f.write("beta_last = {}\n".format(self.opti_obj.x))
            f.write("g_last = {}\n".format(np.linalg.norm(self.opti_obj.jac, np.inf)))
            
            f.write("Message : {} \t Success = {}\n".format(self.opti_obj.message, self.opti_obj.success))
            f.write("N-Iterations:  = {}\n".format(self.opti_obj.nit))
            
            self.bool_written["opti_scipy_"+sT_inf] = True
            f.close()
        ##############################
        ##-- Passages en attribut --##
        ##############################
        self.betamap    =   betamap
        self.hess       =   hess
        self.cholesky   =   cholesky
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
        inter_plot to see the evolution of the inference
        verbose to print different information during the optimization
        """
        print("-- Home made ADJ_BFGS OP began --\n")
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        # Le code a été pensé pour être lancé avec plusieurs valeurs de T_inf dans la liste T_inf_lst.
        # On fonctionne donc en dictionnaire pour stocker les valeurs importantes relatives à la température en 
        # cours. De cette façon, on peut passer d'une température à une autre, et donc recommencer une optimisation 
        # pour T_inf différente, sans craindre de perdre les résultats de la T_inf précédente
        
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.unicode'] = True
        
        bfgs_adj_grad,   bfgs_adj_gamma     =   dict(), dict()
        bfgs_adj_bmap,   bfgs_adj_bf        =   dict(), dict()
        bfgs_adj_hessinv,bfgs_adj_cholesky  =   dict(), dict()
        
        bfgs_adj_mins,   bfgs_adj_maxs  =   dict(),  dict()
        
        bfgs_adj_mins_dict, bfgs_adj_maxs_dict = dict(), dict()
        bfgs_adj_sigma_post  = dict()
        
        self.too_low_err_hess = dict()
        sup_g_stagne = False

        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0]) # vecteur aléatoire
        
#        rc('text', usetex=True)
        
        for T_inf in self.T_inf_lst :
            sT_inf      =   "T_inf_%d" %(T_inf) # Clé pour simplifier
            sigmas      =   np.sqrt(np.diag(self.cov_obs_dict[sT_inf])) #Std obs
            curr_d      =   self.T_obs_mean[sT_inf] # Temperature moyenne sur les self.num_real tirages
            
            # On récupère  les covariances 
            cov_obs     =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                            self.full_cov_obs_dict[sT_inf]
            
            cov_pri     =   self.cov_pri_dict[sT_inf]
            
            inv_cov_pri =   np.linalg.inv(cov_pri)  
            inv_cov_obs =   np.linalg.inv(cov_obs)
            
            # On construit la fonction de coût en trois temps
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            ## Fonction de coût
            J = lambda beta : J_1(beta) + J_2(beta)  
            
            # Calcule de dJ/dbeta avec méthode adjoint equation (13) avec (12)
            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            err_beta = err_hess = err_j = 1            
            cpt, cptMax =   0, self.cpt_max_adj
            
            colors = iter(cm.plasma_r(np.arange(cptMax)))
            
            sup_g_lst = [] # Pour tester si la correction a stagné
            corr_chol = [] # Pour comptabiliser les fois ou la hessienne n'était pas définie positive
            al2_lst  =  [] # Comptabilise les fois où les DEUX conditions de Wolfe sont vérifiés
            
            print("J(beta_prior) = {} \t T_inf = {}".format(J(self.beta_prior), T_inf))
            
            ########################
            ##-- Initialisation --##
            ########################

            beta_n  =   self.beta_prior
            g_n     =   grad_J(beta_n)
            
            g_sup = np.linalg.norm(g_n, np.inf)
            sup_g_lst.append(np.linalg.norm(g_n, np.inf))
            
            print ("\x1b[1;37;44mSup grad : %f \x1b[0m" %(np.linalg.norm(g_n, np.inf))) # Affichage surligné bleu

            # Hessienne (réinitialisé plus tard)
            H_n_inv =   np.eye(self.N_discr-2)
            self.debug["first_hess"] = H_n_inv
            
            # Tracé des différentes figures (Evolutions de béta et du gradient en coursc d'optimization)
            if inter_plot==True :
                fig, ax = plt.subplots(1,2,figsize=(13,7))
                ax[0].plot(self.line_z, beta_n, label="beta_prior")
                ax[1].plot(self.line_z, g_n,    label="gradient prior")
            
                plt.pause(0.05)
            
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
                   
                    g_n     =   g_nNext
                    g_sup   =   np.linalg.norm(g_n, np.inf) # norme infini du gradient
                    
                    H_n_inv   =   H_nNext_inv

                    #MAJ des figures avec nouveaux tracés
                    plt.figure("Evolution de l'erreur Tinf=%d" %(T_inf))
                    plt.scatter(cpt, g_sup, c='black', s=7)
                    plt.xlabel("Iterations")
                    
#                    rc('text', usetex=True)
                    plt.ylabel("loglog " + r"$||\nabla_k J(\beta)||_\infty$")
#                    rc('text', usetex=False)
                    
                    c = next(colors) 
                    
                    ax[0].plot(self.line_z, beta_n, color=c)
                    ax[1].semilogy(self.line_z, g_n, color=c)
                    
                    ax[0].legend(["beta cpt%d " %(cpt) + r"$T_{\infty} = %d$" %(T_inf)])
                    ax[1].legend(["grad cpt%d" %(cpt)])
                    
                    ax[0].set_xlabel("X-domain")
                    ax[0].set_ylabel("Beta in optimization")
                    
                    ax[1].set_xlabel("X-domain")
#                    rc('text', usetex=True)
                    ax[1].set_ylabel(r"$\nabla J(\beta_k)$")
                    
                    leg_0 = ax[0].get_legend()
                    leg_1 = ax[1].get_legend()
#                    rc('text', usetex=False)
                    
                    leg_0.legendHandles[-1].set_color(c)
                    leg_1.legendHandles[-1].set_color(c)
                    
                    fig.tight_layout()
                    
                    if inter_plot == True :
                        plt.pause(0.05)
                                        
                    # MAJ de la liste des gradient.                      
                    sup_g_lst.append(g_sup)
                    if len(sup_g_lst) > 6 :
                        lst_ = sup_g_lst[(len(sup_g_lst)-5):] # Prend les 5 dernières valeurs des sup_g
                        mat = [[abs(i - j) for i in lst_] for j in lst_] # Matrices des différences val par val
                        sup_g_stagne = (np.linalg.norm(mat, 2) <= 1e-2) # True ? -> alpha = 1 sinon backline_search
                    
                    # Affichage des données itérations précédentes, initialisant celle à venir
                    print("Compteur = %d" %(cpt))
                    print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))
                    print("Stagne ? {}".format(sup_g_stagne))
                    
                    if verbose == True :
                        print ("beta cpt {}:\n{}".format(cpt,beta_n))
                        print("grad n = {}".format(g_n))                            
                        print("beta_n = \n  {} ".format(beta_n))
                        print("cpt = {} \t err_beta = {} \t err_hess = {}".format(cpt, \
                                                           err_beta, err_hess) )
                        print ("Hess cpt {}:\n{}".format(cpt, H_n_inv))
                
                #################
                ##-- Routine --##
                #################
                
                # Calcule : -np.dot(grad_J(bk), d_n). 
                # GD pour Gradient Descent.  Doit être négatif (voir après et Nocedal et Wright)
                GD = lambda H_n_inv :  -np.dot(g_n[np.newaxis, :],\
                                    np.dot(H_n_inv, g_n[:, np.newaxis]))[0,0]                
                ## Calcule de la direction 
                d_n     =   - np.dot(H_n_inv, g_n)
                test_   =   (GD(H_n_inv) < 0) ## Booléen GD ou pas ?
                
                print("d_n descent direction : {}".format(test_))    
                
                if test_  == False : 
                    # Si GD == False : H_n n'est pas définie positive (is not Matrix Positive Definite)
                    self.positive_definite_test(H_n_inv, verbose=False) # Permet de vérifier diagnostique (obsolète)

                    H_n_inv = self.cholesky_for_MPD(H_n_inv, fac = 2.) # Corr CF Nocedal Wright (page ou chap ?)
                    print("cpt = {}\t cholesky for MPD used.")
                    print("new d_n descent direction : {}".format(test(H_n_inv) < 0))
                    
                    d_n     =   - np.dot(H_n_inv, g_n)  # Nouvelle direction conforme
                    corr_chol.append(cpt) # Compteur pour lequel H_n n'était pas MPD (pour post_process) 

                print("d_n :\n {}".format(d_n))
                
                ## Calcule de la longueur de pas 
                ## Peut nous faire gagner du temps de calcule
                if (sup_g_stagne == True and cpt > 20) : 
#                    if g_sup < 1e-2 and cpt > 150 : # Pour accélérer sortie de programme
#                        # Dans ce cas on suppose qu'on n'aura pas mieux
#                        break
                    
                    alpha = 1. 
                    print("\x1b[1;37;44mgradient stagne : coup de pouce alpha = 1. \x1b[0m")
                    
                    time.sleep(0.7) # Pour avoir le temps de voir qu'il y a eu modif           
                
                # Sinon on ne stagne pas : procédure normale Armijo et Strong Wolf condition (Nocedal and Wright)
                else :  
                    alpha, al2_cor = self.backline_search(J, grad_J, g_n, beta_n ,d_n ,cpt, g_sup, rho=1e-2,c=0.5, w_pm = 0.9)
                    if al2_cor  :
                        al2_lst.append(cpt)
                        # Armijo et Strong Wolf verfiées (obsolète)
                        
                ## Calcule des termes n+1
                dbeta_n =  alpha*d_n
                beta_nNext = beta_n + dbeta_n  # beta_n - alpha*d_n              

                g_nNext =   grad_J(beta_nNext)
                # On construit s_nNext et y_nNext conformément au BFGS
                s_nNext =   (beta_nNext - beta_n)
                y_nNext =   g_nNext - g_n

                ## Pour la première itération on peut prendre (voir Nocedal and Wright (page chapitre)) :
                if cpt == 0 :
                    fac_H = np.dot(y_nNext[np.newaxis, :], s_nNext[:, np.newaxis])
                    fac_H /= np.dot(y_nNext[np.newaxis, :], y_nNext[:, np.newaxis])
                    
                    H_n_inv *= fac_H
                
                # Incrémentation de H_n conformément au BFGS (Nocedal and Wright et scipy)
                H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)
                self.debug["curr_hess_%s" %(str(cpt))] = H_nNext_inv
                
                # Calcule des résidus
                err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
                err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
                
                # Erreur sur J entre l'ancien beta et le nouveau                 
                err_j    =   J(beta_nNext) - J(beta_n)
                
                if verbose == True :
                    print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                    print("err_beta = {} cpt = {}".format(err_beta, cpt))
                    print ("err_hess = {}".format(err_hess))
                
                # Implémentation de liste pour vérifier si besoin
                self.alpha_lst.append(alpha)
                err_hess_lst.append(err_hess) 
                err_beta_lst.append(err_beta)
                dir_lst.append(np.linalg.norm(d_n, 2))
                
                print("\n")
                cpt +=  1    
                # n --> n+1 si non convergence, sort de la boucle sinon 
            
            plt.figure("Evolution de l'erreur Tinf=%d" %(T_inf))
            figgg = plt.gca()
            figgg.set_yscale('log')
            plt.legend(["loglog "+r"$||\nabla_k J(\beta_k)||$ " +\
                        "Vs Iterations; "+ r"$T_{\infty} = %d$" %T_inf])
            
            ######################
            ##-- Post Process --##
            ######################
            H_last  =   H_nNext_inv
            g_last  =   g_nNext
            beta_last=  beta_nNext
            d_n_last = d_n
            
            # On remplit le dictionnaire dont les entrées sont écrites dans le carnet de bord (voir write_logbook)
            logout_last = dict()
            logout_last["cpt_last"]    =   cpt   
            logout_last["J(beta_last)"]=   J(beta_last)
            logout_last["beta_last"]   =   beta_last
            logout_last["g_sup_max"]   =   g_sup
            logout_last["g_last"]      =   g_last
            
            logout_last["Corr_chol"]   =   len(corr_chol)
            logout_last["Residu_hess"] =   err_hess
            logout_last["Residu_beta"] =   err_beta
            
            # Affichage récapitulatif pour se rassurer ou ...            
            print ("\x1b[1;35;47mFinal Sup_g = {}\nFinal beta = {}\nFinal direction {}\x1b[0m".format(\
                g_sup, beta_last, d_n_last))

#            rc('text', usetex=False)
            ax[1].cla()
            # Tracés beta_last et grad_J(beta_last)
            ax[0].plot(self.line_z, beta_last, color=c)
            ax[1].plot(self.line_z, g_last, color=c)
            
            ax[0].legend(["Last "+r"$\beta;\ T_{\infty} = %d$" %T_inf], fontsize=11)
#            rc('text', usetex=True)
            ax[1].legend(["Last "+r"$\nabla J(\beta});\ T_{\infty} =%d$" % T_inf], fontsize=11)
#            rc('text', usetex=False)
            
            leg_0 = ax[0].get_legend()
            leg_1 = ax[1].get_legend()
            
            leg_0.legendHandles[-1].set_color(c)
            leg_1.legendHandles[-1].set_color(c)
            
            leg_0.legendHandles[-1].set_linewidth(2.0)
            leg_1.legendHandles[-1].set_linewidth(2.0)
            
            plt.pause(0.001)
            
            # Construction de la C_bmap
            try :
                R   =   np.linalg.cholesky(H_last)
            except np.linalg.LinAlgError : # i.e si H_last n'est pas MPD -> modification
                H_last = self.cholesky_for_MPD(H_last, fac = 5.)
                corr_chol.append(cpt)
                R   =   H_last
            
            bfgs_adj_bmap[sT_inf]   =   beta_last # beta_map
            bfgs_adj_grad[sT_inf]   =   g_last  
            
            bfgs_adj_hessinv[sT_inf]    =   H_last# Hessienne inverse finale  
            bfgs_adj_cholesky[sT_inf]   =   R
            
            bfgs_adj_bf[sT_inf]     =   bfgs_adj_bmap[sT_inf] + np.dot(R, s) # beta_finale voir après 
            
            # On utilise une fonction codée plus haut : pd_write_csv 
            # On écrit la covariance a posteriori dans un fichier pour l'utiliser dans le ML
            write_cov = osp.join(osp.abspath("./data/post_cov"), "adj_post_cov_%s_%s.npy" %(self.cov_mod, sT_inf))
            if osp.exists(write_cov) :
                add_title = 1
                if osp.exists(osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy") == False :
                    write_cov = osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy"

                # Si on rentre dans le prochaine boucle, le fichier est de la forme T_inf_nombre, reste à savoir à combien il en est
                while osp.exists(osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy") :
                    add_title += 1
                write_cov = osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy"
                
                print write_cov
            
            np.save(write_cov, pd.DataFrame(bfgs_adj_cholesky[sT_inf]))
            print("%s written" %(write_cov))

            beta_var = []
            sigma_post = []
            
            # On va à présent faire des tirages (sample) à partir de beta_map et de Cholesky
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append( bfgs_adj_bmap[sT_inf] + np.dot(R, s) ) # équation 11
            
            beta_var.append(bfgs_adj_bf[sT_inf]) # Pour faire 250, on rajoute beta_final
            # Beta var est une liste de listes. Chacune de ces liste mesure self.N_discr - 2 et contient la valeur
            # les samples tirées plus haut.
            # Pour les tracés, on calcule les minimum et maximum de toutes ces listes pour chaque point.
            
            for i in range(self.N_discr-2) :
                bfgs_adj_mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                bfgs_adj_maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                # Les clés sont moches mais permettent de garder un ordre de 0 à self.N_discr - 2
                sigma_post.append(np.std([j[i] for j in beta_var])) 
                # Pour une itération, i (point de discrétisation) est fixé et on parcours les liste contenues dans
                # beta_var. De cette façon on raisonne sur le i-ème élément de tous les samples.
                # On calcule également l'écart type de chaque point.  
                 
            bfgs_adj_sigma_post[sT_inf] = sigma_post 
            
            # On rassemble les valeurs des mins relatives à tous les points de discrétisation dans une seule liste
            bfgs_adj_mins_lst =  [bfgs_adj_mins["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]   
            bfgs_adj_maxs_lst =  [bfgs_adj_maxs["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]
            
            # Puis on les garde pour la température en cours
            bfgs_adj_mins_dict[sT_inf] = bfgs_adj_mins_lst 
            bfgs_adj_maxs_dict[sT_inf] = bfgs_adj_maxs_lst
            
            plt.legend(loc="best") # légende des figures tracé plus haut
            
#            try :
#                # On va récapituler différentes évolutions au cours de l'optimization
#                fiig, axxes = plt.subplots(2,2,figsize=(8,8))
#                
#                axxes[0][0].set_title("alpha vs iterations T_inf %d " %(T_inf))
#                axxes[0][0].plot(range(cpt), self.alpha_lst, marker='o',\
#                                            linestyle='none', markersize=8)
#                axxes[0][0].set_xlabel("Iterations")
#                axxes[0][0].set_ylabel("alpha")
#                
#                axxes[0][1].set_title("err_hess vs iterations ")
#                axxes[0][1].plot(range(cpt), err_hess_lst, marker='s',\
#                                            linestyle='none', markersize=8)
#                axxes[0][1].set_xlabel("Iterations")
#                axxes[0][1].set_ylabel("norm(H_nNext_inv - H_n_inv, 2)")
#                
#                axxes[1][0].set_title("err_beta vs iterations ")
#                axxes[1][0].plot(range(cpt), err_beta_lst, marker='^',\
#                                            linestyle='none', markersize=8)
#                axxes[1][0].set_xlabel("Iterations")
#                axxes[1][0].set_ylabel("beta_nNext - beta_n")            
#                
#                axxes[1][1].set_title("||d_n|| vs iterations")
#                axxes[1][1].plot(range(cpt), dir_lst, marker='v',\
#                                            linestyle='none', markersize=8)
#                axxes[1][1].set_xlabel("Iteration")
#                axxes[1][1].set_ylabel("Direction")            
#            
#            except ValueError :
#                break
#            
            # On précise que le cas en cours a été traité. On pourra alors le tracé (voir class_functions_aux.py subplot_cst) et l'écrire
            self.bool_method["adj_bfgs_"+sT_inf] = True
            
            # On l'écrit
            f = open(self.logout_title, "a")
            f.write("\nADJ_BFGS\n")
            f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(sT_inf))
            for item in logout_last.iteritems() :
                f.write("{} = {} \n".format(item[0], item[1])) # Voir adjoint_bfgs dans la section Post Process
            self.bool_written["adj_bfgs_"+sT_inf] == True
            f.close()
            
            # On écrit les erreurs 
            f = open(self.err_title, "a")
            f.write("\nADJ_BFGS\n")
            f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(sT_inf))
            f.write("\nErreur grad\tErreur Hess\tErreur Beta\n")
            for g,h,j in zip(sup_g_lst, err_hess_lst, err_beta_lst) :
                f.write("{:.7f} \t {:.7f} \t {:.7f}\n".format(g, h, j))
            f.close()
            
        ## Fin boucle sur température 
        # Finalisation des fichiers
        for f in {open(self.err_title, "a"), open(self.logout_title, "a")} :
            f.write("\nFin de la simulation")
            f.close()
        
        ##############################
        ##-- Passages en attribut --##
        ##############################
        
        # Passage en attribut des dictionnaires comprenant les résultats respectifs pour chaque val de T_inf_lst
        self.bfgs_adj_bf     =   bfgs_adj_bf
        self.bfgs_adj_bmap   =   bfgs_adj_bmap
        self.bfgs_adj_grad   =   bfgs_adj_grad
        self.bfgs_adj_gamma  =   bfgs_adj_gamma
        
        self.bfgs_adj_hessinv=  bfgs_adj_hessinv
        self.bfgs_adj_cholesky= bfgs_adj_cholesky
        
        self.bfgs_adj_mins_dict   =   bfgs_adj_mins_dict
        self.bfgs_adj_maxs_dict   =   bfgs_adj_maxs_dict
        self.bfgs_adj_sigma_post     =   bfgs_adj_sigma_post
        
        # obsolète et inutile mais sait-on jamais
        self.al2_lst    =    al2_lst
        self.corr_chol  =   corr_chol
        
        # mise a jour des figures
        plt.pause(0.001)
        #########
        #- Fin -#
        #########
###---------------------------------------------------## 
    def adjoint_circle(self, inter_plot=False, verbose = False) : 
        """
        inter_plot to see the evolution of the inference
        verbose to print different informqtion during the optimization
        """
        print("Début de l\'optimisation maison\n")
        if self.bool_method["stat"] == False : self.get_prior_statistics() 
        
        self.debug = dict()
        # Le code a été pensé pour être lancé avec plusieurs valeurs de T_inf dans la liste T_inf_lst.
        # On fonctionne donc en dictionnaire pour stocker les valeurs importantes relatives à la température en 
        # cours. De cette façon, on peut passer d'une température à une autre, et donc recommencer une optimisation 
        # pour T_inf différente, sans craindre de perdre les résultats de la T_inf précédente

        bfgs_adj_grad,   bfgs_adj_gamma     =   dict(), dict()
        bfgs_adj_bmap,   bfgs_adj_bf        =   dict(), dict()
        bfgs_adj_hessinv,bfgs_adj_cholesky  =   dict(), dict()
        
        bfgs_adj_mins,   bfgs_adj_maxs  =   dict(),  dict()
        
        bfgs_adj_mins_dict, bfgs_adj_maxs_dict = dict(), dict()
        bfgs_adj_sigma_post  = dict()
        
        self.too_low_err_hess = dict()
        sup_g_stagne = False

        s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0]) # vecteur aléatoire
        
        c = lambda ck, k : 1. - ck/float(k)**5 if k!= 0 else 1.
        
        for T_inf in self.T_inf_lst :
            sT_inf      =   "T_inf_%d" %(T_inf) # Clé pour simplifier
            sigmas      =   np.sqrt(np.diag(self.cov_obs_dict[sT_inf])) #Std obs
            curr_d      =   self.T_obs_mean[sT_inf] # Temperature moyenne sur les self.num_real tirages
            
            # On récupère  les covariances 
            cov_obs     =   self.cov_obs_dict[sT_inf] if self.cov_mod=='diag' else\
                            self.full_cov_obs_dict[sT_inf]
            
            cov_pri     =   self.cov_pri_dict[sT_inf]
            
            inv_cov_pri =   np.linalg.inv(cov_pri)  
            inv_cov_obs =   np.linalg.inv(cov_obs)
            
            # On construit la fonction de coût en trois temps
            J_1 =   lambda beta :\
            0.5*np.dot(np.dot(curr_d - self.h_beta(beta, T_inf).T, inv_cov_obs), (curr_d - self.h_beta(beta, T_inf)))
            
            J_2 =   lambda beta :\
            0.5*np.dot(np.dot((beta - self.beta_prior).T, inv_cov_pri), (beta - self.beta_prior))   
                                        
            ## Fonction de coût
            J = lambda beta : J_1(beta) + J_2(beta)  
            
            # Calcule de dJ/dbeta avec méthode adjoint equation (13) avec (12)
            grad_J = lambda beta :\
            np.dot(self.PSI(beta, T_inf), np.diag(self.DR_DBETA(beta,T_inf)) ) + self.DJ_DBETA(beta ,T_inf)
            
            err_beta = err_hess = err_j = 1            
            cpt, cptMax =   0, self.cpt_max_adj
            
            sup_g_lst = [] # Pour tester si la correction a stagné
            corr_chol = [] # Pour comptabiliser les fois ou la hessienne n'était pas définie positive
            al2_lst  =  [] # Comptabilise les fois où les DEUX conditions de Wolfe sont vérifiés
            
            print("J(beta_prior) = {} \t T_inf = {}".format(J(self.beta_prior), T_inf))
            
            ########################
            ##-- Initialisation --##
            ########################

            beta_n  =   self.beta_prior
            g_n     =   grad_J(beta_n)
            
            g_sup = np.linalg.norm(g_n, np.inf)
            sup_g_lst.append(np.linalg.norm(g_n, np.inf))
            
            print ("\x1b[1;37;44mSup grad : %f \x1b[0m" %(np.linalg.norm(g_n, np.inf))) # Affichage surligné bleu

            # Hessienne (réinitialisé plus tard)
            H_n_inv =   np.eye(self.N_discr-2)
            Ak = H_n_inv
            self.debug["first_hess"] = H_n_inv
            
            # Tracé des différentes figures (Evolutions de béta et du gradient en coursc d'optimization)
            fig, ax = plt.subplots(1,2,figsize=(13,7))
            ax[0].plot(self.line_z, beta_n, label="beta_prior")
            ax[1].plot(self.line_z, g_n,    label="gradient prior")
            
            self.alpha_lst, err_hess_lst, err_beta_lst = [], [], []
            dir_lst =   []
            c_nPrev = 1.
            ######################
            ##-- Optimisation --##
            ######################
            
            while (cpt<cptMax) and g_sup > self.g_sup_max :
                if cpt > 0 :
                ########################
                ##-- Incrementation --##
                ######################## 
                    beta_n  =   beta_nNext
                   
                    g_n     =   g_nNext
                    g_sup   =   np.linalg.norm(g_n, np.inf) # norme infini du gradient
                    
                    H_n_inv =   H_nNext_inv
                    c_nPrev = c_n
                    
                    #MAJ des figures avec nouveaux tracés
                    plt.figure("Evolution de l'erreur %s" %(sT_inf))
                    plt.scatter(cpt, g_sup, c='black')
                    if inter_plot == True :
                        plt.pause(0.05)

                    ax[0].plot(self.line_z, beta_n, label="beta cpt%d_%s" %(cpt, sT_inf))
                    ax[1].plot(self.line_z, g_n, label="grad cpt%d" %(cpt), marker='s')
                    
                    # MAJ de la liste des gradient.                      
                    sup_g_lst.append(g_sup)
                    if len(sup_g_lst) > 6 :
                        lst_ = sup_g_lst[(len(sup_g_lst)-5):] # Prend les 5 dernières valeurs des sup_g
                        mat = [[abs(i - j) for i in lst_] for j in lst_] # Matrices des différences val par val
                        sup_g_stagne = (np.linalg.norm(mat, 2) <= 1e-2) # True ? -> alpha = 1 sinon backline_search
                    
                    # Affichage des données itérations précédentes, initialisant celle à venir
                    print("Compteur = %d" %(cpt))
                    print("\x1b[1;37;44mSup grad : {}\x1b[0m".format(g_sup))
                    print("Stagne ? {}".format(sup_g_stagne))
                    
                    if verbose == True :
                        print ("beta cpt {}:\n{}".format(cpt,beta_n))
                        print("grad n = {}".format(g_n))                            
                        print("beta_n = \n  {} ".format(beta_n))
                        print("cpt = {} \t err_beta = {} \t err_hess = {}".format(cpt, \
                                                           err_beta, err_hess) )
                        print ("Hess cpt {}:\n{}".format(cpt, H_n_inv))
                #################
                ##-- Routine --##
                #################
                # Calcule : -np.dot(grad_J(bk), d_n). 
                # GD pour Gradient Descent.  Doit être négatif (voir après et Nocedal et Wright)
                GD = lambda H_n_inv :  -np.dot(g_n[np.newaxis, :],\
                                    np.dot(H_n_inv, g_n[:, np.newaxis]))[0,0]                
                ## Calcule de la direction 
                A_k = self.aij_circle(H_n_inv)
                c_n = c(c_nPrev, cpt)               
                Bk = (1.-c_n)*A_k + c_n*H_n_inv
                
                if self.positive_definite_test(Bk) :
                    D_k = B_k
                else :
                    D_k = A_k
                
                d_n = - np.dot(D_k, g_n)
                test_ =(GD(D_k) < 0) ## Booléen GD ou pas ?
                
                print("d_n descent direction : {}".format(test_))    
                

                print("d_n :\n {}".format(d_n))
                
                ## Calcule de la longueur de pas 
                ## Peut nous faire gagner du temps de calcule
                if (sup_g_stagne == True and cpt > 20) : 
#                    if g_sup < 1e-2 and cpt > 150 : # Pour accélérer sortie de programme
#                        # Dans ce cas on suppose qu'on n'aura pas mieux
#                        break
                    
                    alpha = 1. 
                    print("\x1b[1;37;44mgradient stagne : coup de pouce alpha = 1. \x1b[0m")
                    
                    time.sleep(0.7) # Pour avoir le temps de voir qu'il y a eu modif           
                
                # Sinon on ne stagne pas : procédure normale Armijo et Strong Wolf condition (Nocedal and Wright)
                else :  
                    alpha, al2_cor = self.backline_search(J, grad_J, g_n, beta_n ,d_n ,cpt, g_sup, rho=1e-2,c=0.5, w_pm = 0.9)
                    if al2_cor  :
                        al2_lst.append(cpt)
                        # Armijo et Strong Wolf verfiées (obsolète)
                        
                ## Calcule des termes n+1
                dbeta_n =  alpha*d_n
                beta_nNext = beta_n + dbeta_n  # beta_n - alpha*d_n              

                g_nNext =   grad_J(beta_nNext)
                # On construit s_nNext et y_nNext conformément au BFGS
                s_nNext =   (beta_nNext - beta_n)
                y_nNext =   g_nNext - g_n

                ## Pour la première itération on peut prendre (voir Nocedal and Wright (page chapitre)) :
                if cpt == 0 :
                    fac_H = np.dot(y_nNext[np.newaxis, :], s_nNext[:, np.newaxis])
                    fac_H /= np.dot(y_nNext[np.newaxis, :], y_nNext[:, np.newaxis])
                    
                    H_n_inv *= fac_H
                
                # Incrémentation de H_n conformément au BFGS (Nocedal and Wright et scipy)
                H_nNext_inv = self.Next_hess(H_n_inv, y_nNext, s_nNext)
                self.debug["curr_hess_%s" %(str(cpt))] = H_nNext_inv
                
                # Calcule des résidus
                err_beta =   np.linalg.norm(beta_nNext - beta_n, 2)
                err_hess =   np.linalg.norm(H_n_inv - H_nNext_inv, 2)
                
                # Erreur sur J entre l'ancien beta et le nouveau                 
                err_j    =   J(beta_nNext) - J(beta_n)
                
                if verbose == True :
                    print ("J(beta_nNext) = {}\t and err_j = {}".format(J(beta_nNext), err_j))
                    print("err_beta = {} cpt = {}".format(err_beta, cpt))
                    print ("err_hess = {}".format(err_hess))
                
                # Implémentation de liste pour vérifier si besoin
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
            
            # On remplit le dictionnaire dont les entrées sont écrites dans le carnet de bord (voir write_logbook)
            logout_last = dict()
            logout_last["cpt_last"]    =   cpt   
            logout_last["J(beta_last)"]=   J(beta_last)
            logout_last["beta_last"]   =   beta_last
            logout_last["g_sup_max"]   =   g_sup
            logout_last["g_last"]      =   g_last
            
            logout_last["Corr_chol"]   =   len(corr_chol)
            logout_last["Residu_hess"] =   err_hess
            logout_last["Residu_beta"] =   err_beta
            
            # Affichage récapitulatif pour se rassurer ou ...            
            print ("\x1b[1;35;47mFinal Sup_g = {}\nFinal beta = {}\nFinal direction {}\x1b[0m".format(\
                g_sup, beta_last, d_n_last))
            
            # Tracés beta_last et grad_J(beta_last)
            ax[1].plot(self.line_z, g_last, label="gradient last")
            ax[0].plot(self.line_z, beta_last, label="beta_n last")
            
            # Construction de la C_bmap
            try :

                R   =   np.linalg.cholesky(H_last)
            except np.linalg.LinAlgError : # i.e si H_last n'est pas MPD -> modification
                H_last = self.cholesky_for_MPD(H_last, fac = 5.)
                corr_chol.append(cpt)
                R   =   H_last
            
            bfgs_adj_bmap[sT_inf]   =   beta_last # beta_map
            bfgs_adj_grad[sT_inf]   =   g_last  
            
            bfgs_adj_hessinv[sT_inf]    =   H_last# Hessienne inverse finale  
            bfgs_adj_cholesky[sT_inf]   =   R
            
            bfgs_adj_bf[sT_inf]     =   bfgs_adj_bmap[sT_inf] + np.dot(R, s) # beta_finale voir après 
            
            # On utilise une fonction codée plus haut : pd_write_csv 
            # On écrit la covariance a posteriori dans un fichier pour l'utiliser dans le ML
            write_cov = osp.join(osp.abspath("./data/post_cov"), "adj_post_cov_%s_%s.npy" %(self.cov_mod, sT_inf))
            if osp.exists(write_cov) :
                add_title = 1
                if osp.exists(osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy") == False :
                    write_cov = osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy"

                # Si on rentre dans le prochaine boucle, le fichier est de la forme T_inf_nombre, reste à savoir à combien il en est
                while osp.exists(osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy") :
                    add_title += 1
                write_cov = osp.splitext(write_cov)[0] + "_%s" %(str(add_title)) + ".npy"
                
                print write_cov
            
            np.save(write_cov, pd.DataFrame(bfgs_adj_cholesky[sT_inf]))
            print("%s written" %(write_cov))

            beta_var = []
            sigma_post = []
            
            # On va à présent faire des tirages (sample) à partir de beta_map et de Cholesky
            for i in range(249):
                s = np.asarray(self.tab_normal(0,1,self.N_discr-2)[0])
                beta_var.append( bfgs_adj_bmap[sT_inf] + np.dot(R, s) ) # équation 11
            
            beta_var.append(bfgs_adj_bf[sT_inf]) # Pour faire 250, on rajoute beta_final
            # Beta var est une liste de listes. Chacune de ces liste mesure self.N_discr - 2 et contient la valeur
            # les samples tirées plus haut.
            # Pour les tracés, on calcule les minimum et maximum de toutes ces listes pour chaque point.
            
            for i in range(self.N_discr-2) :
                bfgs_adj_mins[sT_inf + str("{:03d}".format(i))] = (min([j[i] for j in beta_var]))
                bfgs_adj_maxs[sT_inf + str("{:03d}".format(i))] = (max([j[i] for j in beta_var]))
                # Les clés sont moches mais permettent de garder un ordre de 0 à self.N_discr - 2
                sigma_post.append(np.std([j[i] for j in beta_var])) 
                # Pour une itération, i (point de discrétisation) est fixé et on parcours les liste contenues dans
                # beta_var. De cette façon on raisonne sur le i-ème élément de tous les samples.
                # On calcule également l'écart type de chaque point.  
                 
            bfgs_adj_sigma_post[sT_inf] = sigma_post 
            
            # On rassemble les valeurs des mins relatives à tous les points de discrétisation dans une seule liste
            bfgs_adj_mins_lst =  [bfgs_adj_mins["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]   
            bfgs_adj_maxs_lst =  [bfgs_adj_maxs["T_inf_%d%03d" %(T_inf, i) ]\
                                            for i in range(self.N_discr-2)]
            
            # Puis on les garde pour la température en cours
            bfgs_adj_mins_dict[sT_inf] = bfgs_adj_mins_lst 
            bfgs_adj_maxs_dict[sT_inf] = bfgs_adj_maxs_lst
            
            plt.legend(loc="best") # légende des figures tracé plus haut
            
            try :
                # On va récapituler différentes évolutions au cours de l'optimization
                fiig, axxes = plt.subplots(2,2,figsize=(8,8))
                
                axxes[0][0].set_title("alpha vs iterations T_inf %d " %(T_inf))
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
            
            # On précise que le cas en cours a été traité. On pourra alors le tracé (voir class_functions_aux.py subplot_cst) et l'écrire
            self.bool_method["adj_bfgs_"+sT_inf] = True
            
            # On l'écrit
            f = open(self.logout_title, "a")
            f.write("\nADJ_BFGS\n")
            f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(sT_inf))
            for item in logout_last.iteritems() :
                f.write("{} = {} \n".format(item[0], item[1])) # Voir adjoint_bfgs dans la section Post Process
            self.bool_written["adj_bfgs_"+sT_inf] == True
            f.close()
            
            # On écrit les erreurs 
            f = open(self.err_title, "a")
            f.write("\nADJ_BFGS\n")
            f.write("\n\x1b[1;37;43mMethod status for %s: \x1b[0m\n" %(sT_inf))
            f.write("\nErreur grad\tErreur Hess\tErreur Beta\n")
            for g,h,j in zip(sup_g_lst, err_hess_lst, err_beta_lst) :
                f.write("{:.7f} \t {:.7f} \t {:.7f}\n".format(g, h, j))
            f.close()
            
        ## Fin boucle sur température 
        # Finalisation des fichiers
        for f in {open(self.err_title, "a"), open(self.logout_title, "a")} :
            f.write("\nFin de la simulation")
            f.close()
        
        ##############################
        ##-- Passages en attribut --##
        ##############################
        
        # Passage en attribut des dictionnaires comprenant les résultats respectifs pour chaque val de T_inf_lst
        self.bfgs_adj_bf     =   bfgs_adj_bf
        self.bfgs_adj_bmap   =   bfgs_adj_bmap
        self.bfgs_adj_grad   =   bfgs_adj_grad
        self.bfgs_adj_gamma  =   bfgs_adj_gamma
        
        self.bfgs_adj_hessinv=  bfgs_adj_hessinv
        self.bfgs_adj_cholesky= bfgs_adj_cholesky
        
        self.bfgs_adj_mins_dict   =   bfgs_adj_mins_dict
        self.bfgs_adj_maxs_dict   =   bfgs_adj_maxs_dict
        self.bfgs_adj_sigma_post     =   bfgs_adj_sigma_post
        
        # obsolète et inutile mais sait-on jamais
        self.al2_lst    =    al2_lst
        self.corr_chol  =   corr_chol
        #########
        #- Fin -#
        #########
###---------------------------------------------------##   
######                                            ######
######    Fonctions pour calculer la Hessienne    ######
######                                            ######
##----------------------------------------------------##
##----------------------------------------------------## 
    def Next_hess(self, prev_hess_inv, y_nN, s_nN) :
        """
        Procedure close to the scipy's one. 
        It is used in adjoint_bfgs function
        """
        # Comme Scipy. De manière générale on évite de diviser par zéro
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
    def backline_search(self, J, g_J, djk, xk, dk, cpt_ext, g_sup, rho=1., c=0.5, w_pm = 0.9) :
        # c is the armijo parameter
        # w_pm is the wolfe parameter
        alpha = alpha_lo = alpha_hi = 1.
        correction = False
        bool_curv  = False
        
        self.warn = "go on"
        
        # Condition d'Armijo
        armi  = lambda alpha : (J(xk + alpha*dk)) <=\
                (J(xk) + c * alpha * np.dot(djk.T, dk)) 
        # Strong Wolf Condition
        curv  = lambda alpha : (np.linalg.norm(g_J(xk + alpha*dk))) <=\
                (w_pm*np.linalg.norm(djk,2))  
        
        cpt, cptmax = 0, 10
        
        # On conmmence avec le backline classique qui consiste à chercher la alpha vérifiant condition d'Armijo
        # Algo inspiré (si ce n'est calqué) de l'algo 3.1 de Nocedal and Wright
        while (armi(alpha) == False) and cpt< cptmax and self.warn=="go on" :
            alpha_lo =  alpha
            alpha   *=  rho
            cpt     +=  1
            print (alpha,  armi(alpha))
            alpha_hi =  alpha
#            if alpha <= 1.e-14 :
#                self.warn = "out"
#                break
                # inutile de continuer dans ces cas là
        print("alpha = {}\t cpt = {}".format(alpha, cpt))
        print("Armijo = {}\t Curvature = {}".format(armi(alpha), curv(alpha)))
        
        if (((alpha <= 1e-7 and cpt_ext > 80)) and g_sup < 5000) and self.warn == "go on":
            temp = alpha
            if alpha <= 1e-10 :
                alpha = 5e-4 # Ceci a été rajouté pour éviter les explosions d'une itérations à l'autre quitte à laisser le calcul être plus long
            else : 
                alpha = 1.
            print("\x1b[1;37;44mCompteur = {} Alpha from {} to {}\x1b[0m".format(cpt_ext, temp, alpha))
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

#        if self.warn == "out" and armi(alpha) == False :
#            alpha = 1e-8 ## Au pire on recentrera avec l'itération suivante mais on veut éviter l'explosion
#            print warnings.warn("Alpha = 1e-8 previously under 1e-14 ")
          
        if armi(alpha) == True and curv(alpha) == True :
            print("\x1b[1;37;43mArmijo = True \t Curvature = True \x1b[0m") 
            # On sait qu'ils sont True, on gagne du temps en ne recalculant pas armi(alpha) et curv(alpha)
        
        return alpha, correction
##----------------------------------------------------##
    def aij_circle(self, Bk, delta=1):
        Bk_t = Bk.transpose()
        a_ij = a_ji = np.zeros((len(self.line_z),len(self.line_z)), dtype = np.float)
        
        # construction a_ij
        for i in range(len(self.line_z) ) :
            for j in range(len(self.line_z) ) :
                if j < i :
                    a_ij[i,j] = Bk[i,j]
                if j == i :
                    sum1 = sum([np.abs(Bk[s, j]) for s in range(i,len(self.line_z))])
                    if j > 1 :
                        sum2 = sum([np.abs(Bk[i, t]) for t in range(j-1)]) 
                    else : 
                        sum2 = 0
                    a_ij[i,j] = 0.5*(sum1 + sum2 + delta)
                if j > i :
                    a_ij[i,j] = 0.

        # Construction de a_ji    
        for i in range(len(  self.line_z)) :
            for j in range(len(  self.line_z)) :
                if j < i :
                    a_ij[i,j] = Bk_t[i,j]
                if j == i :
                    sum1 = sum([np.abs(Bk_t[s, j]) for s in range(i,len( self.line_z) )])
                    if j > 1 :
                        sum2 = sum([np.abs(Bk_t[i, t]) for t in range(j-2)]) 
                    else : 
                        sum2 = 0
                    a_ij[i,j] = 0.5*(sum1 + sum2 + delta)
                if j > i :
                    a_ij[i,j] = 0.
        print np.array([[a_ij[i,j] + a_ji[i,j] for i in range(len(self.line_z))] for j in range(len(self.line_z))])
        return np.array([[a_ij[i,j] + a_ji[i,j] for i in range(len(self.line_z))] for j in range(len(self.line_z))])
        
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
if __name__ == "__main__" :
    import class_functions_aux as cfa #Pour les tracés post-process
    
    parser = cfa.parser()
    
    
    T = Temperature_cst(parser)
#    temp.obs_pri_model()
#    temp.get_prior_statistics()
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
