#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import sys, warnings, argparse

import os
import os.path as osp

import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time
import solvers

import csv

plt.ion()

pathtosave = osp.join(osp.abspath("./data"), "roe")
if osp.exists(pathtosave) == False :
    os.mkdir(pathtosave)

conditions= {'L'    :   1,
             'Nt'   :   250,
             'Nx'   :   100,
             'tf'   :   0.7,
             'f'    :   lambda u : u**2,
             'fprime' : lambda u : u,
             'type_init' : "sin",
             'amp'  :   1.
             }

def parser() :
    parser=argparse.ArgumentParser(description='\
    This parser will be used in several steps both in inference and ML postprocessing\n\
    Each entry is detailed in the help, and each of it has the most common default value. (run ... .py -h)\
    This on is to initialize different aspect of Burger Equation problem')
    ## VaV T_inf
    # Caractéristiques de la simulation voulue          
    parser.add_argument('--Nx', '-Nx', action='store', type=int, default=250, dest='Nx', 
                        help='Define the number of discretization points : default %(default)d \n' )
    parser.add_argument('--N_temp', '-Nt', action='store', type=int, default=100, dest='Nt', 
                        help='Define the number of time steps; default %(default)d \n' )
    parser.add_argument('--domain_length', '-L', action='store', type=int, default=float(1), dest='L',
                        help='Define the length of the domain; default %(default)f \n' )
    parser.add_argument('--tfinal', '-tf', action='store', type=float, default=float(1), dest='tf',\
                        help='Define final time; default %(default)f \n')
    parser.add_argument('--CFL', '-CFL', action='store', type=float, default=float(0.4), dest='CFL',\
                        help='CFL: velocity adimensionned; default %(default)f \n')
    parser.add_argument('--amplitude', '-amp', action='store', type=float, default=0.5, dest="amp",\
                        help='amplitude: the amplitude of sinus or init choc; Value default : %(default)f\n') 
    parser.add_argument('--Iteration_max', '-itmax', action='store', type=int, default=500, dest='itmax', 
                        help='itmax: max iteration in obs or inference : default %(default)d \n' ) 
    parser.add_argument('--n_harmonic', '-harm', action='store', type=int, default=1, dest='harm', 
                        help='harm: harmonic in the sin init : default %(default)d \n' )                   
    parser.add_argument('--phase', '-phase', action='store', type=float, default=0.,  dest='phase',
                        help='phase: choose the phase in the sin init (pi inside) : default %(default)f \n' )
    parser.add_argument('--U_advection', '-U_adv', action='store', type = float, default=1., dest='U_adv',
                        help='U_adv is a advection velocity. default %(default)f \n' )
    
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=10, dest='num_real', 
                        help='Define how many samples of epsilon(T) to draw for exact model. Default to %(default)d\n' )
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', default=1 ,dest='beta_prior',\
                        help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    # Strings
    parser.add_argument('--init_u', '-init_u', action='store', type=str, default='sin', dest='type_init', 
                        help='Choose initial condition on u. Defaut sin\n')
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data/roe/', dest='datapath', 
                        help='Define the directory where the data will be stored and read. Default to %(default)s \n')
    parser.add_argument('--type_J', '-typeJ', action='store', type=str, default="u", dest='typeJ',\
                        help='Define the type of term you want to simulate')
    
    return parser.parse_args()

class Class_Roe_BFGS():
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def __init__(self, parser):
        self.L, self.tf = parser.L, parser.tf
        self.Nx,  self.Nt   =   parser.Nx, parser.Nt, 
        
        self.dx =  float(self.L) / (self.Nx-1)
        self.dt =  float(self.tf) / self.Nt
        
        self.r = self.dt / self.dx
        dt = parser.CFL * self.dx / parser.U_adv
        
        self.CFL = parser.CFL
        
        if dt < self.dt :
            print ("Use of CFL to calculate dt.")
            print ("Originally dt = %6f" % self.dt)
            print ("Now switch to dt = %6f "% dt)
            self.dt = dt
        
        self.amp = parser.amp
        
        self.datapath  = osp.abspath(parser.datapath)
        self.num_real  = parser.num_real
        self.itmax     = parser.itmax
        self.typeJ     = parser.typeJ
        self.cov_mod = "full"      
        self.type_init = parser.type_init
        self.line_x = np.linspace(0, self.L, self.Nx)
        
        self.phase = parser.phase
        self.harm  = parser.harm
        
        self.itmax= parser.itmax
        
        bruits = [0.005 * np.random.randn(self.Nx) for time in range(self.num_real)]
        self.bruits = bruits
        
        if not osp.exists(self.datapath) :
            os.mkdir(self.datapath)
        
        self.cov_path = osp.join(self.datapath, "post_cov")
        self.beta_path = osp.join(self.datapath, "betas")
        self.chol_path = osp.join(self.datapath, "cholesky")
        self.inferred_U = osp.join(self.datapath, "U")
        
        self.U_adv = parser.U_adv
        
        for i in [self.cov_path, self.beta_path, self.chol_path, self.inferred_U] :
            if not osp.exists(i): os.mkdir(i) 
        
        self.parser = parser
        
        self.beta_name = lambda nx, nt, type_i, CFL, it : osp.join(self.beta_path,\
            "beta_Nx:{}_Nt:{}_".format(nx, nt) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
        self.u_name = lambda nx, nt, type_i, CFL, it : osp.join(self.inferred_U,\
            "U_Nx:{}_Nt:{}_".format(nx, nt) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
        self.chol_name = lambda nx, nt, type_i, CFL, it : osp.join(self.chol_path,\
            "chol_Nx:{}_Nt:{}_".format(nx, nt) + "typei:{}_CFL:{}_it:{:03}.npy".format(type_i, CFL, it))
        
        self.beta_prior = [parser.beta_prior for i in range(self.Nx)]
        
        self.sCFL = "%.2f" % self.CFL
        self.samp = "%.2f" % self.amp
        self.sU_adv = "%.2f" % self.U_adv
        self.sphase = "%.2f" % self.phase
        self.sharm = "%.2f" % self.harm
                
        self.sCFL = self.sCFL.replace('.', '_')
        self.samp = self.samp.replace('.', '_')
        self.sU_adv = self.sU_adv.replace('.', '_')
        self.sphase = self.sphase.replace('.', '_')
        self.sharm = self.sharm.replace('.', '_')

        file_name = "roe_inference_case_done.txt"
        file_loc = osp.join(self.datapath, file_name)
        
        if osp.isfile(file_loc) == False :
            ffile = open(file_loc, "w+")
            ffile.write("Nx\t Nt\t CFL\t amp\t U_adv\t init\t harm\t phase\t itmax\n")
            ffile.write("%d\t %d\t %s\t %s\t %s\t %s\t %s\t %s\t %d\n"\
                    %(self.Nx, self.Nt, self.sCFL, self.samp, self.sU_adv, self.type_init, self.sharm, self.sphase, self.itmax))
        else :
            ffile = open(file_loc, "a+")
            ff = csv.reader(ffile, delimiter='\t')
            
            for r in csv.reader(open(file_loc, "a+"), delimiter='\t') :
                curr_line = "%d\t %d\t %s\t %s\t %s\t %s\t %s\t %s\t %d"\
                    %(self.Nx, self.Nt, self.sCFL, self.samp, self.sU_adv, self.type_init, self.sharm, self.sphase, self.itmax)

                rr = [i.replace(' ','') for i in r]
                
                if rr == curr_line.split() :
                    sys.exit("%s\nAlready exists case (see %s)" %(curr_line, file_name))
            
            ffile.write("%d\t %d\t %s\t %s\t %s\t %s\t %s\t %s\t %d\n"\
                    %(self.Nx, self.Nt, self.sCFL, self.samp, self.sU_adv, self.type_init, self.sharm, self.sphase, self.itmax))
        ffile.close()
        
#---------------------------------------------------------------#        
#---------------------------------------------------------------#
    def define_functions(self, f, fprime) :    
        self.f = f
        self.fprime = fprime
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def true_solve(self, u) :
        u_next = np.zeros_like(u)
        for j in range(1, len(self.line_x)-1) :
            u_next[j] = solvers.timestep_roe(u, j, self.CFL, self.f, self.fprime)
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[2]
        
        return u_next
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def u_beta(self, beta, u):
        u_next = np.zeros_like(self.line_x)
        
#        print np.shape(u)
#        print np.shape(beta)
#        print np.shape(u_next)
#        
        for j in range(1, len(self.line_x)-1):
            u_next[j] = u[j] - self.dt*beta[j-1]
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[2]
        
        return u_next 
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def init_u(self, low = 0., high=1.) :
        u_init = np.zeros_like(self.line_x)
        if self.type_init == 'choc' :
            for j, x in enumerate(self.line_x) :
                if abs(x) < self.L/2. :
                    u_init[j] = low
                else :
                    u_init[j] = high
                    
        if self.type_init == 'sin' :
            u_init = self.amp*np.sin(2.*self.harm*np.pi*self.line_x/self.L + np.pi*self.phase)
        
        return u_init
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def obs_res(self, write=False, plot=False, overwrite=False, low=0, high=1.)  :
        
        self.file = lambda Nt, Nx, CFL, amp, U_adv, type_init, harm, phase, it, real : \
                    "Nt%d_Nx%d_CFL%s_amp%s_U_adv%s_%s_harm%s_phase%s_u_it%d_%d.npy" \
                    %(Nt, Nx, CFL, amp, U_adv, type_init, harm, phase, it, real)

        init = self.type_init

        ## Déf des données du probleme 
        if overwrite == True :
            curr = os.getcwd()
            os.chdir(self.datapath)
            os.system("rm *.npy")
            os.chdir(curr)
            print os.getcwd()
            print ("Deleted npy files")
            
            time.sleep(1)
            
        for j, bruit in enumerate(self.bruits) :
            # Initialisation des champs u (boucles while)
            u = self.init_u(low = low, high = high) + bruit
            u_nNext = np.zeros_like(u)
            
            # Tracés figure initialisation : 
            if plot == True :
                plt.figure("Resolution")
                plt.plot(self.line_x, u)
                plt.title("U vs X iteration 0 noise %d" %(j))
                plt.ylim((-self.amp * 1.5, self.amp * 1.5))
                plt.pause(0.01)
                
            if write == True or overwrite == True: 
                filename = osp.join(self.datapath, 
                self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, 0, j)
                                   )
                            
                np.save(filename, u)
            
            t = it = 0
            
            while it <= self.itmax:
                filename = osp.join(self.datapath, 
            self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, it+1, j))

                if osp.exists(filename) == True and overwrite == False:
                    it+=1
                    continue
            
                u_next = self.true_solve(u)
                u = np.copy(u_next)
                
                if write == True or overwrite == True :
                    np.save(filename, u_next)
                
                it += 1
                t += self.dt

                if plot == True :
                    if it % 50 == 0 :
                        plt.clf()
                        plt.plot(self.line_x[0:self.Nx-1], u[0:self.Nx-1], c='k') 
                        plt.grid()
                        plt.title("u vs X, iteration %d bruit %d" %(it, j)) 
                        plt.xticks(np.arange(self.dx, self.L-self.dx, 0.25))
                        print u
                        plt.ylim((-self.amp * 1.5, self.amp * 1.5))
                        plt.pause(0.1)  
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def get_obs_statistics(self, write = True):
        U_moy_obs = dict()
        full_cov_obs_dict = dict()
        diag_cov_obs_dict = dict()
        init = self.type_init
        for it in range(self.itmax) :
            u_sum = np.zeros((self.Nx))

            # Calcul de la moyenne pour l'itération en cours
            for n in range(self.num_real) :
                file_to_get = osp.join(self.datapath, 
                    self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, it+1, j))
                
                u_t_n = np.load(file_to_get)
                for i in range(len(u_t_n)) : u_sum[i] += u_t_n[i] / float(self.num_real)
                
            U_moy_obs["u_moy_it%d" %(it)] = u_sum
            full_cov = np.zeros((self.Nx, self.Nx))        
            
            # Calcul de la covariance associée à l'itération
            full_cov_filename = osp.join(self.cov_path, "full_cov_obs_" +
                self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, it, n))

            diag_cov_filename = osp.join(self.cov_path, "diag_cov_obs_" +  
                self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, it, n))
            
            if osp.exists(full_cov_filename) == True and osp.exists(diag_cov_filename) :
                full_cov_obs_dict["full_cov_obs_it%d"%(it)] = np.load(full_cov_filename) 
                diag_cov_obs_dict["diag_cov_obs_it%d"%(it)] = np.load(diag_cov_filename)
#                print ("Lecture %s" %(cov_filename))
                continue
            
            for n in range(self.num_real) :
                file_to_get = osp.join(self.datapath,
                self.file(self.Nt, self.Nx, self.sCFL, self.samp, self.sU_adv, init, self.sharm, self.sphase, it, n))
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
#---------------------------------------------------------------#
#---------------------------------------------------------------#
    def minimization(self, maxiter, solver="BFGS", step=5):
        fig, axes= plt.subplots(1, 2, figsize = (8,8))
        evol = 0
        if self.stats_done == False :
            print("Get_Obs_Statistics lunched")
            self.get_obs_statistics(True)
        #---------------------------------------------    
        #---------------------------------------------
        beta_n = self.beta_prior
        u_n = self.U_moy_obs["u_moy_it0"]
#        print ("u_n shape = {}".format(np.shape(u_n)))
        
#        alpha = 1.e-4 # facteur de régularisation
        self.opti_obj = dict
        self.beta_n_dict = dict()
        self.U_beta_n_dict = dict()
        self.optimization_time = dict()
        
        t = 0
        reg_fac = 1e-2
        Id = np.eye(self.Nx)
        
        for it in range(self.itmax -1) :
            if it > 0 :
                beta_n = beta_n_opti
                u_n = u_n_beta
                
            t1 = time.time()
            
            print ("it = {}".format(it))
            
            u_obs_nt = self.U_moy_obs["u_moy_it%d" %(it+1)]
            cov_obs_nt = self.full_cov_obs_dict["full_cov_obs_it%d"%(it+1)] # Pour avoir la bonne taille de matrice
            
            if it == 0 :
#                print np.shape(beta_n)
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
            # On cherche beta faisant correspondre les deux solutions au temps it + 1. Le beta final est enfin utilisé pour calculer u_beta au temps it 
#            for i in range(len(beta_n)) : # Un peu de bruit
#                beta_n[i] *= np.random.random()    

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
            np.save(self.beta_name(self.Nx, self.Nt, self.type_init, self.CFL, it), beta_n_opti) 
            np.save(self.u_name(self.Nx, self.Nt, self.type_init, self.CFL, it), u_n_beta)
            
                # Calcule de Cholesky et enregistrement
            hess_beta = optimi_obj_n.hess_inv
            cholesky_beta = np.linalg.cholesky(hess_beta)
            np.save(self.chol_name(self.Nx, self.Nt, self.type_init, self.CFL, it), cholesky_beta)

            # On calcule l'amplitude en utilisant cholesky pour savoir si les calculs sont convergés ou pas             
            sigma = dict()
            mins, maxs = [], []
            
            fields_v = dict()
            
            for j in range(self.Nx) :
                fields_v["%03d" %(j)] = []
            
            cpt = 0
            # Tirage avec s un vecteur aléatoire tirée d'une distribution N(0,1)
            while cpt < 100 :
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
                if evol == 0 :
                    evol = 0
                    for i in [0,1] : 
                        axes[i].clear()

                if evol == 0 :
                    axes[0].plot(self.line_x[:-1], beta_n_opti[:-1], label="iteration %d" %(it), c= "darkred")
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
                
#                plt.savefig("./res_all_T_inf/burger_fig/nu%.4f_CFL%.2f_Nx_%d_InferenceVSTrue_it%d.png" %(self.nu, self.CFL, self.Nx, it))
                
#                evol += 1
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

if __name__ == '__main__':

#   run Roe_inference_burgers.py -L 3 -U 1 -amp 1.3 -itmax 200 -harm 1 -phase 3.141592653589793*0.5

    parser = parser()
    cnt = 0
    for U_adv in [0.3, 0.6, 0.9, 1.2, 1.5] :
        for amp in [0.5, 1, 1.5] :
            for harm in [1,2,3] :
                for phase in [0, 1./6., 1./3., 1./2.] :
                    print "U_adv = %.2f  amp = %.2f  harm = %.2f  phase = %.2f\n" % \
                    (U_adv, amp, harm, phase)
                    
                    if abs(U_adv - 0.3) < 1e-5 or (U_adv - 0.5) < 1e-5 :
                        itmax = 350
                    
                    else :
                        itmax = 200
                        
                    parser.U_adv = U_adv
                    parser.amp = amp
                    parser.harm = harm
                    parser.phase = phase
                    parser.itmax = itmax
                    test = Class_Roe_BFGS(parser)
                    test.define_functions(lambda x : test.U_adv*x, lambda x : test.U_adv)
                    test.obs_res(write=True, plot=True, overwrite=False)
                    
                    del test
                    
                    cnt += 1
    print "%d cases" % cnt
#    test.get_obs_statistics()
    
    
