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

def THparser() :
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
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=3, dest='num_real', 
                        help='Define the number of realizations drawn; default %(default)d \n' )
                        
    parser.add_argument('--domain_length', '-L', action='store', type=int, default=float(3), dest='L',
                        help='Define the length of the domain; default %(default)f \n' )
    parser.add_argument('--CFL', '-CFL', action='store', type=float, default=0.45, dest='CFL', 
                        help='Define this simulations\'s CFL (under 0.5); default %(default)d\n' )
                        
    parser.add_argument('--diffusion_rate', '-nu', action='store', type=float, default=2.5e-5, dest='nu', 
                        help='Define the convection coefficient h \n' )
    
    parser.add_argument("--Umax", "-U", action="store", type=float, default=5., dest="U",
                        help="Define U max. Default: %(default)f m/s \n")

    parser.add_argument("--HMax", "-H", action="store", type=float, default=0.0, dest="H",
                        help="Define H max. Default: %(default)f m/s \n")
    
    parser.add_argument("--reynolds_ratio", "-ratio", action="store", default=0.01, type=float, dest="diff_ratio",
                        help="Define the diffusivity ratio Nu/Lambda. Default: %(default)f \n")
    # Pour l'algorithme
#    parser.add_argument('--delta_t', '-dt', action='store', type=float, default=1e-4, dest='dt', 
#                        help='Define the time step disctretization. Default to %(default).5f \n' )
    parser.add_argument('--Iteration_max', '-itmax', action='store', type=int, default=500, dest='itmax', 
                        help='Define temporal maximum iteration (-itmax) in both solving problems : default %(default)d \n' )
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', default=1 ,dest='beta_prior',\
                        help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    
    # Strings
    parser.add_argument('--init_u', '-init_u', action='store', type=str, default='sin', dest='type_init', 
                        help='Choose initial condition on u. Defaut sin\n')
    parser.add_argument('--datapath', '-dp', action='store', type=str, default='./data/thomas_dataset/', dest='datapath', 
                        help='Define the directory where the data will be stored and read. Default to %(default)s \n')
    parser.add_argument('--logbook_path', '-p', action='store', type=str, default='./logbooks/thomas/', dest='logbook_path', 
                        help='Define the logbook\'s path. Default to %(default)s \n')
   
    return parser.parse_args()
    
    
    
class Thomas_class() :
##---------------------------------------------------    
    def __init__(self, parser) :
        
        self.Nx = parser.Nx  
        self.Nt = parser.Nt 
        self.CFL = parser.CFL
        self.nu  = parser.nu
        
        L = parser.L
        dx = L/(self.Nx-1)
        
        dt = {"dt_v" : self.CFL / self.nu * dx**2,
              "dt_l" : self.CFL*dx}
              
        if dt["dt_v"] < dt["dt_l"] :
            dt = dt["dt_v"]
            print ("dt visqueux")
        else :
            dt = dt["dt_l"]
            print ("dt lineaire")
        
        print ("dt = ", dt)
        
        self.fac = self.nu*dt/dx**2
        tf = self.Nt * dt
        
        self.r = 0.5 * dt / dx
        
        self.line_x = np.arange(0,L+dx, dx)
        
        self.U = parser.U
        
        if np.abs(parser.H) < 1e-5 :
            self.H = 0.2*self.U
            print ("Hmax = %f" % self.H)
        
        else :
            self.H = parser.H
        
        self.diff_ratio = parser.diff_ratio
        self.mag_diff = self.nu / self.diff_ratio 
        
        
        self.nu_reynolds = self.U * dx / self.nu
        self.mag_reynolds = self.H * dx / self.mag_diff
        
        self.logbook_path=  osp.abspath(parser.logbook_path)
        self.datapath    =  osp.abspath(parser.datapath)
        self.num_real    =  parser.num_real
        self.itmax       =  parser.itmax
        
        self.L,  self.tf  = L , tf
        self.dx, self.dt  = dx, dt
        
        self.nu_str = str(self.nu).replace(".","_")
        self.CFL_str = str(self.CFL).replace(".","_")
        self.type_init = parser.type_init
        
        self.beta_prior = np.asarray([parser.beta_prior for i in range(self.Nx)])
        
        if osp.exists(self.datapath) == False :
            os.makedirs(self.datapath)
        
        if osp.exists(self.logbook_path) == False :
            os.makedirs(self.logbook_path)
        
        self.parser = parser
        
        self.sum_up = {"Nx" : self.Nx,
                       "Nt" : self.Nt,
                       "nu" : self.nu,
                       "Nu_reynolds" : self.nu_reynolds,
                       "mag_diff" : self.mag_diff,
                       "mag_reynolds" : self.mag_reynolds,
                       "Diffusivity ratio" : self.diff_ratio,
                       "itmax" : self.itmax,
                       "Domaine length" : self.L,
                       "CFL": self.CFL,
                       "UMax" : self.U,
                       "HMax": self.H}
                       
##---------------------------------------------------         
##--------------------------------------------------- 
##--------------------------------------------------- 

    def init_fields(self, type_init,  phase = 0, plot=False) :
        if type_init == "random" :
            U_intervals = np.linspace(-self.U, self.U, 10000)
            H_intervals = np.linspace(-self.H, self.H, 10000)
        
            u_init_field, h_init_field = [], []
        
            for i in range(1, self.Nx-1): 
                u_init_field.append(np.random.choice(U_intervals))
                h_init_field.append(np.random.choice(H_intervals))
            
            u_init_field.insert(0, 0.)
            h_init_field.insert(0, 0.)
            
            u_init_field.insert(len(u_init_field), 0.0)
            h_init_field.insert(len(u_init_field), 0.0)

        if type_init == "sin" :
            u_init_field = self.U*np.sin(2*np.pi*self.line_x/self.L + phase)
            h_init_field = self.H*np.sin(2*np.pi*self.line_x/self.L + phase)
            
        if plot :
            plt.figure()
            plt.plot(self.line_x, u_init_field, label="u init", color='blue')
            plt.plot(self.line_x, h_init_field, label="h init", color='red', linestyle='--')
            plt.legend(loc='best')
        
        self.u_init_field = np.array(u_init_field)
        self.h_init_field = np.array(h_init_field)
        
##---------------------------------------------------         
##--------------------------------------------------- 
##--------------------------------------------------- 
    
    def thomas_dynamics(self, plot=False):
        cpt = 1 
        u_nNext = np.zeros((self.Nx))
        g_nNext = np.zeros((self.Nx))
        
        u = np.copy(self.u_init_field)
        h = np.copy(self.h_init_field)
        
        t = self.dt
        
        g = h * np.sqrt(3)
        
        fig, axes = plt.subplots(1, 2, figsize=(8,8), num="Evolution u et g")
        
        while cpt <= self.itmax :
            if cpt % 10 == 0 or cpt ==1 :
                for i in [0,1]: axes[i].clear()
                axes[0].plot(self.line_x, u, label="u it=%d" % cpt, color="blue")
                axes[0].set_ylim(-2.5, 2.5)
                                
                axes[1].plot(self.line_x, g, label="h it=%d" % cpt, color="red")
                axes[1].set_ylim(-1.25, 1.25)
                plt.legend(loc='best')
                
                plt.pause(1)

            for j in range(1, self.Nx-1) :
                u1 = u[j] * (1 - self.r*( u[j+1] - u[j-1] ))
                g1 = g[j] * (1 + self.r*( u[j+1] - u[j-1] ))
            
                u2 = g[j]*self.r*(g[j+1] - g[j-1])
                g2 = u[j]*self.r*(g[j+1] - g[j-1])
            
                u3 = self.fac*(u[j+1] - 2.*u[j] + u[j-1])
                g3 = self.mag_diff * self.dt / self.dx**2 * (g[j+1] - 2.*g[j] + g[j-1])
            
                u_nNext[j] = u1 + u2 + u3
                g_nNext[j] = g1 - g2 + g3
            
            u_nNext[0] = u_nNext[-2]
            u_nNext[-1] = u_nNext[1]
            
            g_nNext[0] = g_nNext[-2]
            g_nNext[-1] = g_nNext[1]
            
            u = u_nNext
            g = g_nNext
            
            
            cpt += 1
            t += self.dt
            
        self.u_final = u
        self.h_final = g / np.sqrt(3)
        
        plt.figure()
        plt.plot(self.line_x, self.u_final, label="u itmax %d" % self.itmax, color='blue')
        plt.plot(self.line_x, self.h_final, label="h itmax %d" % self.itmax,  color='red', linestyle='--')
        plt.legend(loc='best')
        
    def write_casesumup(self): 
        thiscase = time.strftime("%Y_%m_%d_%Hh%m_%S") + "MHD_nu:%.5f_ratio:%.2f_U:%.2f_H:%.2f.txt"\
                        %(self.nu, self.diff_ratio, self.U, self.H)
        
        pathtowrite = osp.join(self.logbook_path, thiscase)

        f = open(pathtowrite, "w")
        f.write("*"*28 + "\n")
        f.write("*"*10 + " Sum Up " + "*"*10)
        f.write("*"*28 + "\n\n")
        f.write("*"*28 + "\n")
        f.write("Case : %s \n" % thiscase)
        f.write("*"*28 + "\n")
                
        for item in self.sum_up.iteritems() :
            f.write("{} : {}\n".format(item[0], item[1]))
        
        f.write("\n")
        f.write("-"*28 + "\n")
        f.write("-"*28 + "\n")
        f.write("Init fields \nU:\n{} \n\nH:\n{}".format(self.u_init_field, self.h_init_field))        
        
        f.write("\n")
        f.write("-"*28 + "\n")
        f.write("-"*28 + "\n")
        f.write("Final fields \nU:\n{} \n\nH:\n{}".format(self.u_final, self.h_final))      
        
        f.close()
##---------------------------------------------------         
##--------------------------------------------------- 
##--------------------------------------------------- 

if __name__ == "__main__" :
    np.random.seed(100000)
    plt.ion()
    
    parser = THparser()
    tc = Thomas_class(parser)
    
    tc.init_fields("sin", phase=0, plot=True)
    tc.thomas_dynamics(True)

