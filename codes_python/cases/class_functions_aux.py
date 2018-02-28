#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import argparse 
import types

def parser() :
    parser=argparse.ArgumentParser(description='\
    This parser will be used in several steps both in inference and ML postprocessing\n\
    Each entry is detailed in the help, and each of it has the most common default value. (run ... .py -h)')
    ## VaV T_inf
    #lists
    parser.add_argument('--T_inf_lst', '-T_inf_lst', nargs='+', action='store', type=int, default=[50],dest='T_inf_lst', 
                        help='List of different T_inf. Default : 50\n' )
    # Ints                       
    parser.add_argument('--N_discr', '-N', action='store', type=int, default=71, dest='N_discr', 
                        help='Define the number of discretization points : default %(default)d \n' )
    parser.add_argument('--compteur_max_adjoint', '-cptmax', action='store', type=int, default=100, dest='cpt_max_adj', 
                        help='Define compteur_max (-cptmax) for adjoint method: default %(default)d \n' )
    parser.add_argument('--N_sample', '-N_sample', action='store', type=int, default=3, dest='N_sample', 
                        help='Define the number of dupplication : default %(default)d \n' )
    parser.add_argument('--GPN_sample', '-GPN_sample', action='store', type=int, default=3, dest='GPN_sample', 
                        help='Define the number of distribution (beta, cholesky) samples: default %(default)d \n' )                        
    parser.add_argument('--max_epoch', '-epoch', action='store', type=int, default=50, dest='N_epoch', 
                        help='Define the number of training epoch in NN optimization: default %(default)d \n' )
    
    # Floats
    parser.add_argument('--H', '-H', action='store', type=float, default=0.5, dest='h', 
                        help='Define the convection coefficient h \n' )
    parser.add_argument('--delta_t', '-dt', action='store', type=float, default=1e-4, dest='dt', 
                        help='Define the time step disctretization. Default to %(default).5f \n' )
    parser.add_argument('--kappa', '-kappa', action='store', type=float, default=1.0, dest='kappa', 
                        help='Define the diffusivity number kappa. Default to %(default).2f\n' )
    parser.add_argument('--number_realization', '-num_real', action='store', type=int, default=100, dest='num_real', 
                        help='Define how many samples of epsilon(T) to draw for exact model. Default to %(default)d\n' )
    parser.add_argument('--tolerance', '-tol', action='store', type=float, default=1e-5, dest='tol', 
                        help='Define the tolerance on the optimization error. Default to %(default).8f \n' )
    parser.add_argument('--g_sup_max', '-g_sup', action='store', type=float, default=0.001, dest='g_sup_max', 
                        help='Define the criteria on grad_J to stop the optimization. Default to %(default).5f \n' )
    parser.add_argument('--beta_prior', '-beta_prior', type=float ,action='store', default=1 ,dest='beta_prior',\
                        help='beta_prior: first guess on the optimization solution. Value default to %(default)d\n')
    parser.add_argument('--learning_rate', '-lr', type=float ,action='store', default=1e-5 ,dest='lr',\
                        help='Define the learning rate for the NN back-propagation. Value default to %(default)d\n')                        
    
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
#    #strings
    
    # Pour la simulation de manière générale

    
    return parser.parse_args()
##-------------------------------------------------------------##
##-------------------------------------------------------------##
def subplot_cst(T, method='adj_bfgs', picpath = "./res_all_T_inf", save = False, comp=True) : 
    """
    Fonction pour comparer les beta. 
    Afficher les max et min des différents tirages 
    Comparer l'approximation de la température avec beta_final et la température exacte
    """
    if method in {"optimization", "Optimization", "opti"}:
        dico_beta_map   =   T.betamap
        dico_beta_fin   =   T.beta_final

        mins    =   T.mins_dict
        maxs    =   T.maxs_dict
        titles = ["Opti: Beta comparaison (bp = {},  cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]

    if method=="adj_bfgs":
        dico_beta_map   =   T.bfgs_adj_bmap
        dico_beta_fin   =   T.bfgs_adj_bf

        mins    =   T.bfgs_adj_mins_dict
        maxs    =   T.bfgs_adj_maxs_dict
        
        titles = ["ADJ_BFGS: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]
    
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

        axes[0].plot(T.line_z, mins[sT_inf], label='Valeurs minimums', marker='s',\
                                                    linestyle='none', color='magenta')
        axes[0].plot(T.line_z, maxs[sT_inf], label='Valeurs maximums', marker='s',\
                                                    linestyle='none', color='black')

        axes[0].fill_between(T.line_z, mins[sT_inf], maxs[sT_inf], facecolor= "0.2", alpha=0.4, interpolate=True)                
        
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])

        axes[0].legend(loc='best', fontsize = 10, ncol=2)
        axes[1].legend(loc='best', fontsize = 10, ncol=2)
        
        if save == True :
            title_to_save = "{}_{}_betmap_T_field_{}.png".format(T.cov_mod, titles[0][:3], sT_inf)
            title_to_save = osp.join(osp.abspath(picpath), title_to_save)
            plt.savefig(title_to_save)
#        plt.show()
#        
        print(T.bool_method["adj_bfgs_" + sT_inf] , T.bool_method["opti_scipy_" + sT_inf])
        
        if comp == True :
            if T.bool_method["adj_bfgs_" + sT_inf] == True and T.bool_method["opti_scipy_" + sT_inf] == True :
                comp_bmaps=  [T.betamap[sT_inf], T.bfgs_adj_bmap[sT_inf]]
                comp_mins =  [T.mins_dict[sT_inf], T.bfgs_adj_mins_dict[sT_inf]]
                comp_maxs =  [T.maxs_dict[sT_inf], T.bfgs_adj_maxs_dict[sT_inf]]
                comp_keys =  ["Scipy - optimization", "BFGS Adjoint Opti"]
                         
                comparaison(T, comp_bmaps, comp_mins, comp_maxs, comp_keys, T_inf, picpath = picpath, save=save)
    return axes
##---------------------------------------------------##
def subplot_Ncst(T, method='adj_bfgs', picpath = "./res_all_T_inf", save = False, comp=True) : 
    """
    Fonction pour comparer les beta. 
    Afficher les max et min des différents tirages 
    Comparer l'approximation de la température avec beta_final et la température exacte
    """
    if method in {"optimization", "Optimization", "opti"}:
        dico_beta_map   =   T.betamap
        dico_beta_fin   =   T.beta_final

        mins    =   T.mins_dict
        maxs    =   T.maxs_dict
        titles = ["Opti: Beta comparaison (bp = {},  cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]

    if method=="adj_bfgs":
        dico_beta_map   =   T.bfgs_adj_bmap
        dico_beta_fin   =   T.bfgs_adj_bf

        mins    =   T.bfgs_adj_mins_dict
        maxs    =   T.bfgs_adj_maxs_dict
        
        titles = ["ADJ_BFGS: Beta comparaison (bp = {}, cov_mod = {})".format(T.beta_prior[0], T.cov_mod), "Temperature fields"]
    
    curr_d = T.T_obs_mean[T.body]
    fig, axes = plt.subplots(1,2,figsize=(20,10))
    colors = 'green', 'orange'
            
    axes[0].plot(T.line_z, dico_beta_fin[T.body], label = "Beta for {}".format(T.body), 
        marker = 'o', linestyle = 'None', color = colors[0])
        
    axes[0].plot(T.line_z, dico_beta_map[T.body], label = 'Betamap for {}'.format(T.body),      
         marker = 'o', linestyle = 'None', color = colors[1])
    
    axes[0].plot(T.line_z, T.true_beta(curr_d), label = "True beta profil {}".format(T.body))
    
    axes[1].plot(T.line_z, T.h_beta(dico_beta_fin[T.body]), 
        label = "h_beta {}".format(T.body), marker = 'o',\
                                                linestyle = 'None', color = colors[0])
    axes[1].plot(T.line_z, T.h_beta(dico_beta_map[T.body], T.T_inf), 
        label = "h_betamap {}".format(T.body), marker = 'o',\
                                                linestyle = 'None', color = colors[1])
    axes[1].plot(T.line_z, curr_d, label= "curr_d {}".format(T.body))

    axes[0].plot(T.line_z, mins[T.body], label='Valeurs minimums', marker='s',\
                                                linestyle='none', color='magenta')
    axes[0].plot(T.line_z, maxs[T.body], label='Valeurs maximums', marker='s',\
                                                linestyle='none', color='black')

    axes[0].fill_between(T.line_z, mins[T.body], maxs[T.body], facecolor= "0.2", alpha=0.4, interpolate=True)                
    
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])

    axes[0].legend(loc='best', fontsize = 10, ncol=2)
    axes[1].legend(loc='best', fontsize = 10, ncol=2)
    
    if save == True :
        title_to_save = "{}_{}_betmap_T_field_{}.png".format(T.cov_mod, titles[0][:3], T.body)
        title_to_save = osp.join(osp.abspath(picpath), title_to_save)
        plt.savefig(title_to_save)
#        plt.show()
#        
    print(T.bool_method["adj_bfgs_" + T.body] , T.bool_method["opti_scipy_" + T.body])
    
    if comp == True :
        if T.bool_method["adj_bfgs_" + T.body] == True and T.bool_method["opti_scipy_" + T.body] == True :
            comp_bmaps=  [T.betamap[T.body], T.bfgs_adj_bmap[T.body]]
            comp_mins =  [T.mins_dict[T.body], T.bfgs_adj_mins_dict[T.body]]
            comp_maxs =  [T.maxs_dict[T.body], T.bfgs_adj_maxs_dict[T.body]]
            comp_keys =  ["Scipy - optimization", "BFGS Adjoint Opti"]
                     
            comparaison_Ncst(T, comp_bmaps, comp_mins, comp_maxs, comp_keys, picpath = picpath, save=save)
    return axes
##---------------------------------------------------##
##-------------------------------------------------------------##
def comparaison_Ncst(T, betamaps, minslsts, maxslsts, keys, picpath = "./res_all_T_inf", save = False) :
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
    curr_d = T.T_obs_mean[T.body]
    
    # Main plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], axisbg="#f5f5f5")
    ax.plot(T.line_z, betamap1, label=labelbmap1 + " for {}".format(T.body), linestyle='none', marker='o', color='magenta')
    ax.plot(T.line_z, betamap2, label=labelbmap2 + " for {}".format(T.body), linestyle='none', marker='+', color='yellow')
    ax.plot(T.line_z, T.true_beta(curr_d), label = "True beta profil {}".format(T.body), color='orange')
    
    ax.fill_between(T.line_z, mins_lst1, maxs_lst1, facecolor= "1", alpha=0.4, interpolate=True, hatch='\\', color="cyan", label=labelmm1)
    ax.fill_between(T.line_z, mins_lst2, maxs_lst2, facecolor= "1", alpha=0.7, interpolate=True, hatch='/', color="black", label=labelmm2)
    
    plt.legend(loc='best')
#    
    if save == True:
        title_to_save = "Scipy_opti_and_Adj_bfgs_bmaps_comparison_%s_(%s)" %(T.body, T.cov_mod) 
        title_to_save = osp.join(osp.abspath(picpath), title_to_save)
        plt.savefig(title_to_save)
##---------------------------------------------------##        
##---------------------------------------------------##
def comparaison(T, betamaps, minslsts, maxslsts, keys, T_inf, picpath = "./res_all_T_inf", save = False) :
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
#    
    if save == True:
        title_to_save = "Scipy_opti_and_Adj_bfgs_bmaps_comparison_%s_(%s)" %(sT_inf, T.cov_mod) 
        title_to_save = osp.join(osp.abspath(picpath), title_to_save)
        plt.savefig(title_to_save)
        
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
def sigma_plot_cst(T, method='adj_bfgs', exp = 0.02, save=False) :
    """
    Fonction pour comparer les sigma posterior
    """
    if method in {"optimization", "Optimization", "opti"}:
        sigma_post = T.sigma_post_dict
        title = "Optimization sigma posterior comparison "

    if method in {"",  "adjoint" }:
        sigma_post = T.adj_sigma_post
        
        title = "Adjoint (Steepest D) sigma posterior comparison "
    
    if method=="adj_bfgs":
        sigma_post = T.bfgs_adj_sigma_post
        
        title = "Adjoint (BFGS) sigma posterior comparison "        
    
    print ("Title = %s" %(title))
        
    for t in T.T_inf_lst :
        sT_inf = "T_inf_"+str(t)
        title_to_save = osp.join(osp.abspath("./res_all_T_inf"),title.replace(" ", "_")[:-1]+"_%s.png" %(sT_inf))
        print title_to_save
        
        dual = True if T.bool_method["opti_scipy_" + sT_inf] == True and T.bool_method["adj_bfgs_" + sT_inf] == True\
                else False
        
        base_sigma  =   np.asarray([T.prior_sigma[sT_inf] for j in range(T.N_discr-2)])
        exp_sigma   =   np.asarray([exp for j in range(T.N_discr-2)])
        
        plt.figure()
        plt.title(title_to_save)
        plt.semilogy(T.line_z, exp_sigma, label='Expected Sigma', marker = 's', linestyle='none')
        plt.semilogy(T.line_z, sigma_post[sT_inf], label="Sigma Posterior %s" %(title[:3]))
        plt.semilogy(T.line_z, base_sigma, label="Sigma for beta = beta_prior (base)")
        plt.legend(loc='best')
        
        if save == True :
            plt.savefig(title_to_save)
#        plt.show()
        
        if dual == True :
            title_dual = "Scipy_opti and Adj_bfgs sigma post comparison %s (%s)" %(sT_inf, T.cov_mod)
            title_to_save = osp.join(osp.split(title_to_save)[0],title_dual.replace(" ", "_")+".png")
            print title_to_save
            
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
    #        plt.show()
    
##---------------------------------------------------##
##---------------------------------------------------##
def sigma_plot_Ncst(T, method='adj_bfgs', exp = 0.02, save=False) :
    """
    Fonction pour comparer les sigma posterior
    """
    if method in {"optimization", "Optimization", "opti"}:
        sigma_post = T.sigma_post_dict
        title = "Optimization sigma posterior comparison "

    if method in {"",  "adjoint" }:
        sigma_post = T.adj_sigma_post
        
        title = "Adjoint (Steepest D) sigma posterior comparison "
    
    if method=="adj_bfgs":
        sigma_post = T.bfgs_adj_sigma_post
        
        title = "Adjoint (BFGS) sigma posterior comparison "        
    
    print ("Title = %s" %(title))
        
    title_to_save = osp.join(osp.abspath("./res_all_T_inf"),title.replace(" ", "_")[:-1]+"_%s.png" %(T.body))
    print title_to_save
    
    dual = True if T.bool_method["opti_scipy_" + T.body] == True and T.bool_method["adj_bfgs_" + T.body] == True\
            else False
    
    base_sigma  =   np.asarray([np.mean(np.diag(np.sqrt(T.cov_pri_dict[T.body]))) for j in range(T.N_discr-2)])
    exp_sigma   =   np.asarray([exp for j in range(T.N_discr-2)])
    
    plt.figure()
    plt.title(title_to_save)
    plt.semilogy(T.line_z, exp_sigma, label='Expected Sigma', marker = 's', linestyle='none')
    plt.semilogy(T.line_z, sigma_post[T.body], label="Sigma Posterior %s" %(title[:3]))
    plt.semilogy(T.line_z, base_sigma, label="Sigma for beta = beta_prior (base)")
    plt.legend(loc='best')
    
    if save == True :
        plt.savefig(title_to_save)
#        plt.show()
    
    if dual == True :
        title_dual = "Scipy_opti and Adj_bfgs sigma post comparison %s (%s)" %(T.body, T.cov_mod)
        title_to_save = osp.join(osp.split(title_to_save)[0],title_dual.replace(" ", "_")+".png")
        print title_to_save
        
        opti_sigma_post = T.sigma_post_dict[T.body]
        adj_bfgs_sigma_post = T.bfgs_adj_sigma_post[T.body]
        
        plt.figure()
        plt.title(title_dual)
        plt.semilogy(T.line_z, exp_sigma, label='Expected Sigma', marker = '^', linestyle='none')
        plt.semilogy(T.line_z, opti_sigma_post, label="Opti Sigma posterior")
        plt.semilogy(T.line_z, adj_bfgs_sigma_post, label="Adj_bfgs Sigma posterior")
        plt.semilogy(T.line_z, base_sigma, label="Sigma for beta = beta_prior (base)")
        plt.legend(loc='best')
        if save == True :
            plt.savefig(title_to_save)
#        plt.show()     
##---------------------------------------------------##
##---------------------------------------------------##
def check_T_beta(T, verbose=False):
    for t in T.T_inf_lst :
        sT_inf = "T_inf_"+str(t)
        tfieldtr = T.T_obs_mean[sT_inf]
        tfieldop = temp.h_beta(T.betamap[sT_inf], t)
        tfieldbf = temp.h_beta(T.bfgs_adj_bmap[sT_inf], t)  
        
        bfieldth = T.true_beta(tfieldtr, t)                 
        bfieldop = T.betamap[sT_inf]
        bfieldbf = T.bfgs_adj_bmap[sT_inf]
        
        plt.figure("Comp T_field T_inf = %d" %(t))
        plt.plot(T.line_z, tfieldtr, label="True T_field %s"%(sT_inf), c='green')
        
        plt.plot(T.line_z, tfieldop, label="Opti T_field %s"%(sT_inf), marker='+',\
                c='r',fillstyle = 'none', linestyle='none')
        plt.plot(T.line_z, tfieldbf, label="BFGS T_field %s"%(sT_inf), c='k',\
                marker='o', fillstyle = 'none', linestyle='none')
        plt.legend(loc='best')
        
        plt.figure("Comp B_field T_inf = %d" %(t))
        plt.plot(T.line_z,bfieldth, label="True B_field %s"%(sT_inf), c='green')
        
        plt.plot(T.line_z, bfieldop, label="Opti B_field %s"%(sT_inf),marker='+',\
                c='r', fillstyle = 'none', linestyle='none')
        plt.plot(T.line_z, bfieldbf, label="BFGS B_field %s"%(sT_inf), c='k',\
                marker='o', fillstyle = 'none', linestyle='none')
        plt.legend(loc='best')
##---------------------------------------------------##
##---------------------------------------------------##        