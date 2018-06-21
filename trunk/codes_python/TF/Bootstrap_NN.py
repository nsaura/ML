#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os, csv
from sklearn.model_selection import train_test_split

import tensorflow as tf
#import tensorlayer as tl

import NN_class_try as NNC
NNC = reload(NNC)

plt.ion()


class Bootstraped_Neural_Network :

    def __init__(self, n_estimators, dataset) :
        
        self.n_estimators = n_estimators
        self.X = dataset["data"]
        self.y = dataset["target"]
        
        self.length_resample = len(self.X)
        
        self.bkey = lambda b : "dataset_%s" % str(b)
        self.nnkey = lambda b : "NN_%s" % str(b)
        
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------

    def resample_dataset(self):
        bootstrap_dataset = dict()
        
        for b in range(self.n_estimators) :
            bootstrap_dataset[self.bkey(b)] = {}
        
        for b in range(self.n_estimators) : 
            Xcp = np.copy(self.X)
            ycp = np.copy(self.y)
            
            permute = np.random.permutation(self.length_resample)
            bootstrap_dataset[self.bkey(b)]["data"] = Xcp[permute]
            bootstrap_dataset[self.bkey(b)]["target"] = ycp[permute]
        
        self.bootstrap_dataset = bootstrap_dataset
        
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
    
    def build_NN(self, lr, dict_layers, act, opti, loss, palet, scaler="Standard",\
                    max_epoch=100, reduce_type="sum", rdn=0, **kwargs) :
        NN_dict = {}
        testkey = lambda key : key not in kwargs.keys()
        
        val = False if testkey("val") else kwargs["val"]
        n_compo = "mle" if testkey("n_components") else kwargs["n_components"]
            
        for b in range(self.n_estimators) :
            # Define an NN object
            nn_b = NNC.Neural_Network(lr, scaler=scaler, N_=dict_layers, max_epoch=max_epoch,reduce_type=reduce_type, color = next(palet), **kwargs)
            
            # Getting and Spliting The Data
            X_NN = np.copy(self.bootstrap_dataset[self.bkey(b)]["data"])
            y_NN = np.copy(self.bootstrap_dataset[self.bkey(b)]["target"])
            
            #Preparing the Tensorflow Graph
            nn_b.split_and_scale(X_NN, y_NN, shuffle=False, val=val, n_components=n_compo, random_state=rdn)
            nn_b.tf_variables()
            nn_b.layer_stacking_and_act(activation=act)
            
            #Setting Optimizer and Loss for the graph
            nn_b.def_optimizer(opti)
            nn_b.cost_computation(loss)
            
            #Display a Recap
            nn_b.case_specification_recap()
            
            try :
                nn_b.training_phase(tol=1e-3)

            except KeyboardInterrupt :
                print ("Session closed")
                nn_b.sess.close()
            
            self.NN_test_plot(nn_b)
                
            NN_dict[self.nnkey(b)] = nn_b

        self.NN_dict = NN_dict
        
##--------------------------------------------------------------------------------------------    
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------

    def bootstrap_prediction(self, xs, rescale, variance=False) :
        xs = np.copy(xs)
        prediction = []
        
        for b in range(self.n_estimators):
            if rescale == True :
                nxs = self.NN_dict[self.nnkey(b)].scale_inputs(xs)
            
            else :
                nxs = np.copy(xs)
            
            prediction.append(self.NN_dict[self.nnkey(b)].predict(nxs, rescale_tab=False)[0,0])
        
        bstrap_pred = 1./(self.n_estimators-1) * sum(prediction)
        
        if variance == True :
            return bstrap_pred, prediction
        else : 
            return bstrap_pred
            
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------

    def bootstrap_variance(self, xs, rescale) :
        xs = np.copy(xs)
        ymean_bstrap, ybs_pred = self.bootstrap_prediction(xs, rescale, variance=True)
        
        var_ymean_bstrap = sum([(ymean_bstrap - yb)**2 for yb in ybs_pred])
        var_ymean_bstrap /= (self.n_estimators -1)
        
        return var_ymean_bstrap
        
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
    
    def r_2_dataset(self) :#,lr, dict_layers, act, opti, loss, scaler="Standard",\
                    #max_epoch=100, reduce_type="sum", rdn=0, **kwargs) :
        # On utilise les NN déjà entrainés
        # On reshuffle X :
        
        Xcp = np.copy(self.X)
        ycp = np.copy(self.y)
        
        r = np.zeros((1))
        r_lambda = lambda moy, ex, var : max((moy-ex)**2 - var, 0)
        
        for i in range(Xcp.shape[0]) :
            moy_ = self.bootstrap_prediction(Xcp[i], rescale=True)
            var_ = self.bootstrap_variance(Xcp[i], rescale=True)
            
            r = np.block([[r], [r_lambda(moy_, ycp[i], var_)]])
        
        r = np.delete(r, 0, axis=0)
        
        return r 
        
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------

    def NN_test_plot(self, nn_b) :
        lr = nn_b.lr
        act  =  nn_b.activation
        loss =  nn_b.err_type        
        opti =  nn_b.train_mod
        scaler = nn_b.scaler_name
        max_epoch = nn_b.max_epoch
        
        beta_test_preds = np.array(nn_b.predict(nn_b.X_test, rescale_tab=False))
        test_line = range(len(nn_b.X_test))
        
        dev_lab = "Pred_lr_{}_{}_{}_Maxepoch_{}".format(lr, opti, act, scaler, max_epoch)
        
        deviation = np.array([ abs(beta_test_preds[j] - nn_b.y_test[j]) for j in test_line])
        error_estimation = sum(deviation)
        
        if plt.fignum_exists("Comparaison sur le test set") :
            plt.figure("Comparaison sur le test set")
            plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                        fillstyle='none', linestyle='none', c=nn_b.kwargs["color"])

        else :
            plt.figure("Comparaison sur le test set")
            plt.plot(test_line, nn_b.y_test, label="Expected value", marker='o', fillstyle='none',\
                        linestyle='none', c='k')   
            plt.plot(test_line, beta_test_preds, label=dev_lab, marker='+',\
                        fillstyle='none', linestyle='none', c=nn_b.kwargs["color"])
     
        plt.legend(loc="best", prop={'size': 7})
        
        if plt.fignum_exists("Deviation of the prediction") :
                plt.figure("Deviation of the prediction")
                plt.plot(nn_b.y_test, beta_test_preds, c=nn_b.kwargs["color"], marker='o',\
                         linestyle='none', label=dev_lab, ms=3)
            
        else :
            plt.figure("Deviation of the prediction")
            plt.plot(nn_b.y_test, nn_b.y_test, c='k', label="reference line")
            plt.plot(nn_b.y_test, nn_b.y_test, c='navy', marker='+', label="wanted value",linestyle='none')
            plt.plot(nn_b.y_test, beta_test_preds, c=nn_b.kwargs["color"], marker='o',\
                          linestyle='none', label=dev_lab, ms=3)
        
###----------------------------------------------------------------
###----------------------------------------------------------------        
if __name__ == "__main__" :
    from sklearn.datasets import load_boston

    boston = load_boston()
    dataset = {}
    dataset["data"] = boston.data     
    dataset["target"] = boston.target
        
    BNN = Bootstraped_Neural_Network(4, dataset) 
    BNN.resample_dataset()       
    
    color = iter(["blue", "darkred", "aqua", "olive", "magenta", "orange", "mediumpurple", "chartreuse", "tomato", "saddlebrown", "powderblue", "khaki", "salmon", "darkgoldenrod", "crimson", "dodgerblue", "limegreen"])
    
    BNN.build_NN(1e-3, {"I" : 13, "N1" : 80, "N2" : 80, "N3" : 80, "N4" : 80,  "N5" : 80, "N6": 80 ,"O": 1}, "selu", "Adam", "Ridge",
                 color, scaler="Standard", max_epoch=500, reduce_type="sum", rdn=0, bsz = 32, BN=True)
    
