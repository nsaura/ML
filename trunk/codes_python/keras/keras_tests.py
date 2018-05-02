#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import os.path as osp
from sklearn.model_selection import train_test_split

import sklearn.datasets as sdata

import keras 
import tensorflow as tf
import Class_Keras_NN as cknn

cknn = reload(cknn)

boston = sdata.load_boston()
diabetes = sdata.load_diabetes()
mf1 = sdata.make_friedman1(n_samples=2500)
mf2 = sdata.make_friedman2(n_samples=2500)

datas =\
[   [boston.data, boston.target, "boston"],
    [diabetes.data, diabetes.target, "diabetes"],
    [mf1[0], mf1[1], "friedman1"],
    [mf2[0], mf2[1], "friedman2"],
]

par = cknn.parser()

dict_layers = lambda x,size,act :\
{   "I" : x,\
    "N1" : [size,act],\
    "N2" : [size,act],\
    "N3" : [size,act],\
    "N4" : [size,act],\
    "N5" : [size,act],\
    "N6" : [size,act],\
    "N7" : [size,act],\
    "N8" : [size,act],\
    "N9" : [size,act],\
    "N10" :[size,act],\
    "O"  : [1, act]\
}

f = open("%s_%s_results_keras_test.txt" %(par.opti, par.act), "w")

for j, (X, y, name) in enumerate(datas) :
    print name 
    if X.shape[0] <= par.bsz :
        par.bsz = 50
    
    kwargs = cknn.defined_optimizer(par)
    
    k_nn_dict = dict_layers(X.shape[1], 10, "selu")
    k_nn = cknn.K_Neural_Network(k_nn_dict, opti=par.opti, loss=par.loss, metrics=par.metrics, max_epoch=par.maxepoch,\
                            verbose=False, non_uniform_act=True, **kwargs)

    k_nn.build()
    k_nn.compile(save=True, name="keras_tests.h5")
    k_nn.train_and_split(X, y, shuffle=True, scale=True)
    
    k_nn.fit_model()
    
    plt.figure("Comparaison prediction/Vraie valeure %s" % name)
    plt.plot(k_nn.y_test, k_nn.y_test, label="Wanted %s" % name, color='black')
    plt.plot(k_nn.y_test, k_nn.prediction(k_nn.X_test), label="Predicted %s" % name, linestyle="none", marker='o', c='navy')
    plt.legend(loc='best')
    
    plt.pause(10)
    plt.savefig("%s_keras_test_fig.png" % name)
    
    f.write("-"*20 +"\n")
    f.write("{}\n".format(k_nn_dict))
    
    f.write("-"*20 +"\n")
    f.write("NAME = %s\n" % name)
    f.write("-"*20 +"\n")
    
    f.write("Scores :\n")
    for i in k_nn.metrics :
        f.write("%s final score : %f \n" % (i, k_nn.fit.history[i][-1]))
    
    f.write("-"*20 +"\n")
    f.write("\n")
    
f.close()
