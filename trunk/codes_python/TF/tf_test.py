#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import os.path as osp

import sklearn.datasets as sdata

import NN_class_try as NNC
NNC = reload(NNC)

import tensorflow as tf

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

dict_layers = lambda x,size :\
{   "I" : x,\
    "N1" : size,\
    "N2" : size,\
    "N3" : size,\
    "N4" : size,\
    "N5" : size,\
    "N6" : size,\
    "N7" : size,\
    "N8" : size,\
    "N9" : size,\
    "N10" :size,\
    "O"  : 1\
}


kwargs = {}

act = "selu"
opti = "Adam"
loss = "OLS"
reduce_type = "mean"

kwargs["beta1"] = 0.75
kwargs["beta2"] = 0.3
lr = 5e-4

f = open("%s_%s_results_tf_test.txt" % (opti, act), "w")

kwargs["color"] = "red"
kwargs["step"] = 10
kwargs["verbose"] = True
kwargs["b_sz"] = 50
max_epoch = 3000

scale=True

for j, (X, y, name) in enumerate(datas) :
    print name 
#    print X, y, name
    NN_dict = dict_layers(X.shape[1], 10)
    nn_obj = NNC.Neural_Network(lr, N_= NN_dict, max_epoch=max_epoch, **kwargs)
    
    nn_obj.train_and_split(np.array(X), np.array(y.reshape(-1,1)), strat=False, shuffle=True, scale=scale)
    nn_obj.tf_variables()
    nn_obj.feed_forward(activation=act)
    nn_obj.def_training(opti)
    nn_obj.cost_computation(loss, reduce_type=reduce_type)
    nn_obj.def_optimization()
    try :
        nn_obj.training_session(tol=1e-3, verbose=True)

    except KeyboardInterrupt :
        print ("Session closed")
        nn_obj.sess.close()
    
    try :
        verbose = kwargs["verbose"]
    except KeyError :
        verbose = False
    
    
    plt.figure("Comparaison prediction/Vraie valeure %s" % name)
    plt.plot(nn_obj.y_test, nn_obj.y_test, label="Wanted %s" % name, color='black')
    plt.plot(nn_obj.y_test, nn_obj.predict(nn_obj.X_test), label="Predicted %s" % name, linestyle="none", marker='o', c='darkblue')
    plt.legend(loc='best')
    
    plt.pause(10)
    plt.savefig("%s_TF_test_fig.png" % name)
    
    f.write("-"*20 +"\n")
    f.write("{}\n".format(NN_dict))
    
    f.write("-"*20 +"\n")
    f.write("NAME = %s\n" % name)
    f.write("-"*20 +"\n")
    
    f.write("Scores :\n")
    f.write("OLS :%f\n" % nn_obj.costs[-1])
    
    f.write("-"*20 +"\n")
    f.write("\n")
    
f.close()
