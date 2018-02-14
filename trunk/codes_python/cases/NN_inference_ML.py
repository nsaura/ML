#!/usr/bin/python
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

import tensorflow as tf
from sklearn.model_selection import train_test_split

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import Gaussian_Process_class as GPC
import class_functions_aux as cfa
import Class_Temp_Cst as ctc
import NN_class_try as NNC

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)

parser = cfa.parser()

# On déclare un objet de la classe T pour pouvoir avoir accès à des grandeurs propres à la méthode.
# On n'a cependant pas besoin de relancer l'inférence
# Puisque les distributions importantes ont été écrites
T = ctc.Temperature_cst(parser) 
T.get_prior_statistics()

X,y,v = GPC.training_set(T, parser.N_sample)
dict_layers = {"I" : 2,\
               "N1" : 100,\
               "N2" : 100,\
               "N3" : 50,\
               "N4" : 50,\
               "N5" : 100,\
               "O"  : 1}
N_hidden_layer = len(dict_layers.keys()) - 1


nn = NNC.Neural_Network(1e-7, N_=dict_layers,\
    max_epoch=100)

## Les différents attibuts que les prochaines méthodes vont créer seront utiliser après ouverture d'une session

nn.train_and_split(X,y,strat=False)
#       nn.X_train, nn.X_test
#       nn.y_train, nn.y_test

nn.tf_variables()
#       nn.w_tf_d 
#       nn.w_tf_d

nn.feed_forward(activation="leakyrelu")
#       nn.Z

init = tf.initialize_all_variables()

#nn.def_training("RMS", decay=0.99, momentum=0.9)
nn.def_training("GD")
#       nn.train_op

nn.cost_computation("OLS")

nn.def_optimization()
#       nn.minimize_loss

nn.training_session(batched=False)
#print("L\'arbre est construit. Selon le dictionnaire N_: \n{}\n\
#Les matrices sont a priori bien dimensionnées, on vérifie avec les différents poids pour chaque couches: \n{}".format(nn.N_, nn.w_tf_d))




#nn.error_computation(err_

     


