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

X,y,v = GPC.training_set(T, 5)
dict_layers = {"I" : 2,\
              "N1" : 10,\
              "N2" : 10,\
              "N3" : 10,\
              "N4" : 10,\
              "O"  : 1}
N_hidden_layer = len(dict_layers.keys()) - 1


nn = NNC.Neural_Network(0.004, N_=dict_layers,\
    max_epoch=10)

nn.train_and_split(X,y,strat=False)
nn.build_graph()

nn.tf_variables()
nn.feed_forward()

     


