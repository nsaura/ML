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

plt.ion()

# On déclare un objet de la classe T pour pouvoir avoir accès à des grandeurs propres à la méthode.
# On n'a cependant pas besoin de relancer l'inférence
# Puisque les distributions importantes ont été écrites
T = ctc.Temperature_cst(parser) 
T.obs_pri_model()
T.get_prior_statistics()

X,y,v = GPC.training_set(T, parser.N_sample)

dict_layers = {"I" : 2,\
               "N1" : 1000,\
               "N2" : 500,\
               "N3" : 100,\
               "N4" : 100,\
               "N5" : 100,\
               "N6" : 100,\
               "N7" : 100,\
               "N8" : 100,\
               "N9" : 100,\
               "N10": 100,\
               "O"  : 1}
               
#dict_layers = {"I" : 2,\
#               "N1" : 1000,\
#               "N2" : 100,\
#               "N3" : 1000,\
#               "N4" : 100,\
#               "N5" : 1000,\
#               "N6" : 100,\
#               "N7" : 1000,\
#               "O"  : 1}               
N_hidden_layer = len(dict_layers.keys()) - 1
