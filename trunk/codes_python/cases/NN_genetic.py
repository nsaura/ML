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

from random import randint, random
from operator import add

## Import de la classe TF ##
nnc_folder = osp.abspath(osp.dirname("../TF/NN_class_try.py"))
sys.path.append(nnc_folder)

import NN_class_try as NNC
import Class_Temp_Cst as ctc
import class_functions_aux as cfa
import Gaussian_Process_class as GPC
import NN_inference_ML as NNI

ctc = reload(ctc)
cfa = reload(cfa)
GPC = reload(GPC)
NNC = reload(NNC)
NNI = reload(NNI)

parser = cfa.parser()

nn_params = dict()
#keys = ["N_HL", "N_HN", "Act", "Opt"]

nn_params["N_HL"] = [10**i for i in range(1,4)]
nn_params["N_HN"] = [2**j for j in range(4,7)]
nn_params["Act"]  = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.selu, tf.nn.sigmoid]
nn_params["Opt"]  = [tf.train.RMSPropOptimizer, tf.train.adam]

def individual(nn_params):
    """
    nn_params doit être un dictionnaire de liste prenant plusieurs valeurs des 4 paramètre :
        1 - Nombre de HL 
        2 - Nombre de nœuds par HL
        3 - Fonction d'activation
        4 - Méthode d'optimisation
    """
            
