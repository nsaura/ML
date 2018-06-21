#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

## Import de la classe TF ##
bootstrap_folder = osp.abspath(osp.dirname("../TF/Bootstrap_NN.py"))
sys.path.append(bootstrap_folder)

import Bootstrap_NN as BNN

import NN_class_try as NNC
import Class_Vit_Choc as cvc
import harmonic_sinus as harm


