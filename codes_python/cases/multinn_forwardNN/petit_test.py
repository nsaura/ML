#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split

from tensorflow import reset_default_graph

tf_folder = osp.abspath(osp.dirname("../../TF/"))
sys.path.append(tf_folder)

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

import NN_class_try as NNC

print ok
