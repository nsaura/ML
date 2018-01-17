#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import time
import sys, warnings, argparse

import numpy as np
import pandas as pd
import os.path as os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import optimize as op

import class_temp as ct #Pour utiliser les fonctions de classs_temp
import class_functions_aux as cfa #Pour les tracés post-process

parser = cfa.parser()

temp = ct.Temperature(parser)
print(parser)

temp.obs_pri_model()
temp.get_prior_statistics()

