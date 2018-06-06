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


class Boostraped_Neural_Network :
    def __init__(self, n_estimators, dataset) :
        
