#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def kraggle_digits (path_train_file = './train.csv') :
    """
    Function that will load train.csv data (which consists in a bank of pixel digits), and suffle it.
    
    Output :    
    -----------
    X and y :
    X   Dataset
    y   Corresponding label
    """
    
    #Using pandas library
    df = pd.read_csv(path_train_file) 
    
    #We next transform df into dataset (DS) for float 32 elements: 
    data = df.as_matrix().astype(np.float32)
    
    #We then shuffle the data
    np.random.shuffle(data)
    
    #In this Datasets first Column is for Label and the other one for DS
    X = data[:, 1:] 
    Y = data[:, 0]
    
    std = X.std(axis=0)
    np.place(std, std==0, 1) # We avoid std == 0 because we divide with it
    
    #Normalisation for better generalization
    X = (X - X.mean(axis=0)) / std
    
    return X, Y
    
    
