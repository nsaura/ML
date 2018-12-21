#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# Ici
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/?__s=pv31fqczs5ynqtudbiip
	
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps
n_steps = 3

# split into samples
X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# run CNN_for_time_series_JML.py	
# (array([10, 20, 30]), 40)
# (array([20, 30, 40]), 50)
# (array([30, 40, 50]), 60)
# (array([40, 50, 60]), 70)
# (array([50, 60, 70]), 80)
# (array([60, 70, 80]), 90)

#Conv1D convolutional hidden layer that operates over a 1D sequence.
	
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='selu', 
                input_shape=(n_steps, n_features))) # en vrai relu, filters = 64
                
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='selu')) # en vrai 50, 'relu'
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print yhat 

# Jusqu'ici
yh = []
for i in range(100):
    yhat = model.predict(x_input, verbose=0)
    yh.append(yhat)

yh = np.array(yh)
print yh.mean()
print yh.std()




