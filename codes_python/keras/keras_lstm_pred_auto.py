#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import os
import csv
import sys 
import time
import os.path as osp

import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.utils import plot_model
from keras.callbacks import TensorBoard

# define input sequence
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
n_in = len(sequence)

# reshape array input into [samples, timesteps, features]
seq_in = sequence.reshape(1, n_in, 1)

# Let's now say we expect the output to be : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1 

# we can define the encoder-decoder LSTM architecture that expects input sequences with nine time steps and one feature, and outputs a sequence with nine time steps and one feature

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
# The repeatvector is size of n_out
model.add(RepeatVector(n_out))
print model.output_shape
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

# fit model
# The difference will be in the feed_dict of the fit function
model.fit(seq_in, seq_out, epochs=300, verbose=False,
            callbacks=[TensorBoard(log_dir='/tmp/kerasauto')])
            
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

yhat = model.predict(seq_in, verbose=0)
print yhat
