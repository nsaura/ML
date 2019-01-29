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
sequence = sequence.reshape(1, n_in, 1)

# we can define the encoder-decoder LSTM architecture that expects input sequences with nine time steps and one feature, and outputs a sequence with nine time steps and one feature

# define model

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(sequence, sequence, epochs=300, verbose=False,
            callbacks=[TensorBoard(log_dir='/tmp/kerasauto')])
            
ce = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

yhat = model.predict(sequence, verbose=0)
print yhat
