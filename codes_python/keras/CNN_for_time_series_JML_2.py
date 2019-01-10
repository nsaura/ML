#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

# CNNs can support parallel input time series as separate channels, like red, green, and blue components of an image. 
# Therefore, we need to split the data into samples maintaining the order of observations across the two input sequences.

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
	
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
	
# define input sequence
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# ret --> array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))]) # Colonnes

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1)) # Colonnes transformées en lignes

# ret --v
# in_seq : array([[10],
#                 [20],
#                 [30],
#                 [40],
#                 [50],
#                 [60],
#                 [70],
#                 [80],
#                 [90]])

# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq)) # Matrices juxtaposées telles quelles

# choose a number of time steps
n_steps = 3

# convert into input/output
X, y = split_sequences(dataset, n_steps)

# summarize the data
for i in range(len(X)):
	print(X[i], y[i])	

n_features = X.shape[2]

print n_features

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))



#(None, 2, 64)

model.add(MaxPooling1D(pool_size=2))

#(None, 1, 64)

model.add(Flatten())
# Flatten serves as a connection between the convolution and dense layers

#(None, 64)

model.add(Dense(50, activation='relu'))

#(None, 50)

model.add(Dense(1))

#(None, 1)

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = np.array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

print yhat

#-----------------------------------------#
####   Notes sur Dimension des poids   ####
#-----------------------------------------#
#
# --> voir model.get_weights() 
# 
#
# A = np.array(model.get_weights()), A.shape = (6,), 6 matrices de matrices (parfois) ou matrices simples

# A[0]
#   
#   La couche Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features) générera des poids de la taille (kernel_size, n_features, filters)
#
#
# A[1] <<<----- Biais ajouté à la couche de convolution de la taille du nombre de filtre
#   
# La couche de pooling réduit la dimension et résume les informations des filtres 
# Le Flatten applatit les tableaux (None, 1, 64) en (None, dim1*dim2*etc)
#
#
# A[2] et A[3]
# 
#   La couche Dense lie la couche précédente avec la taille demandée en premier argument de la fonction. Ainsi ici les poids de la première couche dense sera un matrice de taille (filters, n_neurons_dense_layer)
#   A cette couche de poids, s'ajoute les biais, un vecteur de taille (n_neurons_dense_layer)
#
#
# A[4] et A[5]
#   Idem pour la couche suivant, on aura une matrice de poids (n_neurons_dense_layer, n_output)
#   Et donc un vecteur de biais (n_output)
#
#
#  Au total on aura sum(A[i].size for i in range(A.size)) = 3621 paramètres à optimiser


#------------------------------------------------#
####   Notes sur les dimensions des couches   ####
#------------------------------------------------#
#
# --> voir model.output_shape sur chaque couche  #
#
#
# L'entrée est toujours de taille (n_steps, n_features)
#
# On note new_steps = (n_steps-kernel_size)/strides + 1
#
#
#   La convolution sort une matrice de taille (None, new_steps, 64) sans padding plus généralement (batch, new_steps, filters)
#
#   Le Pooling réduit la taille de feature 
#
#   Le flatten "empile" toutes les données dans une matrice (None, filters*new_steps*autres_dim)
#
#   La couche dense aura une sortie de la taille de n_neurons_dense_layer

	
