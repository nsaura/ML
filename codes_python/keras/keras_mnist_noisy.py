#!/usr/bin/python
# -*- coding: latin-1 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
import keras.backend as K

import keras_utility as ku
ku = reload(ku)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#-- On reshape la forme des données pour qu'elles soient de la forme 
## "Nombre d'image x 784" 784 = 28**2 qui siginife le nombre de pixe;s
## On donc fait en sorte que chaque pixel soit une caractéristique de l'entrée
# Notons enfin que cette étape n'est nécessaire uniquement à cause de la forme des donnés
#--

noise_factor = 0.4

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_noisy = x_train   + noise_factor * np.random.normal(size = x_train.shape)
x_test_noisy  = x_test    + noise_factor * np.random.normal(size = x_test.shape)
#np.random.normal : Draw random samples from a normal (Gaussian) distribution. shape arg : Output shape

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
# np.clip : Considère un tableau et un intervalle. Si des valeurs dépassent de ce tableau, elles sont remplacées par la valeur de la borne la plus proche

input_size, hidden_size, code_size = 784, 128, 32
# Avoir une layer code de taille inférieure aux autres empêche l'overfitting

## On construit le graphe
input_img = Input(shape=(input_size,)) ## On commence par créer le tenseur des entrées
hidden_1 = Dense(hidden_size, activation='relu')(input_img)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_img = Dense(input_size, activation='sigmoid')(hidden_2)

## On précise le modèle et on l'entraine sur le graphe avec x_train_noisy
## Le modèle va de input_img à output_img donc on écrit :
autoencoder = Model(input_img, output_img)

## L'instance qui équivaut à eval dans tensorflow :
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=3)

## On entraine le modèle avec en entrée les input bruitées et en label les bonnes images

ku.plot_autoencoder_outputs(autoencoder, 7, (28,28), x_test_noisy)

# Avec epoch = 3, et activation de output_img = relu on obtient :
#Epoch 1/3
#60000/60000 [==============================] - 8s 131us/step - loss: 0.4092
#Epoch 2/3
#60000/60000 [==============================] - 8s 128us/step - loss: 0.4100
#Epoch 3/3  
#60000/60000 [==============================] - 8s 133us/step - loss: 0.3905

# Avec epoch = 3, et activation de output_img = sigmoid on obtient :
#Epoch 1/3
#60000/60000 [==============================] - 8s 133us/step - loss: 0.1651
#Epoch 2/3
#60000/60000 [==============================] - 8s 130us/step - loss: 0.1271
#Epoch 3/3
#60000/60000 [==============================] - 8s 128us/step - loss: 0.1196


