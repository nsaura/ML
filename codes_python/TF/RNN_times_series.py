#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import tensorflow as tf

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
o
try :
    tf.reset_default_graph()
except :
    pass

# Page 575

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#On a a présent un vecteur output de taille n_neurons, mais on veut une seule valeur pour une iteration
# On wrap la loop cell dans un OutputProjectionWrapper
# Cela consiste a lié toute les sorties avec une couche dense de neurone sans fonction d'activation

