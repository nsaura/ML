#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import tensorflow as tf

n_inputs = 3
n_neurons = 5

np.random.seed(10000)

try :
    tf.reset_default_graph()
except :
    pass
#---------------------------------------------------------------------------------

# Without RNN tensorflow's RNN operations

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons]), dtype=tf.float32)

# Weights and biaises shared in the two different layers
# Each layer has inputs
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2]])#, [3,4,5], [6,7,8], [9,0,1]])
X1_batch = np.array([[9, 8, 7]])#, [0,0,0], [6,5,4], [3,2,1]])

with tf.Session() as sess :
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1:X1_batch})


#---------------------------------------------------------------------------------    
tf.reset_default_graph()

# We code the same thing using unrolled RNN network :
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

Y0, Y1 = output_seqs

init = tf.global_variables_initializer()

with tf.Session() as sess :
    init.run()
    Y0_val_2, Y1_val_2 = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1:X1_batch})

#del basic_cell
#del output_seqs

#---------------------------------------------------------------------------------
tf.reset_default_graph()

# How to deal with more than 2 steps. Generalized the placeholder definition

n_steps = 2
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2])) # extracts the list of input sequences for each time step
# n_steps devient le premier argument, la taille du mini batch en deuxieme
# So we have one tensor per time steps

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2]) # Merge all the output tensors into a single tensor

init = tf.global_variables_initializer()

X_batch = np.array([
                   [[0, 1, 2], [9, 8, 7]],
                   [[3, 4, 5], [0, 0, 0]],
                   [[6, 7, 8], [6, 5, 4]],
                   [[9, 0, 1], [3, 2, 1]]
                   ]
                  )


