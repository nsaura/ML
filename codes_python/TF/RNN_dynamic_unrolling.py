#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import tensorflow as tf

n_steps = 2
n_inputs = 3
n_neurons = 5

try :
    tf.reset_default_graph()
except :
    pass

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])


# On peut utiliser un variable_scope pour initialiser les poids de chaque couche.
# Pour l'initialisation de He on utilise initializer=tf.variance_scaling_initializer()
# Pour un autre type d'initialisation, on peut e.g. initializer=tf.zeros_initializer()

with tf.variable_scope("rnn", initializer=tf.variance_scaling_initializer()) : 
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)


init = tf.global_variables_initializer()

# On peut checker l'initialisation des poids comme suit 
#with tf.Session() as sess :
#    init.run()
#    print sess.run(tf.trainable_variables())

X_batch = np.array([
                   [[0, 1, 2], [9, 8, 7]],
                   [[3, 4, 5], [0, 0, 0]],
                   [[6, 7, 8], [6, 5, 4]],
                   [[9, 0, 1], [3, 2, 1]]
                   ]
                  )

with tf.Session() as sess :
    init.run()
    outputval = sess.run(outputs, feed_dict={X:X_batch})

    print outputval
    
# Dans la construction de couche utilisant tf.variable_scope (voir plus haut), il est possible de préciser plusieurs arguments comme :
# regularizer : default regularizer for variables within this scope
# constraint: An optional projection function to be applied to the variable after being updated by an Optimizer(e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected Tensor representing the value of the variable and return the Tensor for the projected value (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
