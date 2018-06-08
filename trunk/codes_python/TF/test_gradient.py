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

from sklearn.datasets import load_boston

from datetime import datetime 

plt.ion()

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

#try :
#    print ("Graph Reseted") 
#    tf.reset_default_graph()
#except AssertionError :
#    pass
#    
max_epoch = 1000
step = 5

X, y = load_boston().data, load_boston().target

now = datetime.utcnow().strftime("%Y_%m_%d_%Hh%M_%S")
root_logdir = "tf_logs"+now
logdir = "{}/run-{}/".format(root_logdir, now)

# number of nodes
n_input = X.shape[1]
nN1 = 200
nN2 = 200

X_train, X_test, y_train, y_test = train_test_split(X, y)

w1_init = np.random.randn(n_input, nN1) / np.sqrt(n_input)
b1_init = np.zeros(nN1)
w2_init = np.random.randn(nN1, 1) / np.sqrt(nN1)

# define tensorflow variables and expressions
x = tf.placeholder(tf.float32, (None, n_input), name="inputs")
t = tf.placeholder(tf.float32, (None), name="output")

jac = tf.placeholder(tf.float32, (None), name="grad_x")

w1 = tf.Variable(w1_init.astype(np.float32), name="w1")
b1 = tf.Variable(b1_init.astype(np.float32), name="b1")
w2 = tf.Variable(w2_init.astype(np.float32), name="w2")

z1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y_pred = tf.matmul(z1, w2)

derr_y_pred = tf.norm( tf.gradients(y_pred, [x]) )

cost = tf.reduce_sum(tf.add(tf.square(y_pred - t), tf.multiply(1e-3, jac)))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3) 

train_op = optimizer.minimize(cost)


sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()

sess.run(init)
mse_summary  = tf.summary.scalar("MSE", cost)
file_writer= tf.summary.FileWriter(logdir, tf.get_default_graph())

plt.figure("Log evolution error")
plt.figure("Evol Euclidian norm")

for epoch in range(max_epoch) :
    
    derr_y_pred_n = sess.run(derr_y_pred, feed_dict = {x:X_train})
        
    plt.figure("Evol Euclidian norm")
    plt.semilogy(epoch, derr_y_pred_n, marker='o', color='navy', linestyle='none')
    
    sess.run(train_op, feed_dict={x : X_train, t: y_train, jac : derr_y_pred_n})
    err = sess.run(cost, feed_dict={x : X_train, t: y_train, jac : derr_y_pred_n})
    
    if epoch % step ==0 :
        plt.figure("Log evolution error")
        plt.semilogy(epoch, err, marker='o', color='darkred', linestyle='none')
        plt.tight_layout()
        
        summary_str = mse_summary.eval(feed_dict={x : X_train, t: y_train, jac : derr_y_pred_n})
        file_writer.add_summary(summary_str, step)
        
        plt.pause(0.1)
file_writer.close()



