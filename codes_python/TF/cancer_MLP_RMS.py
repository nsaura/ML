#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

from sklearn.model_selection    import  train_test_split
from sklearn.datasets           import  load_breast_cancer
cancer = load_breast_cancer()


#import ./../datasets_mglearn as dsets

#try:
#    reload  # Python 2.7
#except NameError:
#    try:
#        from importlib import reload  # Python 3.4+
#    except ImportError:
#        from imp import reload  # Python 3.0 - 3.3

#dsets = reload(dsets)

import tensorflow as tf

def y2indicator(y, T):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, T))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)

def find_divisor(N) :
    div_lst = []
    for i in range(N) :
        if N % (i+1) == 0: div_lst.append(i+1)
    # Since last divisor is N itself, we keep the prior last one
    return div_lst
#-----------------------------------------------------------------
#-----------------------------------------------------------------
sess = tf.InteractiveSession()

lr = 0.0004

X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

X_train_mean= X_train.mean(axis=0)
X_train = (X_train-X_train_mean)/(X_train.std(axis=0))
X_test =  (X_test-X_train_mean)/(X_train.std(axis=0))
# y_train labels known

#----- Building graph -----#
N1 = 300                #Nombre de noeuds pour w^1_ij
N2 = 50
T = y_train.shape[0]    #Nombre de noeuds pour w^2_ij
K = 2                   # Nombre de classe à la sortie

Nraw_Xtrain, Ncol_Xtrain = X_train.shape #Pour faciliter les notations

print("\x1b[0;30;47m RMSPropOptimizer \x1b[0m")
start = time.time()

sess = tf.InteractiveSession()
#   1-- Weights and Bias 
#   One HL here (IN->1 , 1->OUT)
w1_init = np.random.randn(Ncol_Xtrain,N1) / 28
b1_init = np.zeros(N1)

w2_init = np.random.randn(N1, N2) / np.sqrt(N1)
b2_init = np.random.randn(N2)

w3_init = np.random.randn(N2,N2) / np.sqrt(N2)
b3_init = np.random.randn(N2)

w4_init = np.random.randn(N2, K) / np.sqrt(N2)
b4_init = np.random.randn(K)


#   2-- TF variables and expressions
x = tf.placeholder(tf.float32, (None, Ncol_Xtrain))
t = tf.placeholder(tf.float32, (None, K))

w1 = tf.Variable(w1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32)) 

w2 = tf.Variable(w2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))

w3 = tf.Variable(w3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))

w4 = tf.Variable(w4_init.astype(np.float32))
b4 = tf.Variable(b4_init.astype(np.float32))

#   3-- define the model 
## Remind a nn consists in several nodes that transformed linear expressions into non-linear.
## Here we use relu fonction ( max(0,w.x) )
z1 = tf.nn.relu( tf.matmul(x, w1) + b1) 
z2 = tf.nn.relu( tf.matmul(z1, w2) + b2)
z3 = tf.nn.relu( tf.matmul(z2, w3) + b3)

y_ = tf.matmul(z3, w4) + b4

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=t))

#   4-- We train the model and foresee to analyse prediction
train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
# momentum helps the NN when Gradient is stuck in a local minima
# In this case, it is a gain of time
predict_op = tf.argmax(y_,1)

#   5-- We prepare tf to run
costs = []
max_epoch, err, epoch = 30, 1, 0

batch_sz = find_divisor(Nraw_Xtrain)[-3]
#print batch_sz

n_batches= Nraw_Xtrain // batch_sz
y_train_ind = y2indicator(y_train, K)
y_test_ind = y2indicator(y_test, K)

sess.run(tf.global_variables_initializer())

for epoch in xrange(max_epoch) :
    for j in xrange(n_batches) :
        X_batch = X_train[j*batch_sz:(j*batch_sz + batch_sz)]
        y_batch = y_train_ind[j*batch_sz:(j*batch_sz + batch_sz)]
        
        sess.run(train_op, feed_dict={x:X_batch, t:y_batch})
        
        prediction = sess.run(predict_op,feed_dict={x: X_test})
    ## predict op needs to e fed with x because it calls y_ which is tf.matmul(z1,w2) + b2 where z1 is defined as a function of x
        err = error_rate(prediction, y_test)
        
        test_cost = sess.run(cost, feed_dict={x:X_test, t:y_test_ind})
#            print("Cost / err at epoch ep = %d, %.3f / %.3f" %(epoch, test_cost, err))
        costs.append(test_cost) 

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(t,1))
print(correct_prediction.eval(feed_dict={x:X_test, t:y_test_ind})) 

R = correct_prediction.eval(feed_dict={x:X_test, t:y_test_ind})

res = {k:v for k,v in zip (["False","True"], np.bincount(R))}
perc = res["True"]/float(res["True"] + res["False"])

print("{}".format(perc))

sess.close()
end = time.time()

print ("Running Time : {}".format(end-start))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

