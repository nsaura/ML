#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from os import mkdir
from os.path import join, abspath, exists, isfile
from os.path import split, splitext

import time

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

n_steps = 20
n_inputs = 1
n_neurons = 200
n_outputs = 1

plt.ion()

try :
    tf.reset_default_graph()
except :
    pass

config = tf.ConfigProto(device_count = {'GPU': 0})

sess = tf.InteractiveSession(config=config)

# Page 590 (a peu pres)
# On va faire pareil sans utiliser la couche de projection
with tf.variable_scope("rnn", initializer=tf.variance_scaling_initializer(), ) : 
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])

    # On crée une couche récurrente
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu, name="cell")
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)  

    # Puis on reshape cette sortie selon le nombre de neurones :

    # On empile ici toutes les sorties de la boucle recurrente
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons], name="firstreshape")

    # On projette ensuite tout ces solutions pour n'avoir qu'une seule sortie par entrée
    stacked_outputs =tf.layers.dense(stacked_rnn_outputs, n_outputs, name="fully" ) #sans activation car projection

    # Et on redistribue la valeurs pour chaque batch
    outputs = tf.reshape(stacked_outputs, [-1, n_outputs], name= "secondreshape")

## Loss function : MSE
learning_rate = 1e-3
loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

time1 = time.time()

sess.run(tf.global_variables_initializer())

dirtosave = abspath("./models/")
if exists(dirtosave) == False :
    mkdir(dirtosave)



chkpt_tosave = join(dirtosave, "RNN_Times_series2.ckpt")
saver.save(sess, chkpt_tosave)

n_iterations = 200
batch_size = 50

file = pd.read_csv("./../../Dataset/Data/Stocks/aa.us.txt")
Open = file["Open"][:1000]

#Open_filt2 = savgol_filter(Open, 51, 2)
Open_filt3 = savgol_filter(Open, 51, 3)
#Open_filt4 = savgol_filter(Open, 51, 4)
Open_filt5 = savgol_filter(Open, 51, 5)
Open_filt7 = savgol_filter(Open, 51, 7)

plt.figure("volume 1970")
plt.plot(range(len(Open)), Open, label="Real", c='b')
plt.plot(range(len(Open)), Open_filt3, label="Filtered3", c='grey')
plt.plot(range(len(Open)), Open_filt7, label="Filtered7", c='darkred')
#plt.plot(range(len(Open)), Open_filt2, label="Filtered2", c='orange')
#plt.plot(range(len(Open)), Open_filt4, label="Filtered4", c='purple')
#plt.plot(range(len(Open)), Open_filt7, label="Filtered7", c='darkred')
plt.legend()

xdata = np.zeros((20))
ydata = np.zeros((20))

a_1 = 0
for a in range(20, len(Open), 20) :
    xdata = np.block([ [xdata], [Open_filt7[a_1:a]] ])
    ydata = np.block([ [ydata], [Open_filt7[a_1+1:a+1]] ])
    
    a_1 = a
#xdata = np.delete(xdata, 0, axis=0)
#ydata = np.delete(ydata, 0, axis=0)

permute_indices = np.random.permutation(np.arange(len(ydata)))

xdata = xdata[permute_indices]
ydata = ydata[permute_indices]
means = xdata.mean(axis=0)
stds = xdata.std(axis=0)
mse=[]

for iteration in range(n_iterations) :
    for b in range(len(Open) // batch_size) :
        rand_index = np.random.choice(len(xdata), size=batch_size)
        X_batch = xdata[rand_index]
        y_batch = ydata[rand_index]
        
        X_batch = (X_batch - means) / stds
        
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))
        y_batch = y_batch.reshape((-1, n_outputs))
        
#        sys.exit()
        
        sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        
        if iteration % 100 == 0 :
            mse.append(loss.eval(feed_dict={X:X_batch, y:y_batch}))
            print (iteration, "\tMSE: ",mse[-1])

    if iteration % 100  == 0 and iteration != 0: 
        saver.save(sess, chkpt_tosave)
        print("Regular save")
            
time2 = time.time()
print ("Execution time : %.5f s" % abs(time1-time2))        


plt.figure("MSE Cost")
plt.semilogy(range(len(mse)), mse, c='grey')

xx = (Open_filt7[400:420] - means) / stds
yy = Open_filt7[401:421]

preds = sess.run(outputs, feed_dict={X:np.array([xx]).reshape((-1, n_steps, n_inputs))})

plt.figure("Comparaisons")
plt.plot(range(0, len(xx)+1), Open_filt7[400:421], label="True", marker='o', linestyle="None", ms=10)
plt.plot(range(1, len(xx)+1), preds.ravel(), label="Preds", marker='*', linestyle="None", ms=7)
plt.legend()

#tf.reset_default_graph()
metaload = chkpt_tosave + ".meta"
saver = tf.train.import_meta_graph(metaload)
saver.restore(sess, tf.train.latest_checkpoint('./models/'))

from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file(chkpt_tosave, tensor_name="secondreshape", all_tensor_names=True)

