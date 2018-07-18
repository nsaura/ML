#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import os.path as osp

import tensorflow as tf
import time
import gym

from collections import deque

env = gym.make("MsPacman-v0")
obs = env.reset()

mspacman_color = np.array([210, 164, 74]).mean()
# env.action_space --> Discrete(9) i.e 9 actions possible

try :
    tf.reset_default_graph()
except :
    pass

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    
    img[img==mspacman_color] = 0
    
    img = (img-128) / 128 - 1
    
    return img.reshape(88, 80, 1)


input_width = 80
input_height = 88
input_channels = 1

conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4,2,1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3

n_hidden_in = 64 * 11 * 10
n_hidden = 512

hidden_activation = tf.nn.relu
n_outputs = env.action_space.n

initializer = tf.contrib.layers.variance_scaling_initializer()

# For both Actions and Q_value networks
def q_network(X_state, name) :  
    prev_layer = X_state
    with tf.variable_scope(name) as scope :
        for n_maps, kernel_size, strides, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation) :
            
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size,
                                          strides=strides, padding=padding,
                                          activation=activation, kernel_initializer=initializer)

        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])

        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation = hidden_activation,
                                 kernel_initializer = initializer)
                                 
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
        
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    
    trainable_vars_by_name = {var.name[len(scope.name):] : var for var in trainable_vars}
    
    return outputs, trainable_vars_by_name
    

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

online_q_values, online_vars = q_network(X_state, name="q_neworks/online")
target_q_values, target_vars = q_network(X_state, name="q_neworks/target")

copy_ops = [target_var.assign(online_vars[var_name])
                for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(target_q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])

error = tf.abs(y - q_value)
clipped_error = tf.clip_by_value(error, 0.0, 1.0)
linear_error = 2 * (error-clipped_error)

loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

learning_rate = 0.001
momentum = 0.95

global_step = tf.Variable(0, trainable=False, name='global_step')

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Replay MeMory part. Set some tools
replay_memory_size = 5e5 #500,000
replay_memory =  deque([], maxlen=replay_memory_size)

def sample_memory(batch_size) :
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]
    
    for idx in indices :    
        memory = replay_memory[idx]
        
        for col, value in zip(cols, memory) :
            col.append(value)
    
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))
    
# epsilon-greedy policy 
# decrease form 1 to 0.1 in 2 million training steps

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2e6

def epsilon_greedy(q_values, step) :
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
    
    if np.random.rand () < epsilon :
        return np.random.randint(n_outputs)
    else :
        return np.argmax(q_values)
        
#Begin the training
n_steps = 4e6
training_start = 10000

training_interval = 4

save_steps = 1e3
copy_steps = 10000

discount_rate = 0.99
skip_start = 90

batch_size = 50
iteration = 0
checkpoint_path = "./mydqn.ckpt"
done=True

with tf.Session() as sess :
    if osp.isfile(checkpoint_path + ".index") :
        saver.restore(sess, checkpoint_path)
    else :
        init.run()
        copy_online_to_target.run()
    while True :
        step = global_step.eval()
        if step >= n_steps :
            break
        iteration += 1
        
        if done :
            obs = env.reset()
            for skip in range(skip_start) :
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)
        
        q_values = online_q_values.eval(feed_dict={X_state : [state]})
        action = epsilon_greedy(q_values, step)
        
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)
        
        env.render()
        
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state
        
        if iteration < training_start or iteration % training_interval != 0 :
            continue
        
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                        sample_memory(batch_size))
        next_q_values = target_q_values.eval(feed_dict={X_state : X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        
        y_val = rewards + continues*discount_rate*max_next_q_values
        
        training_op.run(feed_dict={X_state : X_state_val, X_action : X_action_val, y : y_val})
        
        if step % copy_steps == 0 : 
            copy_online_to_target.run()
        
        if step % save_steps == 0 :
            saver.save(sess, checkpoint_path)

