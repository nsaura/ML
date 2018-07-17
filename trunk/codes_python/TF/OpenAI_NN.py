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

#sudo pip install gym
#xvfb-run -s -screen 0 1400x900x24 python pour lancer xfvb

import gym

n_inputs = 4
n_hidden = 4

n_outputs = 1

try :
    tf.reset_default_graph()
except : 
    pass

env = gym.make("CartPole-v0")

#He initialization works better for layers with ReLu activation.
initializer = tf.contrib.layers.variance_scaling_initializer() #Voir plus bas

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)

logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits) #To output probability from 0.0 to 1.0

p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs]) #Probability daller à gauche ou à droite
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1) # Samples one integer given the log probabilty

y = 1. - tf.to_float(action) #Target
lr = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(lr)

grads_and_vars = optimizer.compute_gradients(cross_entropy) 
# Instead of minimize, because we want to tweak the gradients before apply them
# We store list of gradient vector / variables pairs (one pair per trainable variable)

gradients = [grad  for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars :
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed) #Placeholder et variable 

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, discount_rate) :
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    
    for step in reversed(range(len(rewards))) :
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate #Ajout des rewards precedent
        discounted_rewards[step] = cumulative_rewards
    
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate) :
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

n_iterations = 20
n_max_steps = 1000
discount_rate = 0.95
save_iterations = 10
n_games_per_update = 10


with tf.Session() as sess :
    init.run()
    
    for iteration in range(n_iterations) :
        all_rewards = []
        all_gradients = []
        
        for game in range(n_games_per_update) :
            current_rewards = []
            current_gradients = []
            
            obs = env.reset()
            
            for step in range(n_max_steps) :
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X:obs.reshape(1,-1)})
                print ("action_val = {}".format(action_val))
                print ("Shape : action_val = {}".format(action_val.shape))

                print ("gradients_val = {}".format(gradients_val))
                print ("Shape : gradients_val = {}".format(np.shape(gradients_val)))
                
                time.sleep(10)
                
                # Action : tf.multinomial(tf.log(p_left_and_right), num_samples=1)
                # Grads from compute_gradients
                
                obs, reward, done, info = env.step(action_val[0][0])
                
                current_rewards.append(reward)
                current_gradients.append(gradients_val) 
                
                env.render()
                
                if done :
                    break
            
            all_rewards.append(current_rewards)     # On les enregistre pour les n_max_step
            all_gradients.append(current_gradients) # On les enregistre pour les n_max_step
        
        feed_dict = {}
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)

        
        for var_index, grad_placeholder in enumerate(gradient_placeholders) :
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index] for game_index, rewards in enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)
            
            feed_dict[grad_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict) # Apply gradient
        
        if iteration % save_iterations == 0:    
            saver.save(sess, "./my_policy_net_pg.ckpt")
