#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys 
import numpy as np
import os.path as osp

#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge

## Imports des fichiers ##
code_case_dir = osp.abspath(osp.dirname("../cases/"))
sys.path.append(code_case_dir)

import solvers
import tensorflow as tf
import matplotlib.cm as cm
import Keras_AC_DDPG as KAD
from collections import deque
import matplotlib.pyplot as plt

graph = tf.get_default_graph()

KAD = reload(KAD)

def samples_memories(BATCH_SIZE):
    indices = np.random.permutation(len(replay_memory))[: BATCH_SIZE]
    # cols[0] : state
    # cols[1] : action
    # cols[2] : reward
    # cols[3] : next_state
    # cols[4] : continue
    cols = [[],[],[],[],[]]
    
    for idx in indices :
        memory = replay_memory[idx]
        for col, value in zip(cols, memory) :
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

## - - - - - - - - - - - - ##
##- - - - - - - - - - - - -##
## - - - - - - - - - - - - ##
##  Initialisation du pbs  ##
## - - - - - - - - - - - - ##
##- - - - - - - - - - - - -##
## - - - - - - - - - - - - ##

L  = 2
Nx = 100
dx = float(L)/(Nx-1)
X  = np.arange(0,L+dx, dx)

tfi = 1.5
Nt = 1000
dt = tfi / Nt
T  = np.arange(0, tfi+dt, dt)

r = dt / dx 

f = lambda u : 0.5*u**2
fprime = lambda u : u

max_steps = 500
numbercases = 3 

replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 20

BATCH_SIZE = 500
TAU = 0.001
lr_actor = 0.001
lr_critics = 0.0001
gamma = 0.96

sess = tf.InteractiveSession()

state_size = action_size = Nx

actor = KAD.ActorNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_actor, HIDDEN1_UNITS, HIDDEN2_UNITS)
critics = KAD.CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_critics, HIDDEN1_UNITS, HIDDEN2_UNITS)
#----------------------------------------------
def action_with_burger(state) :
    """
    On utilise Burger comme prediction
    """
    next_state = np.zeros_like(state)
    
    for j in range(1,len(state)-1) :
        next_state[j] = solvers.timestep_roe(state, j, r, f, fprime)
    next_state[0] = next_state[-2]
    next_state[-1] = next_state[1]
    return next_state
#----------------------------------------------
def action_with_delta_Un(state, action) :
    """
    L'action est delta Un. Ici on transforme state avec action
    """
    next_state = np.array([state[j] + action[j] for j in range(len(state))])
    return next_state
#----------------------------------------------    
def reward (next_state, state) :
    temp_term = np.array([(next_state[j] - state[j]) / dt for j in range(len(state))])
    square_term = np.zeros_like(state)
    for j in range(len(state[1:-1])) :
        square_term[j] =  (state[j+1]**2 - state[j-1]**2) * 0.25 / dx
    
    square_term[0] = (state[1]**2 - state[-1]**2) * 0.25 / dx
    square_term[-1] = (state[1]**2 - state[-2]**2) * 0.25 / dx
    
    return np.sum(temp_term + square_term)
#----------------------------------------------
def play(ep):    
    """
    Function that plays a certain number of iterations of the game (until it finishes).
    This function can also be used to construct the rpm.
    The loop on the episodes is outside of the function. 
    
    Arguments :
    -----------
    ep : episode on which the function is requiered
    """
    episode_memory = []
    s_t = np.zeros_like(X)
    
    total_rew = 0
    a_t = np.zeros([1, s_t.size])
    
    # Pour exploration de l'espace des actions
#    noise = noiselevel
#    noiselevel = noise * 0.999
#    noise_t = np.zeros([1, action_size])
    
    for j in range(max_steps) :
        global graph
        with graph.as_default() :
            a_t_original = actor.model.predict(np.array([s_t]))
            
        
        a_t = a_t_original# + noise_t
        
        a_t = a_t.ravel()
        s_t1 = action_with_delta_Un(s_t, a_t)
        
#        return s_t, a_t, st_1

        r_t = reward(s_t1, s_t)

        print ("state :\n{}".format(s_t))
        print ("action :\n{}".format(a_t))
        print ("reward :\n{}".format(r_t))
        print ("next state :\n{}".format(s_t1))
        
        if r_t < 0.001 :   
            goon = False
        else :
            goon = True
        
        if len(replay_memory) < replay_memory_size :
            replay_memory.append((s_t, a_t, r_t, s_t1, goon))
        
        else :
            if abs(np.random.randn()) > 0.5 :
                replay_memory.popleft() # Pop the leftmost element 
            else: 
                replay_memory.pop() # Pop the rightmost element 
        
        s_t = s_t1
        print ("next_state :\n{}".format(s_t1))
        
        total_rew += r_t
        print "final"
#----------------------------------------------        
def train () :
    """
    Function to train the actor and the critic target networks
    It has to be after the construction of the replay_memory
    """
    loss = 0
    trainnum = 0 
    global graph
    for T in range(train_times) :
        states, actions, rewards, next_states, goons = (samples_memories(BATCH_SIZE))
        
#       Just to test
#        print ("states shape : ", states.shape)
#        print ("actions shape : ", actions.shape)
#        print ("rewards shape: ", rewards.shape)
#        print ("goons shape  : ", goons.shape)
        y_t = np.asarray([0.0]*BATCH_SIZE)
        rewards = np.concatenate(rewards)
        
        with graph.as_default() :
            # Q function evaluation on the target graphs 
            target_q_values = critics.target_model.predict(
                            [next_states, actor.target_model.predict(next_states)])
        target_q_values = target_q_values.reshape([1, target_q_values.shape[0]])[0]
        
        for k in range(BATCH_SIZE) :
            y_t[k] = rewards[k] + goons[k]*gamma*target_q_values[k]
        
        with graph.as_default():
            # We set lr of the critic network
            critics.model.optimizer.lr = lr_critics
            
            logs = critics.model.train_on_batch([states, actions], y_t) #(Q-y)**2
            
            a_for_grad = actor.model.predict(states)
            grad = critics.gradients(states, a_for_grad)
            
            actor.train(states, grad, learning_rate=lr_actor)

            actor.target_train()
            critics.target_train()         
            
            plt.figure("Comparaison")
            plt.plot(X, vvals, label='True', c='k')
            plt.plot(X, actor.target_model.predict(states[0].reshape(1,-1)).ravel(), label="Process", c='yellow', marker='o', fillstyle="none", linestyle='none')
            plt.show()
##            plt.legend()
        loss += logs
        
        trainnum += 1
    print ("Train ", len(replay_memory), loss)
#----------------------------------------------
    
if __name__ == "__main__" :
    while len(replay_memory) < BATCH_SIZE :
        play()
    
    train()

