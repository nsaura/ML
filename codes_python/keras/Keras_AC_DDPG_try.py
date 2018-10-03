#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge

import matplotlib.pyplot as plt
plt.ion()

import tensorflow as tf
graph = tf.get_default_graph()

import Keras_AC_DDPG as KAD

KAD = reload(KAD)

sess = tf.InteractiveSession()
state_size =  50
action_size = 50
BATCH_SIZE = 64
TAU = 0.001
lr_actor = 0.001
lr_critics = 0.0001
gamma = 0.96

trainnum = 0

noiselevel = 0.5

episodes = 5000     # Nombre d'episodes
train_times = 100   # Nombre d'iterations 

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 200

training_start = 100

buffer_size = 50*400

actor = KAD.ActorNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_actor)
critics = KAD.CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_critics)

X = np.linspace(0,1,50)

I = np.eye(50)
vvals = np.sin(X)
#fvals = np.zeros_like(X)

from collections import deque
replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

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
 
def modif_noise(noise_t, noise, ep, cpt):
    if cpt % 3 == 0 :
        if ep % 5 == 0 :
            noise_t[0] = np.random.randn(action_size) * noise
    elif cpt % 3 == 1 :
        if ep % 5 == 0 :
            noise_t[0] = np.random.randn(action_size) * noise * 2.
    else :
        noise_t[0] = np.zeros([1, action_size])
#### noise_t is changed

####    ####
#   play   #
#   play   #
####    ####

max_steps = 500000
reward = lambda fvals : np.transpose(fvals.ravel()-vvals).dot(I).dot(fvals.ravel()-vvals)*100

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
    step = 0
    s_t = np.zeros_like(X)
    print s_t
    penality = 0
    total_rew = 0
    noiselevel=0.5
    
    a_t = np.zeros([1, action_size])
    
    # Pour exploration de l'espace des actions
    noise = noiselevel
    noiselevel = noise * 0.999
    noise_t = np.zeros([1, action_size])
    
    for j in range(max_steps) :
        global graph
        with graph.as_default() :
#            print s_t
            a_t_original = actor.model.predict(np.array([s_t]))
#            return s_t
            
        noise *= 0.98
        modif_noise(noise_t, noise, j, ep)
        
        a_t = a_t_original + noise_t
#        print a_t_original.shape
#        print noise_t.shape
        
        a_t = a_t.ravel()
        
        s_t1 = a_t #Puisque la prediction est la sortie du nn
#        print s_t1.shape
#        print s_t1
        r_t = reward(s_t1)
        
#        print r_t
        if np.linalg.norm(np.array([s_t1.ravel()-vvals]), np.inf) < 1e-4 :
            goon = False
        else :
            goon = True
        
        replay_memory.append((s_t, a_t, r_t, s_t1, goon))
        s_t = s_t1
        
        total_rew += r_t
        
#        return 
####     ####
#   Train   #
#   Train   #
####     ####

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
            critics.model.optimizer.learnig_rate = lr_critics
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

if __name__ == "__main__" :
    while len(replay_memory) < BATCH_SIZE :
        play(15)
    
    train()
        
