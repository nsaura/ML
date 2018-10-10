#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys 
import time 
import noise 
import decays
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
noise = reload(noise)
decays = reload(decays)
solvers = reload(solvers)

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

max_steps = 250
numbercases = 10

episodes = 10

replay_memory_size =  5e3
replay_memory = deque([], maxlen=replay_memory_size)

HIDDEN1_UNITS = 100
HIDDEN2_UNITS = 100

BATCH_SIZE = int(128)
TAU = 0.1
lr_actor_init = 1e-2
lr_critics_init = 1e-3

lr_actor_final = 1e-4
lr_critics_final = 1e-4


EpsForNoise_init = 0.5
EpsForNoise_fina = 0.05

step_new_batch = 1

gamma = 0.9

sess = tf.InteractiveSession()

state_size = action_size = Nx

actor = KAD.ActorNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_actor_init, HIDDEN1_UNITS, HIDDEN2_UNITS)
critics = KAD.CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, lr_critics_init, HIDDEN1_UNITS, HIDDEN2_UNITS)

if plt.fignum_exists("Reward / 50 steps") :
    plt.figure("Reward / 50 steps")
    plt.clf()
    plt.pause(0.01)

#----------------------------------------------
def action_with_burger(state) :
    """
    On utilise Burger comme prediction
    """
    next_state = np.zeros_like(state)
    
    for j in range(1,len(state)-1) :
        next_state[j] = solvers.timestep_roe(state, j, r, f, fprime)
    next_state[0] = next_state[-3]
    next_state[-1] = next_state[2]
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
    
    return np.log(1./np.linalg.norm((temp_term + square_term), np.inf))*10
#----------------------------------------------
def play(u_init):    
    """
    Function that plays a certain number of iterations of the game (until it finishes).
    This function can also be used to construct the rpm.
    The loop on the episodes is outside of the function. 
    
    Arguments :
    -----------
    u_init : To set the initialisation
    """
    episode_memory = []
    s_t = u_init
    
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
            
        epsilon = decays.create_decay_fn("linear",
                                         curr_step=j,
                                         initial_value=EpsForNoise_init,
                                         final_value=EpsForNoise_fina,
                                         max_step=max_steps)
        
#        if j % 200 == 0 :
#            print ("steps = %d\t eps = %0.8f" %(j, epsilon))
        
        args = {"rp_type" : "ornstein-uhlenbeck",
                "n_action" : 1,
                "rp_theta" : 0.1,
                "rp_mu" : 0.,
                "rp_sigma" : 0.2,
                "rp_sigma_min" : 0.05}
        
        a_t = a_t_original + epsilon*noise.create_random_process(args).sample()
        a_t = a_t.ravel()
        
        a_tt = np.copy(a_t)
        for a in range(len(a_t)) :
            if a_tt[a] > 1. :
                a_tt[a] = 1.
            elif a_tt[a] < -1. :
                a_tt[a] = -1.
            else :
                pass
        a_t = np.array([a for a in a_tt])
        s_t1 = action_with_delta_Un(s_t, a_t)
                
        
#        return s_t, a_t, st_1

        r_t = reward(s_t1, s_t)

#        print ("state :\n{}".format(s_t))
#        print ("action :\n{}".format(a_t))
#        print ("reward :\n{}".format(r_t))
#        print ("next state :\n{}".format(s_t1))
        
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
#        print ("next_state :\n{}".format(s_t1))
        
        total_rew += r_t
        
#        if len(replay_memory) % 200 ==0 :
#            print ("Memory size = %d" % len(replay_memory))
#----------------------------------------------        
def train () :
    """
    Function to train the actor and the critic target networks
    It has to be after the construction of the replay_memory
    """
    loss = 0
    losses = []
    trainnum = 0 
    global graph
    for ep in range(episodes) :
        loss = 0
        trainnum = 0
        
        critics.model.optimizer.lr = lr_critics_init
        curr_lr_actor = lr_actor_init
        if ep % 5 == 0 :
                    print ("episodes = %d\t lr_actor_curr = %0.8f \tlr_crits_curr = %0.8f"\
                                        %(ep, curr_lr_actor, critics.model.optimizer.lr))
        it = 0
        iterok=False    
                             
        totalcounter = 0

        while iterok == False and it < max_steps :
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
                logs = critics.model.train_on_batch([states, actions], y_t) #(Q-y)**2
                
                a_for_grad = actor.model.predict(states)
                grad = critics.gradients(states, a_for_grad)
                
                actor.train(states, grad, learning_rate=curr_lr_actor)

                actor.target_train()
                critics.target_train()         
                
                # In this section we decide wheither we continue or not
                for i in range(it+1) :
                    if i ==0 :
                        test_states = np.sin(np.pi*2.*X/float(L))
                    else :
                        test_states = test_next_states

                    test_Delta = actor.target_model.predict(test_states.reshape(1,-1))
                    
                    test_next_states = action_with_delta_Un(test_states.reshape(1,-1), test_Delta)
                
                    test_reward = reward(test_next_states.ravel(), test_states)
                    
                    plt.figure("Prediction")
                    plt.clf() 
                    plt.plot(X, test_next_states.ravel(), label="New state based on actor", color='red')
                    plt.plot(X, action_with_burger(test_states), label="True next profile", fillstyle='none', marker='o', color='blue')
                    plt.legend()
                    plt.pause(5)
                    
                    
                    plt.figure("Evolution de delta Un")
                    plt.plot(X, test_Delta.ravel())
                    plt.pause(0.01)
                    
                    if np.abs(test_reward) < 0.00001 :
                        iterok = True
                    else :
                        iterok = False
                
                if iterok == True :
                    it += 1
                else :
                    it = 0
                
                totalcounter += 1
                
                print ("Total iteration : %d. \tFalse iteration : %d\t reward = %0.4f" %(totalcounter, it, test_reward))
                
                if np.isnan(test_reward) == True :
                    sys.exit("Nan")
    #            plt.figure("Comparaison")
    #            plt.plot(X, vvals, label='True', c='k')
    #            plt.plot(X, actor.target_model.predict(states[0].reshape(1,-1)).ravel(), label="Process", c='yellow', marker='o', fillstyle="none", linestyle='none')
    #            plt.show()
    ###            plt.legend()
                
#                critics.model.optimizer.lr = lr_critics_init
#                curr_lr_actor = lr_actor_init
                critics.model.optimizer.lr = decays.create_decay_fn("linear",
                                                        curr_step = totalcounter % step_new_batch,    
                                                        initial_value = lr_critics_init,
                                                        final_value = lr_critics_final,
                                                        max_step = step_new_batch)
                curr_lr_actor = decays.create_decay_fn("linear",
                                           curr_step = totalcounter % step_new_batch,    
                                           initial_value = lr_actor_init,
                                           final_value = lr_actor_final,
                                           max_step = step_new_batch)
                
                print ("totalcounter = %d, \t lr_actor = %.6f\t lr_crits = %.6f" %(totalcounter, curr_lr_actor, critics.model.optimizer.lr))
                
                if totalcounter % 50 == 0 :
                    plt.figure("Reward / 50 steps")
                    plt.semilogy(totalcounter, test_reward, c='purple', linestyle='none', marker='o', ms = 4)
                    plt.pause(0.01)
                    
                    
                if totalcounter % step_new_batch == 0 :
                    replay_memory.clear()
                    u_init = np.zeros((1, X.size))
                    pi_line = np.linspace(0.4, 1.4, 50)
                    amplitude = [np.random.choice(pi_line) for i in range(1)]
                    for i, amp in enumerate(amplitude) :
                        u_init[i] = amp*np.sin(2*np.pi/L*X)
                    
                    print ("Cleaning of the batch %d ..." %((totalcounter // step_new_batch) -1))
                    while len(replay_memory) < replay_memory_size : 
                        for u in u_init :
                            play(u)
                        time.sleep(1)
                    print ("The batch %d is ready to be used" %( totalcounter // step_new_batch ))
                    
            loss += logs
            
            trainnum += 1
        print ("Episode = %d :" % ep)
        print ("total loss = %.4f" %loss)
        
        losses.append(loss)
        
        plt.figure("Evolution de Loss sur un episodes vs iteration")
        plt.semilogy(ep, loss, marker='o', ms=6, linestyle="none", c='r')
        plt.pause(0.5)
    
    return losses
#----------------------------------------------
    
if __name__ == "__main__" :
    lossess = dict()
    n_init = numbercases
    u_init = np.zeros((n_init, X.size))
    pi_line = np.linspace(0.4, 1.4, 50)
    amplitude = [np.random.choice(pi_line) for i in range(n_init)]

    for i, amp in enumerate(amplitude) :
        u_init[i] = amp*np.sin(2*np.pi/(L-2.*dx)*(X-dx)) 
    

#    plt.figure("Sinus initiaux")
#    for j, u in enumerate(u_init) :
#        plt.plot(X, u, label="amplitude = %.4f" % amplitude[j])
#    plt.legend()
    
    colors = iter(cm.plasma_r(np.arange(500)))
    
    redo = 1
    for r in range(redo) :
        print ("The batch %d is being constructed ..." %(r))
        replay_memory.clear()
        while len(replay_memory) < replay_memory_size : 
            for u in u_init :
                play(u)
        print ("The batch %d is ready to be used" %(r))
        time.sleep(1)
        
        print ("Training start")
        curr_loss = train()
        lossess[r] = curr_loss
        
        print ("Going to delete the curr figures, the results will be displayed in another figs...")
        time.sleep(5)

        plt.figure("Evolution de Loss sur un episodes vs iteration")
        plt.clf()
        
        plt.figure("Comparaison of the training loss with different replay memories")
        for j in range(27) :
            next(colors)
        plt.semilogy(range(episodes), curr_loss, c = next(colors), label="Loss replay %d" % (r))
        plt.xlabel("Number of episodes")
        plt.ylabel("Loss values")
        plt.legend()
        
        plt.pause(0.01)
        
    

