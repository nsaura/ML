#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys 
import time 
import noise 
import decays
import numpy as np
import os.path as osp
from os import mkdir
#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge

## Imports des fichiers ##
code_case_dir = osp.abspath(osp.dirname("../cases/"))
sys.path.append(code_case_dir)

utils_dir = osp.abspath(osp.dirname("./../utils/"))
sys.path.append(utils_dir)

import solvers
import tensorflow as tf
import matplotlib.cm as cm
import Keras_AC_DDPG as KAD
from collections import deque
import matplotlib.pyplot as plt

import plot_from_file as pff
pff = reload(pff)


graph = tf.get_default_graph()

plt.close("all")

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 

KAD = reload(KAD)
noise = reload(noise)
decays = reload(decays)
solvers = reload(solvers)

colors = iter(cm.plasma_r(np.arange(600)))
print colors 


#----------------------------------------------
def samples_memories(BATCH_SIZE):
    indices = np.random.permutation(len(replay_memory))[: BATCH_SIZE]
    # cols[0] : state
    # cols[1] : action
    # cols[2] : reward
    # cols[3] : next_state
    # cols[4] : done
    cols = [[],[],[],[],[]]
    
    for idx in indices :
        memory = replay_memory[idx]
        for col, value in zip(cols, memory) :
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))
#----------------------------------------------
## - - - - - - - - - - - - ##
##- - - - - - - - - - - - -##
## - - - - - - - - - - - - ##
##  Initialisation du pb   ##
## - - - - - - - - - - - - ##
##- - - - - - - - - - - - -##
## - - - - - - - - - - - - ##

cst_simu = dict()
cst_REIL = dict()

# Constantes spatiales 
cst_simu['L'] = 2
cst_simu["Nx"] = 100

# Constantes temporelles 
cst_simu["tf"] = 1.5
cst_simu["Nt"] = 40

cst_simu["dx"] = float(cst_simu["L"])/(cst_simu["Nx"]-1)
cst_simu["dt"] = float(cst_simu["tf"])/(cst_simu["Nt"])
cst_simu["r"] = cst_simu["dt"] / cst_simu["dx"]

fprime = lambda u : u
f = lambda u : 0.5*u**2

X  = np.arange(0, cst_simu["L"] + cst_simu["dx"], cst_simu["dx"])
T  = np.arange(0, cst_simu["tf"] + cst_simu["dt"], cst_simu["dt"])

cst_simu["max_steps"] = 150

#
# Constantes for the REIL
#
cst_REIL["TAU"] = 0.01
cst_REIL["gamma"] = 0.9
cst_REIL["episodes"] = 10
cst_REIL["steps_before_change"] = 20

# Deque
cst_REIL["BATCH_SIZE"] = int(128) 
cst_REIL["replay_memory_size"] = 5e3
replay_memory = deque( [], maxlen=cst_REIL["replay_memory_size"] )


# Number of HN for both actor and critics NN
cst_REIL["HIDDEN1_UNITS"] = 256
cst_REIL["HIDDEN2_UNITS"] = 256


# Lr decays for actor or critics
cst_REIL["max_steps_lr"] = 50

cst_REIL["lr_actor_init"] = 1e-3
cst_REIL["lr_actor_final"] = 1e-4

cst_REIL["lr_critics_final"] = 1e-3
cst_REIL["lr_critics_init"] = 1e-4


# Eps (for noise) decays
cst_REIL["EpsForNoise_init"] = 1
cst_REIL["EpsForNoise_fina"] = 0.8

facteur_rew = 100

sess = tf.InteractiveSession()

state_size = action_size = cst_simu["Nx"]

act = KAD.ActorNetwork(sess, state_size, action_size, cst_REIL["BATCH_SIZE"], 
                                                      cst_REIL["TAU"], 
                                                      cst_REIL["lr_actor_init"],
                                                      cst_REIL["HIDDEN1_UNITS"],
                                                      cst_REIL["HIDDEN2_UNITS"])

cri = KAD.CriticNetwork(sess, state_size, action_size, cst_REIL["BATCH_SIZE"], 
                                                      cst_REIL["TAU"], 
                                                      cst_REIL["lr_critics_init"],
                                                      cst_REIL["HIDDEN1_UNITS"],
                                                      cst_REIL["HIDDEN2_UNITS"])
#----------------------------------------------
def save_weights() :
    act.target_model.save_weights("./ACweights/actmodel.h5", overwrite=True)
    cri.target_model.save_weights("./ACweights/crimodel.h5", overwrite=True)
    print("Saved")
#----------------------------------------------
def load_weights() :
    act.model.load_weights("./ACweights/actmodel.h5")
    act.model.load_weights("./ACweights/actmodel.h5")
    
    act.target_model.load_weights("./ACweights/actmodel.h5")
    cri.target_model.load_weights("./ACweights/crimodel.h5")
    print("Loaded")
#----------------------------------------------

if osp.exists("./ACweights/") == False :
    mkdir("./ACweights/")

save_weights()

#----------------------------------------------
def reward (next_state, state) :
    
#    print("\nIN REWARD\n")
#    print ("next_state.shape", next_state.shape)
#    print ("state.shape", state.shape)
#    
    temp_term = np.array([(next_state[j] - state[j]) / cst_simu["dt"] for j in range(len(state))])
    square_term = np.zeros_like(state)

    for j in range(len(state[1:-1])) :
        square_term[j] =  (state[j+1]**2 - state[j-1]**2) * 0.25 / cst_simu["dx"]
    
    square_term[0] = (state[1]**2 - state[-1]**2) * 0.25 / cst_simu["dx"]
    square_term[-1] = (state[1]**2 - state[-2]**2) * 0.25 / cst_simu["dx"]
    
    return 1./(np.linalg.norm((temp_term + square_term), 2))*facteur_rew

#----------------------------------------------
def action_with_burger(state) :
    """
    On utilise Burger comme prediction
    """
    next_state = np.zeros_like(state)
    
    for j in range(1,len(state)-1) :
        next_state[j] = solvers.timestep_roe(state, j, cst_simu["r"], f, fprime)
    
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
#def play_without_burger(u_init):    
#    """
#    Function that plays a certain number of iterations of the game (until it finishes).
#    This function can also be used to construct the rpm.
#    The loop on the episodes is outside of the function. 
#    
#    Arguments :
#    -----------
#    u_init : To set the initialisation
#    """
#    episode_memory = []
#    s_t = u_init
#    
#    total_rew = 0
#    a_t = np.zeros([1, s_t.size])
#    
#    # Pour exploration de l'espace des actions
##    noise = noiselevel
##    noiselevel = noise * 0.999
##    noise_t = np.zeros([1, action_size])
#    
#    for j in range(cst_simu["max_steps"]) :
#        global graph
#        with graph.as_default() :
#            a_t_original = act.model.predict(np.array([s_t]))
#            
#        epsilon = decays.create_decay_fn("linear",
#                                         curr_step=j,
#                                         initial_value=cst_REIL['EpsForNoise_init'],
#                                         final_value=cst_REIL['EpsForNoise_fina'],
#                                         max_step=cst_simu["max_steps"])
#        
#        args = {"rp_type" : "ornstein-uhlenbeck",
#                "n_action" : 1,
#                "rp_theta" : 0.1,
#                "rp_mu" : 0.,
#                "rp_sigma" : 0.2,
#                "rp_sigma_min" : 0.05}
#        
#        a_t = a_t_original + epsilon*noise.create_random_process(args).sample()
#        a_t = a_t.ravel()
#        
#        s_t1 = action_with_delta_Un(s_t, a_t)
#        
#        r_t = reward(s_t1, s_t)

##        print ("state :\n{}".format(s_t))
##        print ("action :\n{}".format(a_t))
##        print ("reward :\n{}".format(r_t))
##        print ("next state :\n{}".format(s_t1))
#        
#        if abs(r_t) < 0.01 :   
#            goon = False
#        else :
#            goon = True
#        
#        if len(replay_memory) < replay_memory_size :
#            replay_memory.append((s_t, a_t, r_t, s_t1, goon))
#        
#        else :
#            if abs(np.random.randn()) > 0.5 :
#                replay_memory.popleft() # Pop the leftmost element 
#            else: 
#                replay_memory.pop() # Pop the rightmost element 
#        
#        s_t = s_t1
#----------------------------------------------        
def play_with_burger(u_init):    
    """
    Function that plays a certain number of iterations of the game (until it finishes).
    This function can also be used to construct the rpm.
    The loop on the episodes is outside of the function. 
    
    we use timestep_roe to provide next steps
    
    Arguments :
    -----------
    u_init : To set the initialisation
    """
    episode_memory = []
    s_t = u_init
    
    total_rew = 0
    a_t = np.zeros([1, s_t.size])
    plotplot = 0
    
    f = open("play_reward.txt", "w")
    f.close()
    
    for j in range(cst_simu["max_steps"]) :
        s_t1 = action_with_burger(s_t)
        a_t_original = np.array([s_t1[i] - s_t[i] for i in range(len(s_t))])
        
#        plt.figure("Burger Plot")
#        plt.clf()
#        plt.plot(X, s_t, label="st")
#        plt.plot(X, s_t1, label="s_t1")
#        plt.legend()
#        plt.pause(0.01)
        
        
#        epsilon = decays.create_decay_fn("linear",
#                                         curr_step=j,
#                                         initial_value=cst_REIL["EpsForNoise_init"],
#                                         final_value=cst_REIL["EpsForNoise_fina"],
#                                         max_step=cst_simu["max_steps"])
#        
#        args = {"rp_type" : "ornstein-uhlenbeck",
#                "n_action" : 1,
#                "rp_theta" : 0.1,
#                "rp_mu" : 0.,
#                "rp_sigma" : 0.2,
#                "rp_sigma_min" : 0.05}
#        
#        a_t = a_t_original + epsilon*noise.create_random_process(args).sample()
        a_t = a_t_original
        
        gg = [abs(a) > 1. for a in a_t]
        
#        if plotplot == 0 :
#            plt.figure("Action and noise")
#            plt.plot(X, a_t_original.ravel(), color='k')
#            plt.plot(X, a_t.ravel(), color='green')
            
#        plotplot +=1
        
        a_t = a_t.ravel()

        r_t = reward(s_t1, s_t)
        f = open("play_reward.txt", "a")
        f.write("%.4f\n" %(r_t))
        f.close()
                
#        print ("state :\n{}".format(s_t))
#        print ("action :\n{}".format(a_t))
#        print ("reward :\n{}".format(r_t))
#        print ("next state :\n{}".format(s_t1))
        
        if r_t > 0.1*facteur_rew and r_t < 3000 :   
            goon = False
            rew = r_t
            print("Not bad")
        elif r_t > 3000 :
            rew = -r_t
            print ("Very bad")
            goon = False
        else :
            goon = True
            rew = r_t
            print ("Yes")
        
        if len(replay_memory) < cst_REIL["replay_memory_size"] :
            replay_memory.append((s_t, a_t, rew, s_t1, goon))
        
        else :
            if abs(np.random.randn()) > 0.5 :
                replay_memory.popleft() # Pop the leftmost element 
            else: 
                replay_memory.pop() # Pop the rightmost element 
        
        s_t = s_t1
#----------------------------------------------        
def train (u_init) :
    """
    Function to train the actor and the critic target networks
    It has to be after the construction of the replay_memory
    """
    loss = 0
    losses = []
    trainnum = 0 
    
    f = open("rewards.txt", "w")
    f.close()
    
    f = open("delta_max.txt", "w")
    f.close()
    
    global graph
    global colors 
    for ep in range(cst_REIL["episodes"]) :
        loss = 0
        trainnum = 0
        
        cri.model.optimizer.lr = cst_REIL["lr_critics_init"]
        curr_lr_actor = cst_REIL["lr_actor_init"]
        if ep % 5 == 0 :
                    print ("episodes = %d\t lr_actor_curr = %0.8f \tlr_crits_curr = %0.8f"\
                                        %(ep, curr_lr_actor, cri.model.optimizer.lr))
        it = 0
        iterok=False    

        delta_max = []
                             
        totalcounter = 0
        along_reward = []
        rew = 0
        while iterok == False and it < cst_simu["max_steps"] :
            load_weights()
            states, actions, rewards, next_states, goons = (samples_memories(cst_REIL["BATCH_SIZE"]))
            delta_number_step = []
#       Just to test
#        print ("states shape : ", states.shape)
#        print ("actions shape : ", actions.shape)
#        print ("rewards shape: ", rewards.shape)
#        print ("goons shape  : ", goons.shape)
            y_t = np.asarray([0.0]*cst_REIL["BATCH_SIZE"])
            rewards = np.concatenate(rewards)
            rwrds = np.copy(rewards)
#            rewards = np.array([10*rr for rr in rwrds])
            
            with graph.as_default() :
                # Q function evaluation on the target graphs 
                target_q_values = cri.target_model.predict(
                                [next_states, act.target_model.predict(next_states)])
            target_q_values = target_q_values.reshape([1, target_q_values.shape[0]])[0]
            
            for k in range(cst_REIL["BATCH_SIZE"]) :
                y_t[k] = rewards[k] + goons[k]*cst_REIL["gamma"]*target_q_values[k]
            
            with graph.as_default():
                # We set lr of the critic network
                logs = cri.model.train_on_batch([states, actions], y_t)
                
                a_for_grad = act.model.predict(states)
                grad = cri.gradients(states, a_for_grad)
                
                act.train(states, grad, learning_rate=curr_lr_actor)

                act.target_train()
                cri.target_train()         
                
                # In this section we decide wheither we continue or not
                # We use those actor and critic target networks for the next steps_before_change steps
                rew = 0
                cnt = 0
                for step in range(cst_REIL["steps_before_change"]) :
                    curr_it = it
                    if curr_it == 0 and step == 0 :
                        test_states = u_init
                        curr_true = u_init
                    else :
                        test_states = test_next_states
                        curr_true = next_true
                        
                    next_true = action_with_burger(curr_true)
                    
                    test_Delta = act.target_model.predict(test_states.reshape(1,-1)) #Action
                    test_Delta = test_Delta.ravel()
                    
                    delta_number_step.append(test_Delta.max())
                    print delta_number_step
                    
                    test_next_states = action_with_delta_Un(test_states.reshape(1,-1), test_Delta)
                    test_next_states = test_next_states.ravel()
                    
                    test_reward = reward(test_next_states, test_states)
                    rew += test_reward
                    
                    along_reward.append(test_reward)

                    f = open("rewards.txt", "a")
                    f.write("%.5f \n" %(test_reward))
                    f.close()
                    if cnt % 5== 0 :
                        plt.figure("Prediction")
                        plt.clf() 
                        plt.plot(X, test_next_states,
                                        label="New state based on actor", color='red')
                        plt.plot(X, next_true,
                                        label="True next profile", fillstyle='none', marker='o', color='blue')
                        plt.ylim((-2,2))
                        plt.legend()
                                                
                        try :
                            c = next(colors)
                        except StopIteration:
                            del colors 
                            colors = iter(cm.cubehelix_r(np.arange(600)))
                            global colors
                            c = next(colors)
                        
                        plt.figure("Evolution de delta Un")
                        plt.plot(X, test_Delta.ravel(), c=c)
                        plt.pause(0.1)
                        
                save_weights()
                
                delta_max.append(np.array(delta_number_step).max())
                
                f = open("delta_max.txt", "a")
                f.write("%d\n" % delta_max[-1])
                f.close()
                
                if np.abs(rew) > 0.02*facteur_rew  :
                    iterok = False
                else :
                    iterok = True
                
                if iterok == True :
                    it += 1
                else :
                    it = 0
                
                totalcounter += 1
                print ("Total iteration : %d. \tFalse iteration : %d\t reward = %0.4f" %(totalcounter, it, test_reward))
                
                if np.isnan(test_reward) == True :
                    sys.exit("Nan")

                print (along_reward, "\n")
                
                cri.model.optimizer.lr = cst_REIL["lr_critics_init"] 
#                curr_lr_actor = lr_actor_init

#                if totalcounter % 4 ==0 and totalcounter <= max_steps_lr*4 :
#                    critics.model.optimizer.lr = decays.create_decay_fn("linear",
#                                                            curr_step = totalcounter % max_steps_lr,    
#                                                            initial_value = lr_critics_init,
#                                                            final_value = lr_critics_final,
#                                                            max_step = max_steps_lr)
#                    curr_lr_actor = decays.create_decay_fn("linear",
#                                               curr_step = totalcounter % max_steps_lr,    
#                                               initial_value = lr_actor_init,
#                                               final_value = lr_actor_final,
#                                               max_step = max_steps_lr)
                
                print ("totalcounter = %d, \t lr_actor = %.6f\t lr_crits = %.6f" %(totalcounter, curr_lr_actor, cri.model.optimizer.lr))
                
                if totalcounter % 50 == 0 :
                    plt.figure("Reward / 50 steps")
                    plt.semilogy(totalcounter, test_reward, c='purple', linestyle='none', marker='o', ms = 4)
                    plt.pause(0.01)
                    
#                if totalcounter % step_new_batch == 0 :
#                    replay_memory.clear()
#                    u_init = np.zeros((1, X.size))
#                    pi_line = np.linspace(0.4, 1.4, 50)
#                    amplitude = [np.random.choice(pi_line) for i in range(1)]
#                    for i, amp in enumerate(amplitude) :
#                        u_init[i] = amp*np.sin(2*np.pi/L*X)
#                    
#                    print ("Cleaning of the batch %d ..." %((totalcounter // step_new_batch) -1))
#                    while len(replay_memory) < replay_memory_size : 
#                        for u in u_init :
#                            play_without_burger(u)
#                        time.sleep(1)
#                    print ("The batch %d is ready to be used" %( totalcounter // step_new_batch ))
                    
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
    
    print "ok"    
    lossess = dict()
    n_init = 1
    u_init = np.zeros((n_init, X.size))
    pi_line = np.linspace(0.4, 0.5, 50)
    amplitude = [np.random.choice(pi_line) for i in range(n_init)]
    
    for i, amp in enumerate(amplitude) :
        u_init[i] = amp*np.sin(2*np.pi/(cst_simu["L"]-2.*cst_simu["dx"])*(X-cst_simu["dx"])) 
    

#    plt.figure("Sinus initiaux")
#    for j, u in enumerate(u_init) :
#        plt.plot(X, u, label="amplitude = %.4f" % amplitude[j])
#    plt.legend()
#    

    print ("The batch is being constructed ...")
    replay_memory.clear()
    
    while len(replay_memory) < cst_REIL["replay_memory_size"] : 
        for u in u_init :
            play_with_burger(u)

    print ("The batch is ready to be used")
    time.sleep(1)
    
    print ("Training start")
    for ll, u in enumerate(u_init) :
        curr_loss = train(u)
        
        lossess[str(ll)] = curr_loss
    
    print ("Going to delete the curr figures, the results will be displayed in another figs...")
    time.sleep(5)

    plt.figure("Evolution de Loss sur un episodes vs iteration")
    plt.clf()
    
    plt.figure("Comparaison of the training loss with different replay memories")
    for k in range(27) :
        next(colors)
        
    plt.semilogy(range(episodes), curr_loss, c = next(colors), label="Loss replay")
    plt.xlabel("Number of episodes")
    plt.ylabel("Loss values")
    plt.legend()
    
    plt.pause(0.01)
        
    
