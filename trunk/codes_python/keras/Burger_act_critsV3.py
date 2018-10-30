#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys 
import time 
import noise 
import decays
import numpy as np
import os.path as osp
from os import mkdir, remove
import Class_deque as Class_deque
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
import actions_for_KerasAC_DDPG as ACactions

graph = tf.get_default_graph()

#plt.close("all")

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 

pff = reload(pff)
KAD = reload(KAD)
noise = reload(noise)
decays = reload(decays)
solvers = reload(solvers)
ACactions = reload(ACactions)
Class_deque = reload(Class_deque)

colors = iter(cm.plasma_r(np.arange(600)))
#----------------------------------------------
def samples_memories(BATCH_SIZE):
    indices = np.random.permutation(deque_obj.size())[: BATCH_SIZE]
    # cols[0] : state
    # cols[1] : action
    # cols[2] : reward
    # cols[3] : next_state
    # cols[4] : done
    cols = [[],[],[],[],[]]
    
    for idx in indices :
        memory = deque_obj.replay_memory[idx]
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
cst_simu["Nx"] = 50

# Constantes temporelles 
cst_simu["tf"] = 1.5
cst_simu["Nt"] = 40

cst_simu["dx"] = float(cst_simu["L"])/(cst_simu["Nx"]-1)
cst_simu["dt"] = float(cst_simu["tf"])/(cst_simu["Nt"])
cst_simu["r"] = cst_simu["dt"] / cst_simu["dx"]

fprime = lambda u : u
f = lambda u : 0.5*u**2

X  = np.arange(0, cst_simu["L"], cst_simu["dx"])
T  = np.arange(0, cst_simu["tf"], cst_simu["dt"])

cst_simu["max_steps"] = 150

#
# Constantes for the REIL
#
cst_REIL["TAU"] = 0.01
cst_REIL["gamma"] = 0.99
cst_REIL["episodes"] = 50000
cst_REIL["steps_before_change"] = 20

# Deque
cst_REIL["BATCH_SIZE"] = int(512) 
cst_REIL["replay_memory_size"] = 5e4

deque_obj = Class_deque.deque_obj(cst_REIL["replay_memory_size"])

# Number of HN for both actor and critics NN
cst_REIL["HIDDEN1_UNITS"] = 50
cst_REIL["HIDDEN2_UNITS"] = 50


# Lr decays for actor or critics
cst_REIL["max_steps_lr"] = 50

cst_REIL["lr_actor_init"] = 1e-3
cst_REIL["lr_actor_final"] = 1e-4

cst_REIL["lr_critics_final"] = 1e-3
cst_REIL["lr_critics_init"] = 1e-4


# Eps (for noise) decays
cst_REIL["EpsForNoise_init"] = 1.3
cst_REIL["EpsForNoise_fina"] = 0.4

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

decay_cri_lr = True
decay_act_lr = True

#sys.exit()
#----------------------------------------------
def save_weights() :
    act.target_model.save_weights("./ACweights/actmodel.h5", overwrite=True)
    cri.target_model.save_weights("./ACweights/crimodel.h5", overwrite=True)
    print("Saved")
#----------------------------------------------
def load_weights() :
    act.model.load_weights("./ACweights/actmodel.h5")
    cri.model.load_weights("./ACweights/crimodel.h5")
    
    act.target_model.load_weights("./ACweights/actmodel.h5")
    cri.target_model.load_weights("./ACweights/crimodel.h5")
    print("Loaded")
#----------------------------------------------

if osp.exists("./ACweights/") == False :
    mkdir("./ACweights/")

if osp.isfile("./ACweights/actmodel.h5") and osp.isfile("./ACweights/crimodel.h5") :
    print("act and cri weights removed")
    remove("./ACweights/actmodel.h5")
    remove("./ACweights/crimodel.h5")
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
    
    return (np.linalg.norm((temp_term + square_term), 2))**2 * 50

#----------------------------------------------
def play_with_ACpred(u_init, noisy=True):    
    """
    Function that plays a certain number of iterations of the game (until it finishes).
    This function can also be used to construct the rpm.
    The loop on the episodes is outside of the function. 
    
    Arguments :
    -----------
    u_init : To set the initialisation
    """
    episode_memory = []
    
    epsilon = 0
    total_rew = 0
    a_t = np.zeros([1, X.size])
    
    # Pour exploration de l'espace des actions
#    noise = noiselevel
#    noiselevel = noise * 0.999
#    noise_t = np.zeros([1, action_size])
    
    # Mettre ici un tirage aléatoire d'un entier q entre 1 et max step.
    # Résoudre avec Roe jusqua la q eme itération, écrire cela en tant que s_t
    # Puis faire toute la procédure utilisé jusqu'ici.
    # Répéter ce processus jusqua ce que le replay buffer soit plein    

    global graph
    while deque_obj.size() < cst_REIL["replay_memory_size"] :
    
        int_line = range(0, cst_simu["max_steps"])
        curr_step = np.random.choice(int_line)

        temp_state = u_init

        for j in range(0, curr_step+1) :
            temp_state1 = ACactions.action_with_burger(temp_state, cst_simu["r"], f, fprime)

            if j != curr_step :
                temp_state = temp_state1
        
        s_t = temp_state
    
        with graph.as_default() :
            a_t_original = act.model.predict(np.array([s_t]))
            
            OU_noise = np.zeros_like(a_t_original)
        
            if noisy == True : 
                epsilon = decays.create_decay_fn("linear",
                                                 curr_step=j,
                                                 initial_value=cst_REIL['EpsForNoise_init'],
                                                 final_value=cst_REIL['EpsForNoise_fina'],
                                                 max_step=cst_simu["max_steps"]
                                                 )

                args = {"rp_type" : "ornstein-uhlenbeck",
                        "n_action" : 1,
                        "rp_theta" : 0.1,
                        "rp_mu" : 0.,
                        "rp_sigma" : 0.2,
                        "rp_sigma_min" : 0.05}
    
                coeffOU_noise = noise.create_random_process(args).sample()
                OU_noise = coeffOU_noise*(np.array([np.random.rand() for rand in range(X.size)]) - 0.5)                
                
                
            a_t = a_t_original + OU_noise
            a_t = a_t.ravel()
            
            s_t1 = ACactions.action_with_delta_Un(s_t, a_t)
            
            r_t = reward(s_t1, s_t)

#        print ("state :\n{}".format(s_t))
#        print ("action :\n{}".format(a_t))
#        print ("next state :\n{}".format(s_t1))
#        
#        time.sleep(5)
        
            if r_t > 1 and r_t < 1000 :
                done = True # Game over
                rew = r_t
            
            elif r_t > 1000 :
                done = True # Game over
                rew = -r_t  # Grosse pénalité 
            
            else :
                done = False # On continue si c'est bon 
                rew = r_t
                    
            print ("reward :\n{}".format(rew))
            deque_obj.append((s_t, a_t, rew, s_t1, done))

#            plt.figure("comparaison bruit")            
#            plt.plot(X, s_t, label="Avec bruit", c='red')
#            plt.plot(X, [s_t[i] + OU_noise[i] for i in range(X.size)], label="Avec bruit", c='b')
#            plt.legend()
#            plt.pause(5)
#            
#            sys.exit("dd")  
#            
            
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
    
    # Mettre ici un tirage aléatoire d'un entier q entre 1 et max step.
    # Résoudre avec Roe jusqua la q eme itération, écrire cela en tant que s_t
    # Puis faire toute la procédure utilisé jusqu'ici.
    # Répéter ce processus jusqua ce que le replay buffer soit plein
    
    total_rew = 0
    a_t = np.zeros([1, s_t.size])
    plotplot = 0
    
    file = open("play_reward.txt", "w")
    file.close()
    
    for j in range(cst_simu["max_steps"]) :
        s_t1 = ACactions.action_with_burger(s_t, cst_simu["r"], f, fprime)
        a_t_original = np.array([s_t1[i] - s_t[i] for i in range(len(s_t))])
        
#        plt.figure("Burger Plot")
#        plt.clf()
#        plt.plot(X, s_t, label="st")
#        plt.plot(X, s_t1, label="s_t1")
#        plt.legend()
#        plt.pause(0.01)
        
        a_t = a_t_original.ravel()
        
        gg = [abs(a) > 1. for a in a_t]
        
        r_t = reward(s_t1, s_t)
        file = open("play_reward.txt", "a")
        file.write("%.4f\n" %(r_t))
        file.close()
                
#        print ("state :\n{}".format(s_t))
#        print ("action :\n{}".format(a_t))
#        print ("reward :\n{}".format(r_t))
#        print ("next state :\n{}".format(s_t1))
        
        if r_t > 1 and r_t < 10000 :
            done = True # Game over
            rew = r_t
        
        elif r_t > 10000 :
            done = True # Game over
            rew = -r_t  # Grosse pénalité 
        
        else :
            done = False # On continue si c'est bon 
            rew = r_t
                
        if deque_obj.size() < cst_REIL["replay_memory_size"] :
            deque_obj.append((s_t, a_t, rew, s_t1, done))
        
        else :
            if abs(np.random.randn()) > 0.5 :
                deque_obj.popleft() # Pop the leftmost element 
            else: 
                deque_obj.pop() # Pop the rightmost element 
        
        s_t = s_t1
#----------------------------------------------        
def train (u_init, play_type="AC") :
    """
    Function to train the actor and the critic target networks
    It has to be after the construction of the replay_memory
    """
    global graph, colors

    loss, losses, trainnum = 0, [], 0
    save_weights()
    
    file = open("rewards.txt", "w") ; file.close()
    file = open("delta_max.txt", "w") ; file.close()
    
    s_a_file = open("state_action.txt", "w") ; s_a_file.close()
    
    for ep in range(cst_REIL["episodes"]) :
        if ep % 200 == 0 :
            if decay_cri_lr == True : 
                cri.model.optimizer.lr =\
                    decays.create_decay_fn("linear",
                                           curr_step = ep % int(cst_REIL["episodes"] / 500),    
                                           initial_value = cst_REIL["lr_critics_init"],
                                           final_value = cst_REIL["lr_critics_final"] ,
                                           max_step = int(cst_REIL["episodes"] / 500))
            else :
                cri.model.optimizer.lr = cst_REIL["lr_critics_init"]
                
            if decay_act_lr == True : 
                curr_lr_actor = decays.create_decay_fn("linear",
                                               curr_step = ep % int(cst_REIL["episodes"] / 500),
                                               initial_value = cst_REIL["lr_actor_init"],
                                               final_value = cst_REIL["lr_actor_final"] ,
                                               max_step = int(cst_REIL["episodes"] / 500))
            else : 
                curr_lr_actor = cst_REIL["lr_actor_init"]
            
            if ep ==0 :
                deque_obj.clear()
                while deque_obj.size() < cst_REIL["BATCH_SIZE"] : 
                    if play_type == "AC" :
                        play_with_ACpred(u_init)
            
        print ("episodes = %d\t lr_actor_curr = %0.8f \tlr_crits_curr = %0.8f"\
                    %(ep, curr_lr_actor, cri.model.optimizer.lr))
        time.sleep(3)
        
        loss = 0
        trainnum = 0
        
#        curr_lr_actor = cst_REIL["lr_actor_init"]
#        cri.model.optimizer.lr = cst_REIL["lr_critics_init"]

        rew = 0
        delta_max, along_reward = [], []

        it, totalcounter, iterok  = 0, 0, False

        while it < cst_simu["max_steps"] :
            states,actions,rewards,next_states,dones = (samples_memories(cst_REIL["BATCH_SIZE"]))
            delta_number_step = []

            y_t = np.asarray([0.0]*cst_REIL["BATCH_SIZE"])
            rewards = np.concatenate(rewards)
            rwrds = np.copy(rewards)
#            rewards = np.array([10*rr for rr in rwrds])
            
#            print ("states shape : ", states.shape)
#            print ("actions shape : ", actions.shape)
#            print ("rewards shape: ", rewards.shape)
#            print ("dones shape  : ", dones.shape)
            
            with graph.as_default() :
                # Q function evaluation on the target graphs 
                target_q_values = cri.target_model.predict(
                                [next_states, act.target_model.predict(next_states)])
            
            target_q_values = target_q_values.reshape([1, target_q_values.shape[0]])[0]
            
            for k in range(cst_REIL["BATCH_SIZE"]) :
                
                if dones[k] :
                    y_t[k] = rewards[k] 
                else : 
                    y_t[k] = rewards[k] + cst_REIL["gamma"]*target_q_values[k]
                    
            with graph.as_default():
                # We set lr of the critic network
                
                logs = cri.model.train_on_batch([states, actions], y_t)
                
                a_for_grad = act.model.predict(states)
                grad = cri.gradients(states, a_for_grad)
                
                act.train(states, grad, learning_rate=curr_lr_actor)
                
                act.target_train()
                cri.target_train()         
                
                s_a_file = open("state_action.txt", "w")
                s_a_file.write("Episodes : %d\t Iteration : %d\n" %(ep, it))
                for s, a, r  in zip(states, actions, rewards) :
                    s_a_file.write("States \n{} \n".format(s))
                    s_a_file.write("Actions \n{} \n".format(a))
                    s_a_file.write("rewards \n{} \n".format(r))
                
                s_a_file.close()
                
                # In this section we decide wheither we continue or not
                # We use those actor and critic target networks for the next steps_before_change steps
                        
                save_weights()
                print ("totalcounter = %d, \t lr_actor = %.6f\t lr_crits = %.6f"\
                                        %(it, curr_lr_actor, cri.model.optimizer.lr))
                
                
                print (logs / cst_simu["max_steps"])
                load_weights()
                
                new_actions = act.target_model.predict(states)
                new_next = []                
                new_reward = []
#                
#                for s,a in zip(states, actions) :
#                    ns = ACactions.action_with_delta_Un(s,a)
#                    nr = reward(s, ns)
#                    new_next.append(ns)
#                    new_reward.append(nr)

#                    if nr > 1 and nr < 1000 :
#                        done = True 
#                        rew = nr
#                
#                    elif nr > 1000 :
#                        done = True # Game over
#                        rew = -nr   
#                
#                    else :
#                        done = False # On continue si c'est bon 
#                        rew = nr
#                    
#                    deque_obj.append((s, a, nr, ns, done))
#                    
##                    print deque_obj.size()
                
            loss += abs(logs) / cst_simu["max_steps"]
            it += 1
            trainnum += 1
        print ("Episode = %d :" % ep)
        print ("total loss = %.4f" %loss)
        
        losses.append(loss)
        
        plt.figure("Evolution de Loss sur un episodes vs iteration STEP = 100")
        plt.semilogy(ep, loss, marker='o', ms=6, linestyle="none", c='navy')
        plt.pause(0.5)
    
    return losses
#----------------------------------------------
    
if __name__ == "__main__" :
    
    print ("ok"    )
    lossess = dict()
    n_init = 1
    u_init = np.zeros((n_init, X.size))
    pi_line = np.linspace(0.4, 0.5, 50)
    amplitude = [np.random.choice(pi_line) for i in range(n_init)]
    
    for i, amp in enumerate(amplitude) :
        u_init[i] = amp*np.sin(2*np.pi/(cst_simu["L"]-2.*cst_simu["dx"])*(X-cst_simu["dx"])) 
    
    
    play_type = "AC"

    print ("The batch is being constructed ...")
    deque_obj.clear()
    
#    while len(deque_obj.replay_memory) < cst_REIL["BATCH_SIZE"] : 
#        for u in u_init :
#            if play_type== "AC" :
#                play_with_ACpred(u, False)
#            else :
#                play_with_burger(u)
    
    print ("The batch is ready to be used")
    time.sleep(1)
    
    print ("Training start")
    for ll, u in enumerate(u_init) :
        curr_loss = train(u, play_type)
#        
#        lossess[str(ll)] = curr_loss
#        
#    uu = np.copy(u) 
#    uu_prev = np.copy(u)
#    for it in range(cst_simu["max_steps"]) : 
#        u_next = ACactions.action_with_burger(uu, cst_simu["r"], f, fprime)
#        u_next_prev = act.target_model.predict(uu_prev.reshape(1,-1)).ravel() 
#        
#        plt.figure("Comparaison")
#        plt.clf()
#        plt.plot(X, u_next, label="Burger Roe, it = %d" % (it+1), c='r')
#        plt.plot(X, u_next_prev, 'o', label="Burger Act, it = %d" % (it+1), fillstyle='none', c='b')
#        plt.legend()
#        plt.pause(0.2)
#        u_prev = u_next_prev
#        uu = u_next
