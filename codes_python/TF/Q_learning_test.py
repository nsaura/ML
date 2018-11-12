#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np

import matplotlib.pyplot as plt

import os
import os.path as osp

import tensorflow as tf
import time

#Cas page 13 du chapitre 16 de Hands on etc
nan = np.nan

# Transition Matrix
T = np.array([
[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
[[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
[[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
             ])
             
# Reward Matrix
R = np.array([
[[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
[[0.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
[[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]]
             ])
             
possible_actions = [[0, 1, 2], [0, 2], [1]]

discount_rate = 0.95
learning_rate0 = 0.05
learning_rate_decay = 0.1

n_iterations = 20000

s = 0

Q_learning = np.full((3,3), -np.inf)


for state, action in enumerate(possible_actions):
    Q_learning[state, action] = 0.0

for iteration in range(n_iterations) :
    if iteration % 100 == 0:
        print Q_learning
    # Choix d'action aléatoire parmi les possibilité reliées à l'état en cours 
    a = np.random.choice(possible_actions[s])
    
    sp = np.random.choice(range(3), p=T[s, a]) #Next state given the action
    
    reward = R[s, a, sp]
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    
    Q_learning[s, a] = (1 - learning_rate) * Q_learning[s, a] + learning_rate * (
                                        reward + discount_rate*np.max(Q_learning[sp])
                       )
    
    s = sp
    
#In [16]: Q_learning
#Out[16]: 
#array([[  4.1955878 ,   1.13634028,   0.88683189],
#       [  0.        ,         -inf, -15.4624186 ],
#       [        -inf,  13.91306595,         -inf]])

# Choisir l'action donné l'état

