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

T = np.array([
[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
[[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
[[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
             ])

R = np.array([
[[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
[[0.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
[[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]]
             ])

possible_actions = [[0, 1, 2], [0, 2], [1]]

Q = np.full((3,3), -np.inf) #Fill a matrix, whose shape is first arg, with the second arg value

# Q-Value Iteration Algorithm
# First step : Initialization
for state, actions in enumerate(possible_actions) :
    Q[state, actions] = 0.0 
    
discount_rate = 0.95
n_iterations = 100

for iteration in range(n_iterations) :  
    Q_prev = Q.copy()
    
    for s in range(3) : # Trois étapes considérées

        for a in possible_actions[s] : # Toutes les actions considérées selon le state
            # Sum les reward sur toutes les étapes suivantes possibles
            # partie de l'action en cours, vers tous les prochains états possibles
            # On multiplie les meilleurs actions par le discount rate

            Q[s, a] = np.sum([T[s,a,sp]* (R[s,a,sp] + discount_rate*np.max(Q_prev[sp])) 
                              for sp in range(3)])  




