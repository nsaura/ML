#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os
import os.path as osp

import numdifftools as nd

import tflearn
import tensorflow as tf

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

import time

#sudo pip install gym
#xvfb-run -s -screen 0 1400x900x24 python pour lancer xfvb

import gym

plt.ion()

env = gym.make("CartPole-v0")
obs = env.reset()

env.render()

img = env.render(mode="rgb_array")
img.shape

#Ask the environment what actions are possible
#env.action_space

# ---> Discrete(2) i.e. 2 choix possibles 0 left, 1 right

#action = 1
#obs, reward, done, info = env.step(action)

#obs
#array([-0.04114253,  0.17951051,  0.00148627, -0.2757451 ])
# obs[0] : Horizontal Position
# obs[1] : Velocity
# obs[2] : Angle Of The Pole
# obs[3] : Angular Velocity


def basic_policy(obs) :
    angle = obs[2]
    return 0 if angle<0 else 1
    
totals = []
for episode in range(500) :
    episode_reward = 0
    obs = env.reset()
    
    for step in range(1000) :
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
#        env.render()
        
        episode_reward += reward
        
        if done :
            break
    totals.append(episode_reward)

env.close()

print ("The policy managed to keep the pole upright for %d steps (max)" %(np.max(totals)))
print ("The policy managed to keep the pole upright for %d steps (min)" %(np.min(totals)))

