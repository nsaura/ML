#!/usr/bin/python
# -*- coding: latin-1 -*-

import noise 
import numpy as np
from collections import deque

class deque_obj():
    def __init__(self, size):
        self.replay_memory = deque( [], maxlen=size )
    
    def append(self, arg_tuple):
        if len(self.replay_memory) > self.size() :
            self.popleft()
    
        self.replay_memory.append(arg_tuple)
    
    def clear(self):
        self.replay_memory.clear()
    
    def size (self):
        return int(len(self.replay_memory))
    
    def popleft(self):  
        self.replay_memory.popleft()
    
    def pop(self) :        
        self.replay_memory.pop()
