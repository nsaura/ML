#!/usr/bin/python
# -*- coding: latin-1 -*-

import solvers
import numpy as np

def action_with_burger(state, r, f, fprime) :
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


#if __name__ == "__main__" :
#    r = 0.9187500000000001
#    
#    global f, fprime 
#    fprime = lambda u : u
#    f = lambda u : 0.5*u**2
    
#    
#    s = np.array([ -1.22464680e-16,   0.00000000e+00,   3.33027778e-02,
#         5.60122198e-02,   7.80679116e-02,   9.98955572e-02,
#         1.21579641e-01,   1.43134734e-01,   1.64554772e-01,
#         1.85825595e-01,   2.06928567e-01,   2.27841485e-01,
#         2.48538491e-01,   2.68989456e-01,   2.89158983e-01,
#         3.09004962e-01,   3.28476546e-01,   3.47511233e-01,
#         3.66030578e-01,   3.83933619e-01,   4.01086307e-01,
#         4.17303395e-01,   4.32314624e-01,   4.45693030e-01,
#         4.56666559e-01,  -4.56666559e-01,  -4.45693030e-01,
#        -4.32314624e-01,  -4.17303395e-01,  -4.01086307e-01,
#        -3.83933619e-01,  -3.66030578e-01,  -3.47511233e-01,
#        -3.28476546e-01,  -3.09004962e-01,  -2.89158983e-01,
#        -2.68989456e-01,  -2.48538491e-01,  -2.27841485e-01,
#        -2.06928567e-01,  -1.85825595e-01,  -1.64554772e-01,
#        -1.43134734e-01,  -1.21579641e-01,  -9.98955572e-02,
#        -7.80679116e-02,  -5.60122198e-02,  -3.33027778e-02,
#        -1.22464680e-16,   3.33027778e-02,   3.33027778e-02])
#    print (action_with_burger(s, r).shape)

