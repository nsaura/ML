#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import time
import numpy as np
from termcolor import colored
 
def glide1D (arr, stride, show=False) :
    arr_shape = np.shape(arr)
    i = 0
    
    valid = arr_shape[0] % stride == 1
    leakage = arr_shape[0] % stride + 1
    
    if  valid :
        print "Should be valid"
        res = True
    
    else :
        print "Use \"same\" option.\nLeakage of %d last termes with strides = %d\n" % (leakage, stride)
        res = False
        
    if show == True :
        
        while i < arr_shape[0] :
            
            print ' '.join(colored(arr[j], 'cyan') if j != i
                   else colored(arr[j], 'red')
                   for j in range(arr_shape[0]))
            time.sleep(0.3)
            
            i += stride
    
    return res

def find_divisor(number) :
    lst = []
    for i in range(1, number) :
        if number % i == 0 :
            lst.append(i)
    print lst 
    
    
