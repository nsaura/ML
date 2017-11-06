import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datasets_mglearn as dsets
dsets = reload(dsets) 

from sklearn.datasets import load_boston
from datasets_mglearn import load_extended_boston
boston = load_boston()

#In [97]: boston.data.shape
#Out[97]: (506, 13)

#This means boston has 13 features and 506 entries

X,y = dsets.load_extended_boston()
print("X.shape: {}".format(X.shape))
#X.shape: (506, 104)

dsets.plot_knn_classification(n_neighbors=3)
