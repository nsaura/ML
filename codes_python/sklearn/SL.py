import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

keys = ["age", "ncars", "ohouse", "nchil", "mstatus","odog","oboat"]

dataset = dict()

for k in keys :
    dataset[k]=[]
    

dataset["age"] = [66, 52, 22, 25, 44, 39, 26, 40, 53, 64, 58, 33]        
dataset["ncars"] = [1, 2, 0, 1, 0, 1, 1, 3, 2, 2, 2, 1]
dataset["nchil"] = [2, 3, 0, 1, 2, 2, 2, 1, 2, 3, 2, 1]

dataset["ohouse"] = ["yes", "yes", "no" ,"no", "no", "yes", "no", "yes", "yes", "yes", "yes","no" ]
dataset["odog"] = ["no", "no", "yes", "no", "yes", "yes", "no", "yes", "no", "no", "yes", "no"]
dataset["oboat"] = ["yes", "yes", "no", "no", "no", "no", "no", "no", "yes", "no", "yes", "no"] 

dataset["mstatus"] = ["w", "m", "m", "s", "d", "m", "s", "m", "d", "d", "m", "s"]

nf = "custo.dat"
f = open(nf,"w")
f.write("\t".join([str(k) for k in keys]) + "\n")

for i in range(np.size(dataset["age"])):
    ilst=[dataset[k][i] for k in keys]
    f.write("\t".join([str(c) for c in ilst]) + "\n")
    
f.close()
