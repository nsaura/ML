import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print ("cancer.keys(): \n{}".format(cancer.keys()))


##
#print("DESCR: {}\n".format(cancer["DESCR"]))

#print("target: \n {}".format(cancer["target"]))
#Output consists in 0 and 1 representing benign or malignant (print("target_names: \n {}".format(cancer["target_names"]))

#print("feature_names: \n {}".format(cancer["feature_names"]))
#Outpout consists in 30 tumor features that are considered here

#print("shape of cancer[\"data\"] : {}".format(np.shape(cancer["data"])))
#shape of cancer["data"] : (569, 30)

#A particularity of cancer is that it has its keys as attributes so we can try :
#print("Shape of cancer data : {}".format(cancer.data.shape))
#Shape of cancer data : (569, 30)

#A beautiful way of counting benign and malignant tumor :
print("Sample counts per class:\n {}".format({n: v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}))

from sklearn.model_selection    import      train_test_split
from sklearn.neighbors          import      KNeighborsClassifier

### We want first to of estimate the evolution of train match score and test match score according to the number of neighbors in KNC model
X_train, X_test,y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

### Thus we'll compare those evolutions with which we had if we hadn't have specified the stratify option which is (Encyclopedia of Database Systems):

#Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole. For example in a binary classification problem where each class comprises 50% of the data, it is best to arrange the data such that in every fold, each class comprises around half the instances.
#Stratification is generally a better scheme, both in terms of bias and variance, when compared to regular cross-validation.
#https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation

#We add NS suffixe when we pull out Stratification


X_trainNS, X_testNS, y_trainNS, y_testNS = train_test_split(cancer.data, cancer.target, random_state=66)

train_accuracy  , train_accuracyNS   = [], []
test_accuracy   , test_accuracyNS    = [], []
neighbors = range(1,11)
for neigh in neighbors :
    clf = KNeighborsClassifier(n_neighbors=neigh)
    clf.fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
    clf.fit(X_trainNS, y_trainNS)
    train_accuracyNS.append(clf.score(X_trainNS, y_trainNS))
    test_accuracyNS.append(clf.score(X_testNS, y_testNS))
    
plt.figure("Train/Test accuracy")
plt.plot(neighbors, train_accuracy, label="Train accuracy")
plt.plot(neighbors, test_accuracy, 'r--', label="Test accuracy")
plt.xlabel("N_Neighbors")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

plt.figure("Train/TrainNS")
plt.plot(neighbors, train_accuracy, label="Train accuracy S")
plt.plot(neighbors, train_accuracyNS, "c--", label="Train accuracy NS")
plt.xlabel("N_Neighbors")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

plt.figure("Test/TestNS")
plt.plot(neighbors, test_accuracy, label="Test accuracy S")
plt.plot(neighbors, test_accuracyNS, "c--",label="Test accuracy NS")
plt.xlabel("N_Neighbors")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()






