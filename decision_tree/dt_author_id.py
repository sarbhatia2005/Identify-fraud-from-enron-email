#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

from __future__ import print_function
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

def classify_DT(features_train, labels_train):
    cls = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    cls.fit(features_train, labels_train)
    print("Training time:", round(time()-t0, 3), "s")
    return cls

def accuracy_cls(cls, features_test, labels_test):
    t0 = time()
    predicted = cls.predict(features_test)
    print("prediction time:", round(time()-t0, 3), "s")
    return accuracy_score(labels_test, predicted)


if __name__ == "__main__":
    cls = classify_DT(features_train, labels_train)
    print(accuracy_cls(cls, features_test, labels_test))


