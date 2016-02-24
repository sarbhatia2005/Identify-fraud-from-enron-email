#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def train_NB_cls(features_train, labels_train):
    cls = GaussianNB()
    t0 = time()
    cls.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    return cls

def accuracy_cls(cls, features_test, labels_test):
    t0 = time()
    predicted = cls.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"
    return accuracy_score(labels_test, predicted)

if __name__ == "__main__":
    cls = train_NB_cls(features_train, labels_train)
    print accuracy_cls(cls, features_test, labels_test)