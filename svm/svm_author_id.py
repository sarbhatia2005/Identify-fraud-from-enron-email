#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from __future__ import print_function
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

def train_svm(features_train, labels_train, C=1.0, kernel="linear", gamma=0.0):
    clf = SVC(C=C, kernel=kernel, gamma=gamma)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3), "s")
    return clf


if __name__ == "__main__":
    clf = train_svm(features_train, labels_train, C=10000, kernel="rbf")
    t0 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t0, 3), "s")
    print(accuracy_score(labels_test, pred))
    for x in [10, 26, 50]:
        person = "Sara" if pred[x] == 0 else "Chris"
        print(person, "wrote the", x, "th Email.")
    no_of_chris = sum(pred)
    print("Chris wrote", no_of_chris, "Emails.")
    print("Sara wrote", len(pred)-no_of_chris, "Emails.")



