#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
from __future__ import print_function
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
import numpy

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42
)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print("Accuracy: {:.3f}".format(metrics.accuracy_score(y_true=labels_test, y_pred=pred)))
print("Number of POIs:", sum(pred))
print("Total People:", len(features_test))
print("True positives:", len([True for p, t in zip(pred, labels_test) if p==1 and p == t]))
print("Accuracy if predicted everyone 0.0:", (len(features_test)-sum(pred))/len(features_test))
print("Precision of Classifier:", metrics.precision_score(y_true=labels_test, y_pred=pred))
print("Recall of Classifier:", metrics.recall_score(y_true=labels_test, y_pred=pred))

