#!/usr/bin/python
from __future__ import print_function
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


def classify(clf, features_train, labels_train, **kwargs):
    clf = clf(**kwargs)
    clf.fit(features_train, labels_train)
    return clf

def classifyAdaboost(features_train, labels_train, n_estimators=100):
    return classify(AdaBoostClassifier, features_train, labels_train, n_estimators=n_estimators)

def classifyKNN(features_train, labels_train, n_neighbors=8):
    return classify(KNeighborsClassifier, features_train, labels_train, n_neighbors=n_neighbors)

def classifyRF(features_train, labels_train, n_estimators=100):
    return classify(RandomForestClassifier, features_train, labels_train, n_estimators=n_estimators)

if __name__ == "__main__":
    clf_dict = {"knn": classifyKNN,
                "adaboost": classifyAdaboost,
                "randomforest": classifyRF}
    for name, clf in clf_dict.iteritems():
        print(name, ":")
        clf_fitted = clf(features_train, labels_train)
        pred = clf_fitted.predict(features_test)
        print("Accuracy:", accuracy_score(labels_test, pred))
        prettyPicture(clf_fitted, features_test, labels_test)