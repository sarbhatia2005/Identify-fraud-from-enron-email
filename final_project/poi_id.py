#!/usr/bin/python
from __future__ import division, print_function

import argparse
import pickle
import pprint
import sys

import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn import grid_search

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# Load the dictionary containing the dataset
def load_dataset():
    """

    :return:
    """
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict


def outlier_removal(data_dict, list_of_names):
    """

    :param data_dict:
    :param list_of_names:
    :return:
    """
    for i in list_of_names:
        data_dict.pop(i)
    return data_dict


#Extract features and labels from dataset for local testing
def extract_features(dataset, features_list):
    """

    :param dataset:
    :param features_list:
    :return:
    """
    data = featureFormat(dataset, features_list)
    labels, features = targetFeatureSplit(data)
    return labels, features


def add_new_features(data_dict, new_feature_funs):
    """

    :param data_dict:
    :param new_feature_funs: list of functions
    that when applied to a person subdictionary return
    a dictionary with the name and value of the new feature {name: value}
    :return: data_dict with udpated new features. data_dict is modified in place.
    """
    for person, values in data_dict.iteritems():
        for fun in new_feature_funs:  # apply all funs to the person
            data_dict[person].update(fun(data_dict[person]))  # add the new features to the person subdictionary
    return data_dict


def compute_fraction(numerator, denominator):
    """

    :param numerator:
    :param denominator:
    :return:
    """
    try:
        fraction = numerator / denominator
    except TypeError:
        fraction = "NaN"
    return fraction


def fraction_from_poi(person_values):
    """

    :param person_values:
    :return:
    """
    poi_messages = person_values["from_poi_to_this_person"]
    all_messages = person_values["to_messages"]
    return {"fraction_from_poi":
            compute_fraction(poi_messages, all_messages)}


def fraction_to_poi(person_values):
    """

    :param person_values:
    :return:
    """
    poi_messages = person_values["from_this_person_to_poi"]
    all_messages = person_values["from_messages"]
    return {"fraction_to_poi":
            compute_fraction(poi_messages, all_messages)}


def fraction_shared(person_values):
    """

    :param person_values:
    :return:
    """
    poi_messages = person_values["shared_receipt_with_poi"]
    all_messages = person_values["to_messages"]
    return {"fraction_shared":
                compute_fraction(poi_messages, all_messages)}


def ratio_payments_salary(person_values):
    """

    :param person_values:
    :return:
    """
    salary = person_values["salary"]
    total_payments = person_values["total_payments"]
    return {"ratio_payments_salary":
            compute_fraction(salary, total_payments)}


def ratio_stocks_salary(person_values):
    """

    :param person_values:
    :return:
    """
    salary = person_values["salary"]
    total_stock_value = person_values["total_stock_value"]
    return {"ratio_stocks_salary":
            compute_fraction(salary, total_stock_value)}


def NaiveBayesPipeline():
    """

    :return:
    """
    features_selector = SelectKBest(f_classif)
    clf = GaussianNB()
    return Pipeline([("select_best", features_selector),
                     ("naive_bayes", clf)])


def AdaBoostPipeline():
    """

    :return:
    """
    features_selector = SelectKBest(f_classif)
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=123)
    return Pipeline([("select_best", features_selector),
                     ("ada", clf)])


def search_params_classifier(clf, my_dataset, features_list, param_grid):
    """
    Searches for the best parameters for the classifier and returns
    a classifier with the best parameters set.

    :param clf:
    :param my_dataset:
    :param features_list:
    :param param_grid:
    :return:
    """
    labels, features = extract_features(my_dataset, features_list)
    cross_validation_iter = StratifiedShuffleSplit(y=labels, test_size=0.2, random_state=123, n_iter=100)
    search_params = grid_search.GridSearchCV(estimator=clf(),
                                             param_grid=param_grid,
                                             cv=cross_validation_iter,
                                             scoring="f1",
                                             n_jobs=-1)
    search_params.fit(features, labels)
    pprint.pprint(search_params.grid_scores_)
    pprint.pprint(search_params.best_params_)
    pprint.pprint(search_params.best_score_)
    return search_params, search_params.best_estimator_


def my_cross_validation(estim, features_list):
    """

    :param estim:
    :param features_list:
    :return:
    """
    labels, features = extract_features(my_dataset, features_list)
    cross_validation_iter = StratifiedShuffleSplit(y=labels, test_size=0.2, random_state=123, n_iter=100)
    precision = []
    recall = []
    f1 = []
    for train, test in cross_validation_iter:
        features_train = [features[i] for i in train]
        labels_train = [labels[i] for i in train]
        features_test = [features[i] for i in test]
        labels_test = [labels[i] for i in test]
        estim.fit(features_train, labels_train)
        pred = estim.predict(features_test)
        precision.append(metrics.precision_score(labels_test, pred))
        recall.append(metrics.recall_score(labels_test, pred))
        f1.append(metrics.f1_score(labels_test, pred))
    print(estim)
    print("Precision:", np.mean(precision))
    print("Recall:", np.mean(recall))
    print("F1 score:", np.mean(f1))


if __name__ == "__main__":
    # Script can be run with command line arguments to
    # run gridsearchcv. By default no gridsearch is done
    # and classifiers with hardcoded parameters are stored.
    parser = argparse.ArgumentParser(description="Train classifier")
    # http://stackoverflow.com/a/15008806/1952996
    parser.add_argument("--gridsearch", dest="gridsearch", action="store_true",
                        help="Perform gridsearch for the "
                             "classifier parameters? (can take some time)")
    parser.add_argument("--no-gridsearch", dest="gridsearch", action="store_false")
    parser.set_defaults(gridsearch=False)
    args = parser.parse_args()
    GRID_SEARCH = args.gridsearch

    # Task 1: Select what features you'll use.
    features_list = ['poi',
                     'to_messages',
                     'bonus',
                     'total_stock_value',
                     'expenses',
                     'from_poi_to_this_person',
                     'restricted_stock',
                     'salary',
                     'total_payments',
                     'fraction_shared',
                     'fraction_to_poi',
                     'exercised_stock_options',
                     'from_messages',
                     'other']

    # Loading dataset
    my_dataset = load_dataset()

    # Task 2: Remove outliers
    outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK"]
    outlier_removal(my_dataset, outliers)

    # Task 3: Create new feature(s)
    # Store to my_dataset for easy export below.
    new_features_funs_list = [fraction_from_poi, fraction_to_poi, fraction_shared,
                              ratio_payments_salary, ratio_stocks_salary]
    add_new_features(my_dataset, new_features_funs_list)


    # Task 4: Try a varity of classifiers
    # Please name your classifier clf for easy export below.

    if GRID_SEARCH:
        # Task 5: Tune your classifier to achieve better than .3 precision and recall
        param_grid = {"select_best__k": [i for i in range(1, len(features_list) - 1)]}

        nb_search, nb = search_params_classifier(clf=NaiveBayesPipeline, my_dataset=my_dataset,
                                                 features_list=features_list, param_grid=param_grid)
        # The main parameters to tune to obtain good results are n_estimators and the complexity of the base estimators
        # (e.g., its depth max_depth or minimum required number of samples at
        # a leaf min_samples_leaf in case of decision trees).
        ada_param_grid = {"select_best__k": [nb_search.best_params_["select_best__k"]],
                          "ada__n_estimators": [10, 20, 50, 100, 200, 300],
                          "ada__base_estimator__max_depth": [1, 2, 3, 4, 5],
                          "ada__base_estimator__min_samples_leaf": [1, 2, 4, 10, 20]}

        ada_search, ada = search_params_classifier(clf=AdaBoostPipeline, my_dataset=my_dataset,
                                                   features_list=features_list,
                                                   param_grid=ada_param_grid)
    else:

        nb = NaiveBayesPipeline().set_params(select_best__k=6)
        ada = AdaBoostPipeline().set_params(select_best__k=6,
                                            ada__n_estimators=200,
                                            ada__base_estimator__max_depth=2,
                                            ada__base_estimator__min_samples_leaf=1)
        for estim in [nb, ada]:
            my_cross_validation(estim, features_list)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    # Naive Bayes takes the lead and will be dumped!
    clf = nb
    dump_classifier_and_data(clf, my_dataset, features_list)
