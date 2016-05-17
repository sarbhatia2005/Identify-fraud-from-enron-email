"""
Plotting functions to make it easier to create exploratory plots
of the enron dataset
"""
from __future__ import division, print_function

import sys
import matplotlib.pylab as plt
import numpy as np
import pprint


sys.path.append("../tools/")
from feature_format import featureFormat

def histogram(data, var_name, bins=None):
    """
    Makes a histogram of variable "var_name"
    :param data: data dict with enron data
    :param var_name: name of variable to plot
    :return: plot object
    """

    # setting some good default no of bins
    var = featureFormat(data, [var_name])
    if not bins:
        #http://stats.stackexchange.com/a/862/40853
        iqr = np.subtract(*np.percentile(var, [75, 25]))
        h = 2 * iqr * len(var)**(-1/3)
        bins = (np.amax(var)-np.amin(var))//h
    plt.title(var_name)
    plt.hist(var, bins=bins)

def boxplot_poi(data, var_name):
    """
    Makes box plot with variable "var_name"
    split into
    :param data: data dict with enron data
    :param var_name: name of variable to plot
    :return: plot object
    """
    poi_v = []
    no_poi_v = []
    for p in data.itervalues():
        value = p[var_name]
        if value == "NaN":
            value = 0
        if p["poi"] == 1:
            poi_v.append(value)
        else:
            no_poi_v.append(value)
    plt.xlabel("POI")
    plt.ylabel(var_name)
    plt.boxplot([poi_v, no_poi_v])
    plt.xticks([1, 2], ["POI", "Not a POI"])
    # http://stackoverflow.com/a/29780292/1952996
    for i, v in enumerate([poi_v, no_poi_v]):
        y = v
        x = np.random.normal(i+1, 0.04, size = len(y))
        plt.plot(x, y, "r.", alpha=0.2)

def plot_scatter(data, var_1, var_2):
    value1_poi = []
    value2_poi = []
    value1_nonpoi = []
    value2_nonpoi = []
    for value in data.itervalues():
        v1 = value[var_1]
        v2 = value[var_2]
        if value["poi"]:
            value1_poi.append(v1 if v1 != "NaN" else 0)
            value2_poi.append(v2 if v2 != "NaN" else 0)
        else:
            value1_nonpoi.append(v1 if v1 != "NaN" else 0)
            value2_nonpoi.append(v2 if v2 != "NaN" else 0)
    plt.scatter(value1_poi, value2_poi, color = "red")
    plt.scatter(value1_nonpoi, value2_nonpoi, color = "blue")
    plt.xlabel(var_1)
    plt.ylabel(var_2)


if __name__ == "__main__":
    import poi_id
    my_data = poi_id.load_dataset()
    my_data.pop("TOTAL")
    histogram(my_data, "salary")
    plt.show()
    plt.clf()
    plot_scatter(my_data, "bonus", "salary")
    plt.show()


