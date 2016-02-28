#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
from __future__ import division, print_function
import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Person of interest in the dataset
pois_n = len([p for p in enron_data.itervalues() if p["poi"]])

# Total person of interest - some did not work for Enron

with open("../final_project/poi_names.txt") as f:
    pois_total = len(f.readlines()[2:]) # first two lines do not have a name

# wondering if it makes sense to have this function when it's
# so easy to get the data directly from the dict :)
def get_value_stock(person_name):
    return enron_data[person_name]["total_stock_value"]

if __name__ == "__main__":
    print("There are {} people in the dataset.".format(len(enron_data)))
    print("Each person has {} features".format(len(next(enron_data.itervalues()))))
    print("There are {} POIs in the dataset.".format(pois_n))
    print("There is a total of {} POIs.".format(pois_total))
    print("James Prentice had a total stock value of {}.".format(get_value_stock("PRENTICE JAMES")))
    print("Wesley Colwell sent {} messages to POIs".format(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]))
    print("Jeffrey Skilling had {} of stock options exercised.".format(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]))
    for name, values in enron_data.iteritems():
        if name.split()[0] in ["LAY", "SKILLING", "FASTOW"]:
            print("{} took home {} dollars.".format(name, values["total_payments"]))
    n_with_salary = len([True for p in enron_data.itervalues() if p["salary"] != "NaN"])
    n_with_email = len([True for p in enron_data.itervalues() if p["email_address"] != "NaN"])
    print("{} folks have a quantified salary.".format(n_with_salary))
    print("{} folks have a known email address.".format(n_with_email))

    # We've written some helper functions (featureFormat() and targetFeatureSplit()
    # in tools/feature_format.py) that can take a list of feature names and the data
    # dictionary, and return a numpy array.

    #How many people in the E+F dataset (as it currently exists) have "NaN" for their total payments?
    #  What percentage of people in the dataset as a whole is this?
    nan_total_payments = len([True for p in enron_data.itervalues() if p["total_payments"] == "NaN"])
    print("{} people have 'NaN' in their total payments. "
          "This corresponds to {:.2f}% of the dataset".format(nan_total_payments, nan_total_payments/len(enron_data)*100))
    pois = {p:v for p,v in enron_data.iteritems() if v["poi"]}
    nan_pois_total_payments = len([True for p in pois.itervalues() if p["total_payments"] == "NaN"])
    print("{} POIs have 'NaN' in their total payments. "
          "This corresponds to {:.2f}% of the POIs".format(nan_pois_total_payments, nan_pois_total_payments/len(pois)*100))