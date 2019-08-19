#!/usr/bin/python

import csv
import numpy as np
from numpy import loadtxt
from sklearn import model_selection

def preprocess(features_file = "../data/vector1"):
    # feaures and labels are put into numpy arrays, which play nice with sklearn functions
    data = loadtxt(features_file)
    X, y = data[:, 1:], data[:, 0]

    ### test_size is the percentage of events for the test set (remainder go into training)
    f_train, f_test, t_train, t_test = model_selection.train_test_split(X, y, test_size=0.15, random_state=42)

    ### info on the data
    print("train good examples:", sum(t_train))
    print("train bad examples:", len(t_train)-sum(t_train))
    
    print("test good examples:", sum(t_test))
    print("test bad examples:", len(t_test)-sum(t_test))

    return f_train, f_test, t_train, t_test