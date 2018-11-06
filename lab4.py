#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:09:14 2018

@author: mirkan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut 
from sklearn import metrics

DataFrame = pd.read_csv('teams_comb.csv', encoding="latin-1")
DataFrame = DataFrame.dropna()
Xr = DataFrame[["Age", "Experience", "Power"]]
yr = DataFrame["Salary"]
X_array = np.array(Xr)
y_array = np.array(yr)

#scores = cross_val_score(LinearRegression(), Xr, yr, cv=len(Xr), scoring = "r2")
#print("Cross-validated scores:", scores)
#print("Average: ", scores.mean())
#print("Variance: ", np.std(scores))

loo = LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(Xr):
    X_train, X_test = X_array[train_idx], X_array[test_idx] #requires arrays
    y_train, y_test = y_array[train_idx], y_array[test_idx]
    
    model = LinearRegression()
    model.fit(X = X_train, y = y_train) 
    y_pred = model.predict(X_test)
        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
        
    ytests += list(y_test)
    ypreds += list(y_pred)
    print("MSE: {:.5f}".format(metrics.mean_squared_error(ytests, ypreds)))
        
rr = metrics.r2_score(ytests, ypreds)
ms_error = metrics.mean_squared_error(ytests, ypreds)
        
print("Leave One Out Cross Validation")
print("MSE: {:.5f}".format(ms_error))

https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6