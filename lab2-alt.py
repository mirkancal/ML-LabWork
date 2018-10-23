#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:12:00 2018

@author: mirkan
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def main():
    # observations
    x_test = np.array([])
    y_test = np.array([])
    
    x_train = np.array([])
    y_train = np.array([])

    with open("team_1.csv", encoding="latin-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for column in reader:
            x_test = np.append(x_test, column["Age"]).astype(float)
            y_test = np.append(y_test, column["Experience"]).astype(float)
            
    with open("team_2.csv", encoding="latin-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for column in reader:
            x_train = np.append(x_train, column["Age"]).astype(float)
            y_train = np.append(y_train, column["Experience"]).astype(float)

    # estimating coefficients
    b_train = estimate_coef(x_train, y_train)

    b_test = estimate_coef(x_test, y_test)

    # plotting regression line
    
    # plotting the actual points as scatter plot
    plt.scatter(x_test, y_test, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred_train = b_train[0] + b_train[1] * x_train
    y_pred_test = b_test[0] + b_test[1] * x_test
    

    # plotting the regression line
    plt.plot(x_train, y_pred_train, color="g")

    # putting labels
    plt.xlabel('Age')
    plt.ylabel('Experience')

    # function to show plot
    plt.show()
    
    # plotting the actual points as scatter plot
    plt.scatter(x_train, y_train, color="m",
                marker="o", s=30)
    
    plt.plot(x_test, y_pred_test, color="g")
    
    # putting labels
    plt.xlabel('Age')
    plt.ylabel('Experience')
    
    plt.show()




main()