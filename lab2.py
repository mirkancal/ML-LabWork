#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:02:05 2018

@author: mirkan
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# read from data
df_train = pd.read_csv("team_1.csv", encoding="latin-1")
df_test = pd.read_csv("team_2.csv", encoding="latin-1")


# Get Age and Values from both dataframes
x_train = df_train.Age.values.reshape(-1, 1)
y_train = df_train.Experience.values.reshape(-1, 1)

x_test = df_test.Age.values.reshape(-1, 1)
y_test = df_test.Experience.values.reshape(-1, 1)

# Fit a line and find predict values
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

y_head_train = linear_reg.predict(x_train)

linear_reg_ = LinearRegression()
linear_reg_.fit(x_test, y_test)

y_head_test = linear_reg_.predict(x_test)


# Scatter test data, use train data line
# team_1 is train, team_2 is test

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, y_head_train, color="red")
plt.xlabel("Age")
plt.ylabel("Experience")
plt.show()

# team_2 is train, team_1 is test

plt.scatter(x_train, y_train, color="blue")
plt.plot(x_test, y_head_test, color="red")
plt.xlabel("Age")
plt.ylabel("Experience")
plt.show()


# print MSE
print("RSS of train data")
print(mean_squared_error(y_train, y_head_train))
print("RSS of test data")
print(mean_squared_error(y_test, y_head_test))
