#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:02:05 2018

@author: mirkan
"""

# In this file, I've showed train data with train regression line

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# read from data
df_train = pd.read_csv("team_1.csv", encoding="latin-1")
df_test = pd.read_csv("team_2.csv", encoding="latin-1")


# Get Age and Values from both dataframes
x_train = df_train.Age.values.reshape(-1, 1)
y_train = df_train.Experience.values.reshape(-1, 1)

x_test = df_test.Age.values.reshape(-1, 1)
y_test = df_test.Experience.values.reshape(-1, 1)


linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

y_head = linear_reg.predict(x_train)

plt.scatter(x_train, y_train, color="blue")
plt.plot(x_train, y_head, color="red")
plt.xlabel("Age")
plt.ylabel("Experience")
plt.show()
