#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:48:17 2018

@author: mirkan
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# read from data
df = pd.read_csv("team.csv", encoding="latin-1")

x = df.Age.values.reshape(-1, 1)
y = df.Experience.values.reshape(-1, 1)

linear_reg = LinearRegression()
linear_reg.fit(x, y)

y_head = linear_reg.predict(x)

plt.scatter(x, y, color="blue")
plt.plot(x, y_head, color="red")
plt.xlabel("Age")
plt.ylabel("Experience")
plt.show()

# show how to find b0 and b1
# y = b0 + b1*x

b0 = linear_reg.predict(0)
print("b0: ",b0)
b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta, intercept
    
b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope