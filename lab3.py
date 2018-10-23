#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:26:25 2018

@author: mirkan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DataFrame = pd.read_csv('team.csv', encoding="latin-1")

x = DataFrame[["Age", "Experience", "Power"]]
y = DataFrame["Salary"]

x.insert(0, "pivot", 1)
 

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x, y)

coef = regression.coef_

print("Age", coef[1])
print("Experience", coef[2])
print("Power", coef[3])

y_head = regression.predict(x)
u = y_head - y;

plt.scatter(y_head, u)
plt.hlines(y=0, xmin=0, xmax=20000)
plt.show()