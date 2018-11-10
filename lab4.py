from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import csv


def filler():
    x0 = list()
    x1 = list()
    x2 = list()
    x3 = list()
    y_vals = list()
    with open("teams_comb.csv", "r", newline='', encoding="latin-1") as csvfile:
        row = csv.DictReader(csvfile)

        for i in row:
            x0 = np.append(x0, '0').astype(dtype=float)
            x1 = np.append(x1, i['Power']).astype(dtype=float)
            x2 = np.append(x2, i['Age']).astype(dtype=float)
            x3 = np.append(x3, i['Experience']).astype(dtype=float)
            y_vals = np.append(y_vals, i['Salary']).astype(dtype=float)
    csvfile.close()
    x_vals = np.array([x0, x1, x2, x3])
    return x_vals, y_vals

# All data
ERR = list()
x, salary = filler()
X = np.transpose(x)

reg = LinearRegression()
reg.fit(X, salary)
pred = reg.predict(X)

for i, prediction in enumerate(pred):
    ERR.append(prediction - salary[i])


loo = LeaveOneOut()

ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = salary[train_idx], salary[test_idx]

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    # there is only one y-test and y-pred per iteration over the loo.split,
    # so to get a proper graph, we append them to respective lists.

    ytests += list(y_test)
    ypreds += list(y_pred)

# MSE after LOO
rr = metrics.r2_score(ytests, ypreds)
ms_error = metrics.mean_squared_error(ytests, ypreds)

print("Leave One Out Cross Validation")
print("R^2: {:.5f}%, MSE: {:.5f}".format(rr*100, ms_error))

# TASK 2
errr = list()
ERR2 = np.square(ERR)
ERR2 = np.mean(ERR2)
print(ERR2)

actual_err = list()

for i in range(len(ytests)):
    errr.append(ypreds[i] - ytests[i])

# TASK3 I assume
plt.scatter(pred, ERR, c='g')
plt.scatter(ypreds, errr, c='r')
plt.hlines(0, 0, 40000)
plt.show()