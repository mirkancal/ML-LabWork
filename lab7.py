import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import SVR

#%% Importing the dataset


df = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1', nrows=200)
test_df = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1', skiprows=range(1, 201), nrows=38)

feature_cols = ["WNR.1", "WNR.2"]
response_cols= ["Result"]

X_train = df[feature_cols]
y_train = df[response_cols]

X_test = test_df[feature_cols]
y_test = test_df[response_cols]

#%% Visualize
def VisualizeSVM(kernelType):

    classifier = SVC(kernel=kernelType)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    plt.figure()
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
           plt.scatter(X_test.values[i, 0], X_test.values[i,1], c = 'red')
         
        else:
           plt.scatter(X_test.values[i,0], X_test.values[i,1], c = 'blue')

    plt.title(kernelType)



VisualizeSVM('linear')
VisualizeSVM('poly')
VisualizeSVM('rbf')

plt.show()
