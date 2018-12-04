
# %% Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# %% Read data

df = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1', nrows=200)
test_df = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1', skiprows=range(1, 201), nrows=38)

feature_cols = ['FSP.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'NPA.1']
response_cols = ['ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1']

# train data
X_train = df[feature_cols]
y_train = df[response_cols]

X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
# test data
X_test = test_df[feature_cols]
y_test = test_df[response_cols]

# %% mse arrays
mse_4 = np.array([])
mse_auto = np.array([])
mse_sqrt = np.array([])


# %% Loop
for n in range(1, 201):
    reg_4=RandomForestRegressor(max_depth=7,n_estimators=n,max_features=4)
    reg_4.fit(X_train, y_train)
    np.append(mse_4, np.mean(np.square(y_test - reg_4.predict(X_test))))
    
    reg_auto=RandomForestRegressor(max_depth=7,n_estimators=n,max_features='auto')
    reg_auto.fit(X_train, y_train)
    np.append(mse_auto, np.mean(np.square(y_test - reg_auto.predict(X_test))))
    
    reg_sqrt=RandomForestRegressor(max_depth=7,n_estimators=n,max_features='sqrt')
    reg_sqrt.fit(X_train, y_train)
    np.append(mse_sqrt, np.mean(np.square(y_test - reg_sqrt.predict(X_test))))

# %% After the loop ends, do the regression two more times

reg_d7=RandomForestRegressor(max_depth = 7, n_estimators = 200, max_features = 4)
reg_d7.fit(X_train, y_train)
y_pred_depth7 = reg_d7.predict(X_test)

reg_d1=RandomForestRegressor(max_depth = 1, n_estimators = 200, max_features = 4)
reg_d1.fit(X_train, y_train)
y_pred_depth1 = reg_d1.predict(X_test)

# %% Plotting