import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/abhay/Desktop/Machine Learning/Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X = ss_X.fit_transform(X)
y = y.reshape(len(y),1)
y = ss_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

print(ss_y.inverse_transform(regressor.predict(ss_X.transform([[6.5]]).reshape(1,-1))))



