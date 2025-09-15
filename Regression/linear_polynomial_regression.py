import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/abhay/Desktop/Machine Learning/Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

from sklearn.preprocessing import PolynomialFeatures
regressor = PolynomialFeatures(degree=2)
X_poly = regressor.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_poly, y)
y_pred2 = lr2.predict(X_poly)

plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.plot(X, y_pred2, color='orange')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()  
