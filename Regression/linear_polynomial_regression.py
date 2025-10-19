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
regressor = PolynomialFeatures(degree=4)
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

# for higher resolution and smoother curve
# X_grid = np.arange(min(X), max(X), 0.1) means creating a vector with a step size of 0.1 and then reshaping it to a column vector
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lr2.predict(regressor.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lr.predict([[6.5]]))
print(lr2.predict(regressor.fit_transform([[6.5]])))
