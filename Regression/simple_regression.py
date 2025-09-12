import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("/Users/abhay/Desktop/Machine Learning/Regression/Salary_Data.csv")

X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exprience')
plt.xlabel('Salary')
plt.ylabel('Exprience')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exprience')
plt.xlabel('Salary')
plt.ylabel('Exprience')
plt.show()

# to predict result for particular input
print(regressor.predict([[12]]))

# to get values of coefficients
print(regressor.coef_)
print(regressor.intercept_)