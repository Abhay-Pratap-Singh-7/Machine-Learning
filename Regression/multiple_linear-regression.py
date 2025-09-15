'''

backward elimination (fastest and most common method)
1. select a significance level to stay in the model (SL=0.05)
2. fit the full model with all possible predictors
3. consider the predictor with the highest P-value. If P>SL, go to step 4, otherwise the model is ready.
4. remove the predictor
5. fit the model without this variable and go to step 3.

forward selection
1. select a significance level to enter the model (SL=0.05)
2. fit all simple regression models y~x_i for each predictor x_i.
3. select the predictor with the lowest P-value. If P<SL, go to step 4, otherwise the model is ready.
4. fit all models with one more predictor added to the predictors selected so far.
5. select the predictor with the lowest P-value. If P<SL, go to step 4, otherwise the model is ready.

bidirectional elimination
1. select a significance level to enter and to stay in the model (SL=0.05)
2. perform the next step of forward selection
3. perform all steps of backward elimination
4. the model is ready when neither of the above has any effect.

all possible models
1. fit all possible models with all combinations of predictors.
2. select the model with the lowest AIC or BIC.
total number of models = 2^k -1 (k is the number of predictors)

'''

# multiple linear regression with backward elimination

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/abhay/Desktop/Machine Learning/Regression/50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print(regressor.predict([[1,0,0,100000,140000,300000]]))

print(regressor.coef_)
print(regressor.intercept_)