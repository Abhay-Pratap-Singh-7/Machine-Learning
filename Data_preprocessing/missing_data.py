import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("/Users/abhay/Desktop/Machine Learning/Machine_Learning_A-Z/Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # missing_values = np.nan is optional as it is default and strategy = 'mean' is also default
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)