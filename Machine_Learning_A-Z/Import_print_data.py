# to import neccessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# to import dataset
dataset = pd.read_csv("location of dataset")

# to set x and y variable
X = dataset.iloc[:, :-1].values # it select all rows and all columns except last column
y = dataset.iloc[:, -1].values  # it select all rows and only last column

# to print x and y variable
print(X)
print(y)