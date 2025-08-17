import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/abhay/Desktop/Machine Learning/Dataset/housing.csv")

data.plot(kind = 'scatter', x = "median_income" , y = "median_house_value" , figsize=(8,8) , alpha = 0.1)
plt.show()

