import matplotlib.pyplot as plt
import pandas as pd

housing = pd.read_csv('/Users/abhay/Desktop/Machine Learning/housing.csv')

# Plotting longitude and latitude
housing.plot(kind="scatter", x="longitude", y="latitude" , alpha=0.4 , 
             s = housing["population"]/100, figsize=(10,7), label='Population',
             c = housing['median_house_value'], cmap=plt.get_cmap("jet"), colorbar=True)
plt.title('Location')
plt.show()

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()