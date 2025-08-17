import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("/Users/abhay/Desktop/Machine Learning/Dataset/housing.csv")

# Create a new categorical variable 'income_cat' for stratified sampling
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., float('inf')],
    labels=[1, 2, 3, 4, 5]
)

# Initialize the StratifiedShuffleSplit object
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)

# Perform the stratified split
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove the 'income_cat' column from the sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Create CSV files for both the training and testing sets
strat_train_set.to_csv("strat_train_set.csv", index=False)
strat_test_set.to_csv("strat_test_set.csv", index=False)

print("\nCSV files for the training and testing sets have been created.")
print("Training set saved as: strat_train_set.csv")
print("Testing set saved as: strat_test_set.csv")