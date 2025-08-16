import pandas as pd
import numpy as np
housing = pd.read_csv('/Users/abhay/Desktop/Machine Learning/Dataset/housing.csv')

def split_data( data, set_ratio ):
    """
    To split the data into two parts: One for Training and Other for Testing.

    Args: 
        data = Full dataset
        set_ration = Percentage data to store for testing.
    """
    random_index = np.random.permutation(len(housing))
    test_set_size = int( set_ratio * len(housing) )
    test_set_index = random_index[:test_set_size]
    train_set_index = random_index[test_set_size:]
    test_set = housing.iloc[test_set_index]
    train_set = housing.iloc[train_set_index]
    test_set.to_csv('test_set_2.csv', index=False)
    train_set.to_csv('train_set_2.csv', index=False)

split_data( housing, 0.2)