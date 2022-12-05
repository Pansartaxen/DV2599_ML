import pandas as pd
from sklearn.model_selection import train_test_split

# Marius Stokkedal & Sebastian Bengtsson
# Implementation and testing of naive bayes algorithm

def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('spambase.csv')
    return df




if "__main__" == __name__:
    data = get_data()
