import pandas as pd

def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('Ass2\spambase.csv')
    return df

def friedman(data: pd.DataFrame):
    column_sum = 0
    for column in range(data.shape[1]):
        column_sum += (data.iloc[:,column].sum)**2

    cols = data.shape[1]
    rows = data.shape[0]
    return (12/(rows*cols*(cols+1))) * column_sum - 3*rows*(cols+1)

data = get_data()