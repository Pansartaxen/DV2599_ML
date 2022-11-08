import pandas as pd
from matplotlib import pyplot as plt

def get_data():
    df = pd.read_csv('spambase.csv')
    return df

def lgg_conj(H, x):
    pass

def lgg_set(data):
    first = data[0]
    H = first
    for row in data:
        x = row

def normalized_to_values(dF : pd.DataFrame):
    """Changes the normalized values to either low (0-0.33) or mid (0.33-0.66) or high (0.66-1)"""
    for row in dF.index:
        for col in dF.columns:
            if df.loc[row, col] < 0.5:
                df.loc[row, col] = 0
            else:
                dF.loc[row,col] = 1
    return dF

def normalize(dF : pd.DataFrame):
    """Returns a normalized version of the input DataFrame"""
    return dF.iloc[:,0:-4].apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=1)
    # axis is 1 because we want to normalize each row, not each column

if __name__ == '__main__':
    print('På grinden!')
    df = get_data()
    spam = df.loc[df["spam"] == 1]
    nonspam = df.loc[df["spam"] == 0]
    norm_spam = normalize(spam)
    norm_nonspam = normalize(nonspam)
    val_spam = normalized_to_values(norm_spam)
    val_nonspam = normalized_to_values(norm_nonspam)
