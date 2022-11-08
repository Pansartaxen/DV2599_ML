import pandas as pd
from matplotlib import pyplot as plt

def get_data():
    df = pd.read_csv('spambase.csv')
    return df

def lgg_conj(H : pd.DataFrame, x : pd.DataFrame):
    """Returns the least general generalization of H and x"""
    return H.where(H == x, True)

def lgg_set(data : pd.DataFrame):
    row = data.iloc[0]
    H = row
    for row in data.iloc[1:]:
        x = row
        H = lgg_conj(H, x)
    return H

def normalized_to_values(dF : pd.DataFrame):
    """Changes the normalized values to either low (0-0.33) or mid (0.33-0.66) or high (0.66-1)"""
    for row in range(dF.shape[0]): # for each row
        for col in range(dF.shape[1]): # for each column
            if dF.iloc[row, col] < 0.5:
                dF.iloc[row, col] = 0
            else:
                dF.iloc[row,col] = 1
    return dF

def normalize(dF : pd.DataFrame):
    """Returns a normalized version of the input DataFrame"""
    return dF.iloc[:,0:-4].apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=1)
    # axis is 1 because we want to normalize each row, not each column

def make_lgg(dF : pd.DataFrame):
    noramlized = normalize(dF)
    to_values = normalized_to_values(noramlized)
    return to_values
    get_lgg = lgg_set(to_values)
    return get_lgg

def find_amount_of_zeros(dF : pd.DataFrame):
    """Returns the amount of zeros in the dataframe"""
    return dF[dF == 0].count().sum()

if __name__ == '__main__':
    print('På grinden!')
    df = get_data()
    print(find_amount_of_zeros(df))

    spam = df.loc[df["spam"] == 1].copy(deep=True)
    spam_lgg = make_lgg(spam)
    print(spam_lgg.head())
    print("\n\n\n")

    nonspam = df.loc[df["spam"] == 0].copy(deep=True)
    nonspam_lgg = make_lgg(nonspam)
    print(nonspam_lgg.head())
