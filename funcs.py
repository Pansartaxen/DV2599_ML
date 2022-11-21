import pandas as pd
from matplotlib import pyplot as plt

def get_data():
    df = pd.read_csv('spambase.csv')
    return df

def lgg_conj(H : pd.DataFrame, x : pd.DataFrame):
    """Returns the least general generalization of H and x"""
    y = H.copy()
    #y = pd.DataFrame(columns=H.columns)
    for col in range(H.shape[0]-4):
        a,b = H.iloc[col], x.iloc[col]
        if H.iloc[col] or x.iloc[col]:
            y.iloc[col] = 1
        # elif H.iloc[col] == -1 or x.iloc[col] == -1:
        #     y.iloc[col] = 1
        else:
            y.iloc[col] = 0
    return y

def lgg_set(data : pd.DataFrame):
    H = data.iloc[0]
    for row in range(data.shape[0]):
        x = data.iloc[row]
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

def binning(dF : pd.DataFrame):
    """Binns the data"""
    for col in range(dF.shape[1]-4):
        dF = dF.sort_values(by=dF.columns[col], ascending=False)
        for row in range(dF.shape[0]):
            # if dF.iloc[row, col] > 0:
            #     dF.iloc[row, col] = 1
            if dF.iloc[row,col] != 0 and row < int(dF.shape[0]*0.30):
                dF.iloc[row,col] = 1
            elif dF.iloc[row,col] == 0 and row < int(dF.shape[0]*0.30): # if the value is 0 and the row is in the top 30%
                dF.iloc[row,col] = -1
            else:
                dF.iloc[row,col] = 0
    return dF

if __name__ == '__main__':
    print('På grinden!')
    df = get_data()
    print(find_amount_of_zeros(df))

    spam = df.loc[df["spam"] == 1].copy(deep=True)
    #spam_lgg = make_lgg(spam)
    #print(spam_lgg.head())
    #print("\n\n\n")

    #nonspam = df.loc[df["spam"] == 0].copy(deep=True)
    #nonspam_lgg = make_lgg(nonspam)
    #print(nonspam_lgg.head())
    lgg_set(binning(spam)).to_csv('lgg.csv')
    #binning(df)
    print('Done!')
