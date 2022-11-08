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

def above_zero(data: pd.DataFrame):
    words = []
    for col in data.columns:
        cur = data[col].sort_values(ascending=False)[:int(data.shape[0]*0.70)]
        if cur.min() > 0:
            words.append(col)
    return words

def get_diff(data: pd.DataFrame, data2: pd.DataFrame):
    words2 = above_zero(data)
    words1 = above_zero(data2)
    words = [col for col in words1 if col not in words2]
    return words

def normalized_to_values(dF : pd.DataFrame):
    """Changes the normalized values to either low (0-0.33) or mid (0.33-0.66) or high (0.66-1)"""
    for row in dF.index:
        for col in dF.columns:
            if df.loc[row, col] == 0:
                df.loc[row, col] = 0
            elif dF.loc[row,col] < 0.33:
                dF.loc[row,col] = 1
            elif dF.loc[row,col] < 0.66:
                dF.loc[row,col] = 2
            else:
                dF.loc[row,col] = 3
    return dF

def discrete_transform(spam: pd.DataFrame, nonspam: pd.DataFrame):
    cols = spam.columns.values.tolist()
    discrete = pd.DataFrame(columns=cols)
    discrete.loc[0] = False

    for i in range(spam.shape[1]):
        spam_col_sum = 0
        nonspam_col_sum = 0
        spam_count = 0
        nonspam_count = 0

        for j in range(spam.shape[0]):
            if spam.iloc[j,i] in [1,2,3]:
                spam_col_sum += spam.iloc[j,i]
                spam_count += 1
            if nonspam.iloc[j,i] in [1,2,3]:
                nonspam_col_sum += nonspam.iloc[j,i]
                nonspam_count += 1

        spam_mean = spam_col_sum / spam_count
        nonspam_mean = nonspam_col_sum / nonspam_count

        if int(spam_mean) != int(nonspam_mean) and spam_mean > 1:
            discrete.iloc[0,i] = True
            print(cols[i])

    # for col in spam.columns:
    #     if spam[col] != nonspam[col] and spam[col] != "low":
    #         print(col)
    #         discrete[col] = True
    return discrete

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
    discrete = discrete_transform(val_spam, val_nonspam)
    print(discrete)
