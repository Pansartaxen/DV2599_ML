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

def discrete_transform(spam: pd.DataFrame, nonspam: pd.DataFrame):
    cols = spam.columns.values.tolist()
    discrete = pd.DataFrame(columns=cols)
    discrete.loc[0] = False
    for col in spam.columns:
        if spam[col].mean() - nonspam[col].mean() > 0.1:
            print(col)
            discrete[col] = True
    return discrete

def normalize(dF : pd.DataFrame):
    return df.iloc[:,0:-1].apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=0)

if __name__ == '__main__':
    df = get_data()
    spam = df.loc[df["spam"] == 1]
    nonspam = df.loc[df["spam"] == 0]
    norm_spam = normalize(spam)
    