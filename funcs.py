import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import random

def get_data():
    df = pd.read_csv('spambase.csv')
    return df

def lgg_conj(H : pd.DataFrame, x : pd.DataFrame):
    """Returns the least general generalization of H and x"""
    y = H.copy()
    for col in range(H.shape[0]):
        a,b = H.iloc[col], x.iloc[col]
        if H.iloc[col] == x.iloc[col] and H.iloc[col]:
            y.iloc[col] = 1
        else:
            y.iloc[col] = 0
    return y

def lgg_set(data : pd.DataFrame):
    H = data.iloc[0]
    for row in range(data.shape[0]):
        x = data.iloc[row]
        H = lgg_conj(H, x)
    return H

def continuous_to_discrete(data : pd.DataFrame):
    """If value is above 0.01, set to 1, else 0"""
    retDF = data.copy()
    for col in data.columns:
        retDF[col] = retDF[col].apply(lambda x: 1 if float(x) > 0.01 else 0)
    return retDF

def test_performance(data: pd.DataFrame, H: pd.Series):
    """Returns the accuracy of the hypothesis"""
    spam_as_spam = 0
    spam_as_ham = 0
    ham_as_ham = 0
    ham_as_spam = 0

    correct = 0

    H_values = []
    for keys in H.keys():
        if H[keys] == True:
            H_values.append(keys)
    print(H_values)
    for row in range(data.shape[0]):
        spam = True
        for col in range(data.shape[1]-1):
            if H.iloc[col]:
                if not data.iloc[row, col]:
                    spam = False
        if spam:
            if data.iloc[row, -1]:
                spam_as_spam += 1
                correct += 1
            else:
                ham_as_spam += 1
        else:
            if data.iloc[row, -1]: 
                spam_as_ham += 1
            else:
                ham_as_ham += 1
                correct += 1
        
    print("good Spam as spam: ", spam_as_spam)
    print("bad Spam as ham: ", spam_as_ham)
    print("good Ham as ham: ", ham_as_ham)
    print("bad Ham as spam: ", ham_as_spam)
    print("Accuracy: ", correct/data.shape[0])
    return correct/data.shape[0]

def most_above_zero(dF : pd.DataFrame):
    """Returns the columns with the percentage of values above 0 and sorted by the percentage"""
    above_zero = {}
    for col in dF.columns:
        above_zero[col] = dF[dF[col] > 0].count()[col]/dF.shape[0]
    return sorted(above_zero.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    print('På grinden!')
    df = get_data()

    spam = df[df['spam'] == 1].copy()
    ham = df[df['spam'] == 0].copy()
    
    initiated = True

    while initiated or len(h[h==1]) < 5:
        initiated = False
        train_data, split_data = train_test_split(spam, test_size=0.997)

        train_data = continuous_to_discrete(train_data)
        split_data = continuous_to_discrete(split_data)

        h = lgg_set(train_data)

    old_joined = [ham, split_data]
    joined = pd.concat(old_joined)

    print(test_performance(joined, h))

    print("Bra pluggat idag grabbar!")

    
