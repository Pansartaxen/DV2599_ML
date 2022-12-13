import pandas as pd
import numpy as np
from time import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


letters = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

def get_data():
    """Returns the data as a pandas dataframe"""
    try:
        df = pd.read_csv('HandsignInterpreter/training_data/sign_mnist_train.csv')
    except:
        df = pd.read_csv(r'HandsignInterpreter\training_data\sign_mnist_train.csv')
    return df

def get_test_data():
    """Returns the data as a pandas dataframe"""
    try:
        df = pd.read_csv('HandsignInterpreter/training_data/sign_mnist_test.csv')
    except:
        df = pd.read_csv(r'HandsignInterpreter\training_data\sign_mnist_test.csv')
    return df

def df_to_list(data: pd.DataFrame):
    """Returns the dataframe as vector and classes with the same index"""
    vector = [data.iloc[row][1:].tolist() for row in data.index]
    classes = [data.iloc[row][0] for row in data.index]
    return vector,classes

def accuracy_check(pred, actual):
    """
    pred = predicted values
    actual = actual values

    returns the f1 score
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for p,a in zip(pred,actual):
        if p and a:
            tp += 1
        elif not p and not a:
            tn += 1
        elif p and not a:
            fp += 1
        elif not p and a:
            fn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return 2 * (recall * precision) / (recall + precision)

def train_random_forest():
    """Returns the accuracy of the random forest classifier"""
    clf = RandomForestClassifier()
    df_train = get_data()
    vector_train, classes_train = df_to_list(df_train)
    clf.fit(vector_train, classes_train)

    pickle.dump(clf, open('HandsignInterpreter/finalized_model_RF.sav', 'wb'))
    #return accuracy_check(pred, classes_test), clf

def train_svm():
    """Evalmeasure is 1 for time and 2 for accuracy and 3 for f1"""
    sc = StandardScaler()

    df_train = get_data()
    vector_train, classes_train = df_to_list(df_train)

    sc.fit(vector_train)

    vector_train_std = sc.transform(vector_train)

    svm = SVC(kernel='linear', C=0.05, random_state=1)

    svm.fit(vector_train_std, classes_train)

    pickle.dump(svm, open('HandsignInterpreter/finalized_model_svm.sav', 'wb'))

def classify_image_RF(image, clf):
    """Returns the letter that the image is classified as"""
    letter = clf.predict(image)
    ret_let = letters[letter[0]]
    return ret_let

def classify_image_svm(image, svm):
    """Returns the letter that the image is classified as"""
    sc = StandardScaler()
    image_std = sc.transform(image)
    letter = svm.predict(image_std)
    return letters[letter]

if __name__ == "__main__":
    print("Running algorithms.py as main")
    rm_acc, random_forest = train_random_forest()