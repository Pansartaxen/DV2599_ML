import pandas as pd
import numpy as np
from time import time,sleep

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn import tree

from friedman import friedman
import random

from scipy import stats
# Marius Stokkedal & Sebastian Bengtsson
# Blekinge Institute of Technoligy, Karlskrona, Sweden
# Created 5 dec 2022
# Implementation and testing of KNN algorithm

def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('Ass2\spambase.csv')
    return df

def df_to_list(data: pd.DataFrame):
    """Returns the dataframe as vector and classes with the same index"""
    vector = [data.iloc[row][:-1].tolist() for row in data.index]
    classes = [data.iloc[row][-1] for row in data.index]
    return vector,classes

def accuracy_check(pred, actual, mode):
    """
    pred = predicted values
    actual = actual values
    mode = accuracy or f1

    returns the accuracy or f1 score
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

    if mode == "accuracy":
        return (tp + tn) / (tp + tn + fp + fn)
    
    elif mode == "f-measure":
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)

def knn(vector_train, vector_test, class_train, class_test, eval_measure=2):
    """Evalmeasure is 1 for time and 2 for accuracy and 3 for f1"""

    if eval_measure == "Time":
        time_start = time()

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(vector_train, class_train)

    if eval_measure == "Time":
        time_end = time()
        return time_end - time_start

    spam_pred = model.predict(vector_test)
    return accuracy_check(spam_pred, class_test, eval_measure)

def svm(vector_train, vector_test, class_train, class_test, eval_measure):
    """Evalmeasure is 1 for time and 2 for accuracy and 3 for f1"""
    sc = StandardScaler()

    if eval_measure == "Time":
        time_start = time()

    sc.fit(vector_train)

    vector_train_std = sc.transform(vector_train)
    vector_test_std = sc.transform(vector_test)

    svm = SVC(kernel='linear', C=0.05, random_state=1)

    # C value is the penalty parameter of the error term e.g. the cost of misclassification
    # Higher C value means potentially higher accuracy but also higher risk of overfitting

    # The kernel parameter is used to specify the kernel type to be used in the algorithm
    # Kernel means that the data is transformed into a higher dimension
    # Linear kernel is used when the data is linearly separable
    # Which means that the data can be separated by a straight line

    # The fit method is used to train the model using the training da
    svm.fit(vector_train_std, class_train)
    
    if eval_measure == "Time":
        time_end = time()
        return time_end - time_start

    spam_pred = svm.predict(vector_test_std)
    return accuracy_check(spam_pred, class_test, eval_measure)

def dec_tree(vector_train, vector_test, class_train, class_test, eval_measure):
    """Evalmeasure is 1 for time and 2 for accuracy and 3 for f1"""
    if eval_measure == "Time":
        time_start = time()

    DT = tree.DecisionTreeClassifier()
    DT = DT.fit(vector_train, class_train)

    if eval_measure == "Time":
        time_end = time()
        return time_end - time_start

    spam_pred = DT.predict(vector_test)
    return accuracy_check(spam_pred, class_test, eval_measure)

def create_buckets(vector, classes):
    """Returns the data in 10 buckets"""

    train_vector = vector.copy()
    train_classes = classes.copy()

    bucket_train_vector = []
    bucket_train_classes = []
    for i in range(0,10):
        bucket_train_vector.append(list(train_vector[i::10])) 
        bucket_train_classes.append(list(train_classes[i::10]))

    return bucket_train_vector, bucket_train_classes

def get_training_data(bucket_train_vector, bucket_train_classes, round):
    """Returns the training data for the current round (9/10 of the data)"""
    train_vector = bucket_train_vector[:round].copy() + bucket_train_vector[round+1:].copy()
    train_classes = bucket_train_classes[:round].copy() + bucket_train_classes[round+1:].copy()

    train_vector = [item for sublist in train_vector for item in sublist]
    train_classes = [item for sublist in train_classes for item in sublist]

    return train_vector, train_classes

def get_test_data(bucket_train_vector, bucket_train_classes, round):
    """Returns the test data for the current round (1/10 of the data)"""
    test_vector = bucket_train_vector[round]
    test_classes = bucket_train_classes[round]

    return test_vector, test_classes

if "__main__" == __name__:
    data = get_data()
    vectors,classes = df_to_list(data)
    bucket_train_vector, bucket_train_classes = create_buckets(vectors, classes)

    eval_measure = {1: "Time", 2: "accuracy", 3: "f-measure"}
    print("-"*50)
    print("Fold | KNN    | SVM    | Decision Tree")
    print("-"*50)
    knn_tot = [[], [], []] # Time, accuracy, f-measure
    svm_tot = [[], [], []]
    dec_tot = [[], [], []]
    for i in range(3):
        measure = eval_measure[i+1]
        print("-"*50)
        print("Measure: ", measure)
        for round in range(0,10):
            train_vector, train_classes = get_training_data(bucket_train_vector, bucket_train_classes, round)
            test_bucket_vector, test_bucket_classes = get_test_data(bucket_train_vector, bucket_train_classes, round)

            k = knn(train_vector, test_bucket_vector, train_classes, test_bucket_classes, measure)
            knn_tot[i].append(k)

            s = svm(train_vector, test_bucket_vector, train_classes, test_bucket_classes,measure)
            svm_tot[i].append(s)
            t = dec_tree(train_vector, test_bucket_vector, train_classes, test_bucket_classes,measure)
            dec_tot[i].append(t)
            if round != 9:
                print(f"{round+1}    | {k:.4f} | {s:.4f} | {t:.4f}")
            else:
                print(f"{round+1}   | {k:.4f} | {s:.4f} | {t:.4f}")
        knn_stdev = np.std(knn_tot[i])
        svm_stdev = np.std(svm_tot[i])
        dec_stdev = np.std(dec_tot[i])

        friedman_stat = friedman(knn_tot[i].copy(), svm_tot[i].copy(), dec_tot[i].copy())
        scipy_friedman = stats.friedmanchisquare(knn_tot[i].copy(), svm_tot[i].copy(), dec_tot[i].copy())

        print("-"*50)
        print(f"avg  | {sum(knn_tot[i])/len(knn_tot[i]):.4f} | {sum(svm_tot[i])/len(svm_tot[i]):.4f} | {sum(dec_tot[i])/len(dec_tot[i]):.4f}")
        print(f"stdv | {knn_stdev:.4f} | {svm_stdev:.4f} | {dec_stdev:.4f}")
        print("-"*50)
        print(f"Friedman statistic: {friedman_stat:.4f}")
        print(f"Scipy friedman statistic: {scipy_friedman[0]:.4f}")