import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn import tree

# Marius Stokkedal & Sebastian Bengtsson
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


def row_ranker(row: list) -> list:
    score = 1
    score_dict = {}
    for val in set(row):
        count = row.count(val)
        tot = 0
        for i in range(count):
            tot += i+score
        score += count
        score_dict[val] = tot/count
    return [score_dict[val] for val in row]

def knn(vector_train, vector_test, class_train, class_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(vector_train, class_train)
    spam_pred = model.predict(vector_test)
    return accuracy_score(class_test, spam_pred)

def svm(vector_train, vector_test, class_train, class_test):
    sc = StandardScaler()
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
    y_pred = svm.predict(vector_test_std)
    return accuracy_score(class_test, y_pred)

def dec_tree(vector_train, vector_test, class_train, class_test):
    DT = tree.DecisionTreeClassifier()
    DT = DT.fit(vector_train, class_train)
    spam_pred = DT.predict(vector_test)
    return accuracy_score(class_test, spam_pred)


if "__main__" == __name__:
    data = get_data()
    vectors,classes = df_to_list(data)

    print("-"*50)
    print("Fold | KNN | SVM | Decision Tree")
    print("-"*50)
    knn_tot = []
    svm_tot = []
    dec_tot = []

    for round in range(0,10):
        train_vector = vectors.copy()
        train_classes = classes.copy()

        test_bucket_vector = train_vector[round::10]
        test_bucket_classes = train_classes[round::10]
        del train_vector[round::10]
        del train_classes[round::10]
        
        k = knn(train_vector, test_bucket_vector, train_classes, test_bucket_classes)
        knn_tot.append(k)

        s = svm(train_vector, test_bucket_vector, train_classes, test_bucket_classes)
        svm_tot.append(s)

        t = dec_tree(train_vector, test_bucket_vector, train_classes, test_bucket_classes)
        dec_tot.append(t)

        print(f"{round+1} | {k:.4f} | {s:.4f} | {t:.4f}")
    knn_stdev = np.std(knn_tot)
    svm_stdev = np.std(svm_tot)
    dec_stdev = np.std(dec_tot)

    print("-"*50)
    print(f"Average | {sum(knn_tot)/len(knn_tot):.4f} | {sum(svm_tot)/len(svm_tot):.4f} | {sum(dec_tot)/len(dec_tot):.4f}")
    print(f"Stdv    | {knn_stdev:.4f} | {svm_stdev:.4f} | {dec_stdev:.4f}")
    print("-"*50)

