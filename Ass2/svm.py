import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Marius Stokkedal & Sebastian Bengtsson
# Implementation and testing of SVM algorithm


def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('Ass2\spambase.csv')
    return df

def df_to_list(data: pd.DataFrame):
    """Returns the dataframe as vector and classes with the same index"""
    vector = [data.iloc[row][:-1].tolist() for row in data.index]
    classes = [data.iloc[row][-1] for row in data.index]
    return vector,classes

if "__main__" == __name__:
    data = get_data()
    vectors,classes = df_to_list(data)

    vector_train, vector_test, train_class, test_class = train_test_split(vectors, classes, test_size=0.2)

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

    svm.fit(vector_train_std, train_class)

    # The fit method is used to train the model using the training da

    y_pred = svm.predict(vector_test_std)
    # The predict method is used to predict the class of the test
    print(vector_test_std)

    print(f"Accuracy: {accuracy_score(test_class, y_pred)}")