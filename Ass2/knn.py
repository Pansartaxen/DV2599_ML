import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

if "__main__" == __name__:
    data = get_data()
    vectors,classes = df_to_list(data)

    feature_train, feature_test, class_train, class_test = train_test_split(vectors, classes, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(feature_train, class_train)
    spam_pred = model.predict(feature_test)
    print(f"Accuracy: {accuracy_score(class_test, spam_pred)}")
