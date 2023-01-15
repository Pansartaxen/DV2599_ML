import pandas as pd
import numpy as np
from time import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


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

cnn_letters = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y'
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

def train_random_forest():
    """Returns the accuracy of the random forest classifier"""
    clf = RandomForestClassifier()
    df_train = get_data()
    vector_train, classes_train = df_to_list(df_train)
    clf.fit(vector_train, classes_train)

    pickle.dump(clf, open('HandsignInterpreter/finalized_model_RF.sav', 'wb'))

def sc_generator():
    """Returns the standard scaler"""
    sc = StandardScaler()
    df_train = get_data()
    vector_train, classes_train = df_to_list(df_train)
    sc.fit(vector_train)
    return sc

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
    return sc

def train_cnn():
    df = get_data()
    df_test = get_test_data()
    classes_train = df['label']
    classes_test = df_test['label']
    del df['label']
    del df_test['label']

    label_binarizer = LabelBinarizer()
    classes_train = label_binarizer.fit_transform(classes_train)
    classes_test = label_binarizer.fit_transform(classes_test)

    vector_train = df.values
    vector_test = df_test.values
    vector_train = vector_train / 255
    vector_test = vector_test / 255
    vector_train = vector_train.reshape(-1, 28, 28, 1)
    vector_test = vector_test.reshape(-1, 28, 28, 1)

    datagen = ImageDataGenerator(
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=10,  # randomly shift images horizontally (fraction of total height, if < 1, or pixels if >= 1)
        height_shift_range=10,  # randomly shift images vertically (fraction of total height, if < 1, or pixels if >= 1)
        shear_range=2, # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        channel_shift_range=0.1 # Range for random channel shifts
    )

    datagen.fit(vector_train)

    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(10 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 ,padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    model.summary()

    print("Accuracy of the model is - " , model.evaluate(vector_test, classes_test)[1]*100 , "%")
    model.save("HandsignInterpreter/my_model")

def classify_image_RF(image, clf):
    """Returns the letter that the image is classified as"""
    letter = clf.predict(image)
    ret_let = letters[letter[0]]
    return ret_let

def classify_image_svm(image, svm, sc):
    """Returns the letter that the image is classified as"""
    image_std = sc.transform(image)
    letter = svm.predict(image_std)
    ret_let = letters[letter[0]]
    return ret_let

def classify_image_cnn(image):
    """Returns the letter that the image is classified as"""
    model = keras.models.load_model("HandsignInterpreter/my_model")
    img = image
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    letter = model.predict(img)
    ret_let = np.argmax(letter, axis = 1)
    print(ret_let)
    return cnn_letters[ret_let[0]]

if __name__ == "__main__":
    print("Running algorithms.py as main")
    train_cnn()
