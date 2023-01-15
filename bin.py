def train_cnn_old():
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

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    model.summary()

    history = model.fit(datagen.flow(vector_train, classes_train, batch_size = 128), epochs = 20, validation_data = (vector_test, classes_test), callbacks = [learning_rate_reduction])
    print("Accuracy of the model is - " , model.evaluate(vector_test, classes_test)[1]*100 , "%")
    model.save("HandsignInterpreter/my_model")
