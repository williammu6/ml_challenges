import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def get_data():
    training_data = []
    labels = []
    df = pd.read_csv('dataset/train.csv')
    for data in tqdm(df.values):
        label = data[0]
        img = np.array(data[1:], dtype='uint8').reshape(28, 28)/255.0
        training_data.append(img)
        labels.append(label)

    return np.array(training_data), np.array(labels)

def get_model():
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(28, 28, 1, )))
                    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    

    opt = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
    
if __name__ == '__main__':
    
    train_x, train_y = get_data()
    model = get_model()

    train_x = train_x.reshape(-1, 28, 28, 1)
    model.fit(train_x, train_y, epochs=5)

    model.save('model.h5')
