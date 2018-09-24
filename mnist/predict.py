import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    test_data = []
    df = pd.read_csv('dataset/test.csv')
    for data in tqdm(df.values):
        img = np.array(data, dtype='uint8').reshape(28, 28)/255.0
        test_data.append(img)

    return np.array(test_data)

if __name__ == '__main__':
    test_x = get_data()

    index = 15031

    X = test_x.reshape(-1, 28, 28, 1)
    model = tf.keras.models.load_model('model.h5')

    predictions = model.predict(test_x)
    print("Prediction: %d " % np.argmax(predictions[index]))
    plt.imshow(test_x[index])
    plt.show()
