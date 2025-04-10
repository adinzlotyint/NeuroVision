import tensorflow as tf
import keras

def load_data():
    # Ładujemy zbiór CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalizacja – przeskalowanie pikseli do zakresu [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # One-hot encoding etykiet (10 klas)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)