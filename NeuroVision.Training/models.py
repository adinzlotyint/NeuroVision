import tensorflow as tf
from keras import layers, models

def create_simple_cnn():
    #Simple CNN model to classify 10 classes on 32x32 RGB data
    model = models.Sequential([
        #First convolution layer - detects basic features
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)), #Pooling, reducing img by half
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), #Pooling, reducing img by half
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(), #Flattening to Dense
        layers.Dense(64, activation='relu'), #Hidden layer
        layers.Dense(10, activation='softmax') #Classification of 10 classes
    ])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model