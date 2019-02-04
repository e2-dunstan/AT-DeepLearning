from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt

#load fashion mnist
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

#set up layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #images from 2D array of 28 by 28 to 1D array of 784
    keras.layers.Dense(128, activation=tf.nn.relu), #fully/densely connected neural layer
    keras.layers.Dense(10, activation=tf.nn.softmax) #ditto and softmax layer. Probability scores of it belonging to the class
])

#Loss func: how accurate model is during training
#Optimiser: how it's updated from data and loss func
#Metrics: monitor training and testing steps
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#training the model
model.fit(train_images, train_labels, epochs=1)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#make predictions
predictions = model.predict(test_images)

label = np.argmax(predictions[0])
print(class_names[label])

#https://github.com/mtobeiyf/sketch-to-art
#https://arxiv.org/abs/1704.03477
