import cv2
import glob
import numpy as np

colour_images = []
lined_images = []

filePath = "/Dataset/*RESIZED.jpg"

files = glob.glob(filePath)
for file in files:
    img = cv2.imread(file)
    if "lines" in file:
        lined_images.append(img)
    else:
        colour_images.append(img)













#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras import backend as K

#list of layer instances
#none means expect any input shape
#model = Sequential([
#    Dense(32, input_shape=None),
#    Activation('relu'),
#    Dense(10),
#    Activation('softmax'),
#    ])
#
#def mean_pred(y_true, y_pred):
#    return K.mean(y_pred)
#
#model.compile(optimiser = 'rmsprop',
#              loss = 'binary_crossentropy',
#              metrics=['accuracy', mean_pred])

