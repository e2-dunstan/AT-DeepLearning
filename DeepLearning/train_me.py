import os, math
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import theano
import numpy as np
import glob

from matplotlib import pyplot as plt
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianNoise
from keras.layers.local import LocallyConnected2D


#number of passes of the full training set
NUM_EPOCHS = 1000
#number of training examples in one iteration
BATCH_SIZE = 8
#75/25
VALID_RATIO = 25
#learning rate: how much to adjust weights with respect to loss gradient
lr = 0.0008

def plot_scores(scores, test_scores, file_name, on_top=True):
    #clear current graph
    plt.clf()
    axes = plt.gca()
    axes.yaxis.tick_right()
    axes.yaxis.set_ticks_position('both')
    plt.plot(scores)
    plt.plot(test_scores)
    plt.xlabel('Epoch')
    location = 'upper right'
    if not on_top:
        location = 'lower right'
    plt.legend(['Train', 'Test'], loc = location)
    plt.draw()
    plt.savefig(file_name)

def add_pos(arr):
	s = arr.shape
	result = np.empty((s[0], s[1] + 2, s[2], s[3]), dtype=np.float32)
	result[:,:s[1],:,:] = arr
	x = np.repeat(np.expand_dims(np.arange(s[3]) / float(s[3]), axis=0), s[2], axis=0)
	y = np.repeat(np.expand_dims(np.arange(s[2]) / float(s[2]), axis=0), s[3], axis=0)
	result[:,s[1] + 0,:,:] = x
	result[:,s[1] + 1,:,:] = np.transpose(y)
	return result

################################
### -- LOAD TRAINING DATA -- ###
################################

print ("Loading NumPy files")
lined_train = np.load('x_data.npy').astype(np.float32) / 255.0
coloured_train = np.load('y_data.npy').astype(np.float32) / 255.0

num_samples = lined_train.shape[0]
print ("Loaded " + str(num_samples) + " samples")


#####################################
### -- CONFIGURE TRAINING DATA -- ###
#####################################

lined_train = add_pos(lined_train)

#Split data
split_ix = int(num_samples/VALID_RATIO)
lined_test = lined_train[:split_ix]
coloured_test = coloured_train[:split_ix]
lined_train = lined_train[split_ix:]
coloured_train = coloured_train[split_ix:]

#Shuffle test data
np.random.seed(0)
state = np.random.get_state()
np.random.shuffle(lined_train)
np.random.set_state(state)
np.random.shuffle(coloured_train)
lined_train_mini = lined_train[:int(len(lined_train)/VALID_RATIO)]
coloured_train_mini = coloured_train[:int(len(coloured_train)/VALID_RATIO)]

K.set_image_data_format('channels_first')

##########################
### -- CREATE MODEL -- ###
##########################

if False:
    print ("Loading Model...")
    model = load_model('PyplotGraphs/Model.h5')
    model.optimizer.lr.set_value(lr)

else:
    model = Sequential()

    model.add(Conv2D(48, (5, 5), padding='same', input_shape=lined_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(96, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(192, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(192, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(384, (5, 5), padding='same'))
    model.add(Activation("relu"))

    model.add(Conv2D(768, (5, 5), padding='same'))
    model.add(Activation("relu"))

    model.add(Conv2D(768, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(384, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(192, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(96, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(UpSampling2D(size=(2,2)))

    model.add(Conv2D(48, (5, 5), padding='same'))
    model.add(Activation("relu"))

    model.add(Conv2D(3, (1, 1), padding='same'))
    
    model.add(Activation("sigmoid"))

    model.compile(optimizer=Adam(lr=lr), loss='mse')



#########################
### -- TRAIN MODEL -- ###
#########################
print ("Training...")
#plot_model(model, to_file='PyplotGraphs/model.png', show_shapes=True)

train_rmse = []
test_rmse = []

for i in range(0, NUM_EPOCHS):
    model.fit(lined_train, coloured_train, epochs=1, batch_size=BATCH_SIZE)
    
    mse1 = model.evaluate(lined_train_mini, coloured_train_mini, batch_size=BATCH_SIZE, verbose=0)

    #ROOT MEAN SQUARED ERROR
    train_rmse.append(math.sqrt(mse1))
    print ("Train RMSE: " + str(train_rmse[-1]))

    mse2 = model.evaluate(lined_test, coloured_test, batch_size=BATCH_SIZE, verbose=0)

    test_rmse.append(math.sqrt(mse2))
    print ("Test RMSE: " + str(test_rmse[-1]))
        
    model.save('PyplotGraphs/Model.h5')
    print ("Epoch " + str(i) + " complete")
    plot_scores(train_rmse, test_rmse, "PyplotGraphs/Scores.png", True)

print ("Done")
