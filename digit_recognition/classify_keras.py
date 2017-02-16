from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


SAVE_FILE = 'model_weights.h5'
NUM_CLASSES = 10
IMAGE_SIZE = 28


def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    return


def load_model(used_saved=False):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='valid',
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                            name='conv1'))
    model.add(Activation('relu', name='relu1'))
    model.add(Convolution2D(16, 2, 2, border_mode='valid', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Flatten(name='flatten2'))
    model.add(Dense(256, name='fc3'))
    model.add(Activation('relu', name='relu3'))
    model.add(Dense(NUM_CLASSES, name='fc4'))
    model.add(Activation('softmax', name='softmax4'))

    if used_saved:
        model.load_weights(SAVE_FILE, by_name=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer="adadelta",
                  metrics=['accuracy'])

    return model


def train(model):
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x_train = data.train.images
    x_train = np.reshape(x_train, (len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1))
    y_train = data.train.labels

    model.fit(x_train, y_train, batch_size=32, nb_epoch=2, verbose=1)

    model.save_weights(SAVE_FILE)
    return


def test(model):
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x_sample = data.test.images[0]
    x_sample = np.reshape(x_sample, (1, IMAGE_SIZE, IMAGE_SIZE, 1))

    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer('pool2').output)
    intermediate_output = intermediate_layer_model.predict(x_sample)

    # Visualize activation layers of max pool 2.
    for i in range(16):
        show(intermediate_output[0, :, :, i])


model = load_model(True)

for layer in model.layers:
    print(layer, layer.name)

test(model)
