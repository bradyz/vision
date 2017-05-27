import sys

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Sequential, Model

import helpers


SAVE_FILE = 'model_weights.h5'
LOG_FILE = 'training.log'
NUM_CLASSES = 10
IMAGE_SIZE = 28


def load_model(used_saved=False):
    model = Sequential()
    model.add(Convolution2D(32, 10, 10, border_mode='same',
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                            name='conv1'))
    model.add(Activation('relu', name='relu1'))
    model.add(Convolution2D(16, 2, 2, border_mode='same', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Flatten(name='flatten2'))
    model.add(Dense(256, name='fc3'))
    model.add(Dropout(0.25))
    model.add(Activation('relu', name='relu3'))
    model.add(Dense(NUM_CLASSES, name='fc4'))
    model.add(Activation('softmax', name='softmax4'))

    if used_saved:
        model.load_weights(SAVE_FILE, by_name=True)

    return model


def train(model):
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer="adadelta",
                  metrics=['accuracy'])

    x_train = data.train.images[:1000]
    x_train = np.reshape(x_train, (len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1))
    x_train = helpers.convert_greyscale_to_rgb(x_train)
    x_train = helpers.resize_dataset(x_train, 224, 224)

    y_train = data.train.labels[:1000]

    x_valid = data.validation.images[:100]
    x_valid = np.reshape(x_valid, (len(x_valid), IMAGE_SIZE, IMAGE_SIZE, 1))
    x_valid = helpers.convert_greyscale_to_rgb(x_valid)
    x_valid = helpers.resize_dataset(x_valid, 224, 224)

    y_valid = data.validation.labels[:100]

    checkpointer = ModelCheckpoint(filepath=SAVE_FILE, verbose=1)
    csv_logger = CSVLogger(LOG_FILE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.001)
    debugger = helpers.ManualInspection(model)

    model.fit(x_train, y_train, batch_size=32, nb_epoch=20, verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer, csv_logger, reduce_lr, debugger])

    model.save_weights(SAVE_FILE)
    return


def debug(model):
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x_sample = data.test.images[0]
    x_sample = np.reshape(x_sample, (1, IMAGE_SIZE, IMAGE_SIZE, 1))

    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer('pool2').output)
    intermediate_output = intermediate_layer_model.predict(x_sample)

    # Visualize activation layers of max pool 2.
    for i in range(16):
        helpers.show(intermediate_output[0, :, :, i])

    conv1_weights = model.get_layer('conv1').get_weights()[0]
    helpers.view_filters(conv1_weights)


def load_model_vgg():
    img_input = Input(tensor=Input(shape=(224, 224, 3)))

    base_model = VGG16(include_top=False, input_tensor=img_input)

    # Freeze all layers in pretrained network.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=img_input, output=x)
    model.load_weights(SAVE_FILE, by_name=True)

    return model


if __name__ == "__main__":
    model = load_model_vgg()

    train(model)
