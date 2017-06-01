import pickle

import numpy as np

import keras.backend as K

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose


def build_generator(input_shape):
    model = Sequential()

    model.add(Conv2DTranspose(4, (3, 3), (2, 2), activation='relu',
        input_shape=input_shape))
    model.add(Conv2DTranspose(8, (3, 3), (2, 2), activation='relu'))
    model.add(Conv2DTranspose(16, (3, 3), (2, 2), activation='relu'))
    model.add(Conv2DTranspose(32, (3, 3), (2, 2), activation='relu'))

    model.compile(loss=generator_loss, optimizer='sgd')

    return model


def build_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), (2, 2), activation='relu',
        input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), (2, 2), activation='relu'))
    model.add(Conv2D(8, (3, 3), (2, 2), activation='relu'))

    return model


def get_data(dataset, batch_size=32):
    data = np.zeros(shape=(batch_size, 32, 32, 3))

    for i in range(batch_size):
        example = dataset[b'data'][i]
        example = example.reshape((3, 32, 32))
        example = example.transpose([1, 2, 0])

        data[i,:,:,:] = example

    return data


def train(dataset, latent_shape=(64, 1), input_shape=(32, 32, 3)):
    generator = build_generator(latent_shape)
    discriminator = build_discriminator(input_shape)


if __name__ == '__main__':
    with open('cifar-10-batches-py/data_batch_1', 'rb') as fd:
        data = pickle.load(fd, encoding='bytes')

    train(data)
