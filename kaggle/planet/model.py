import scipy
import numpy as np

import keras.backend as K

from keras.activations import sigmoid
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.applications import vgg16

import config
import helpers


def downsample(tensor, desired_size=7):
    result = tensor.output

    while result.shape.as_list()[1] > desired_size:
        result = MaxPooling2D((2, 2))(result)

    return result


def build():
    num_classes = config.num_classes
    input_tensor = Input(shape=(224, 224, 3))

    vgg = vgg16.VGG16(input_tensor=input_tensor,
                      include_top=False,
                      classes=num_classes)

    # Freeze for training.
    for layer in vgg.layers:
        layer.frozen = True

    # Gather feature layers.
    features = [layer for layer in vgg.layers if layer.name in config.FEATURES]

    # Use a bunch of downsampled feature maps.
    net = Concatenate()(list(map(downsample, features)))

    # Block 1.
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(256, (3, 3), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 2.
    net = Conv2D(256, (1, 1), padding='same')(net)
    net = Conv2D(256, (3, 3), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 3.
    net = Conv2D(256, (1, 1), padding='same')(net)
    net = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 4.
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(num_classes, (3, 3), strides=(2, 2), padding='valid')(net)
    net = Lambda(sigmoid)(net)
    net = Lambda(lambda x: K.squeeze(x, 1))(net)
    net = Lambda(lambda x: K.squeeze(x, 1))(net)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[net])
    model.compile(Adam(1e-3), 'categorical_crossentropy', metrics=['accuracy'])

    return model


def load_image(image_path, fileformat='%s/%s.jpg'):
    filename = fileformat % (config.image_dir, image_path)

    # Make things better.
    image = scipy.misc.imread(filename)
    image = image[:,:,:3]
    image = np.float32(image)
    image = scipy.misc.imresize(image, config.image_shape[:-1])
    image = vgg16.preprocess_input(image)

    return image


def batch_generator(labels, batch_size):
    x = np.zeros([batch_size] + config.image_shape)
    y = np.zeros([batch_size] + [config.num_classes])

    while True:
        # Load up a batch.
        for i in range(batch_size):
            index = np.random.randint(len(labels))

            x[i] = load_image(labels[index][0])
            y[i] = labels[index][1]

        yield x, y


def get_datagen(train_path, batch_size, validation_split=0.05):
    labels = helpers.get_labels(train_path)

    index = round(validation_split * len(labels))
    datagen = batch_generator(labels[index:], batch_size)
    valid_datagen = batch_generator(labels[:index], batch_size)

    return datagen, valid_datagen


if __name__ == '__main__':
    model = build()
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)

    model.fit_generator(datagen, steps_per_epoch=1000, epochs=10,
                        validation_data=valid_datagen, validation_steps=100)
