from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
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
    model.add(Convolution2D(32, 2, 2, border_mode='same', name='conv2'))
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


def get_datagen(n, batch_size=2, train=True, vgg=False):
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    if train:
        source = data.train
    else:
        source = data.validation

    x = source.images[:n]
    x = np.reshape(x, (len(x), IMAGE_SIZE, IMAGE_SIZE, 1))

    if vgg:
        x = helpers.convert_greyscale_to_rgb(x)
        x = helpers.resize_dataset(x, 224, 224)

    y = source.labels[:n]

    # Batchwise generator.
    x_batch = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])
    y_batch = np.zeros([batch_size, 10])

    while True:
        index = 0

        for i in range(batch_size):
            x_batch[i] = x[index]
            y_batch[i] = y[index]

            index = (index + 1) % n

        yield x_batch, y_batch


def train(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer="adadelta",
                  metrics=['accuracy'])

    datagen = get_datagen(1000, 32)
    datagen_val = get_datagen(100, 32, train=False)

    checkpointer = ModelCheckpoint(filepath=SAVE_FILE, verbose=1)
    csv_logger = CSVLogger(LOG_FILE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.001)
    debugger = helpers.ManualInspection(model)
    tensorboard = TensorBoard(histogram_freq=1,
                              write_grads=True,
                              write_images=True)

    model.fit_generator(datagen,
                        steps_per_epoch=100, epochs=100,
                        validation_data=datagen_val, validation_steps=10,
                        callbacks=[checkpointer,
                                   csv_logger,
                                   reduce_lr,
                                   debugger,
                                   tensorboard])

    model.save_weights(SAVE_FILE)


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
    model = load_model()

    train(model)
