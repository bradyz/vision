from PIL import Image
import numpy as np

import keras.backend as K

from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.activations import sigmoid
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.applications import vgg16

import config
import helpers


AUGMENT = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
           Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
           Image.TRANSPOSE]


def f_measure(p, r, beta_sq=2.0):
    return (1.0 + beta_sq) * (p * r) / (beta_sq * p + r + K.epsilon())


class MetricCallback(Callback):
    def __init__(self, model, valid_datagen, steps):
        self.model = model
        self.valid_datagen = valid_datagen
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):
        TP = np.zeros(config.num_classes)
        FP = np.zeros(config.num_classes)
        FN = np.zeros(config.num_classes)

        for i, x_y in enumerate(self.valid_datagen):
            if i == self.steps:
                break

            x = x_y[0]
            y_true = np.round(x_y[1]).astype('bool')
            y_pred = np.round(self.model.predict_on_batch(x)).astype('bool')

            TP += np.sum(np.logical_and(y_pred == True, y_true == True), axis=0)
            FP += np.sum(np.logical_and(y_pred == True, y_true == False), axis=0)
            FN += np.sum(np.logical_and(y_pred == False, y_true == True), axis=0)

        precision = TP / (TP + FP + K.epsilon())
        recall = TP / (TP + FN + K.epsilon())
        f_score = f_measure(precision, recall)

        average_precision = np.mean(precision)
        average_recall = np.mean(recall)
        average_f_score = f_measure(TP.sum() / (TP.sum() + FP.sum() + K.epsilon()),
                                    TP.sum() / (TP.sum() + FN.sum() + K.epsilon()))

        print('Precision:\n%s' % precision)
        print('Recall:\n%s' % recall)
        print('F_score:\n%s' % f_score)

        print('Mean Precision:\t%s' % average_precision)
        print('Mean Recall:\t%s' % average_recall)
        print('Mean F_score:\t%s' % average_f_score)


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

    # Block 1 (7 x 7 x 512).
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(512, (3, 3), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 2 (7 x 7 x 512).
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(512, (3, 3), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 3 (5 x 5 x 512).
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    # Block 4 (1 x 1 x 17).
    net = Conv2D(512, (1, 1), padding='same')(net)
    net = Conv2D(256, (3, 3), strides=(2, 2), padding='valid')(net)

    # Block 5 predictions.
    net = Flatten()(net)
    net = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(net)
    net = Dropout(0.5)(net)
    net = Dense(num_classes, kernel_regularizer=l2(1e-4))(net)
    net = Lambda(K.sigmoid)(net)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[net])
    model.compile(Adam(1e-3), 'categorical_crossentropy')

    return model


def load_image(image_path, augment=True, fileformat='%s/%s.jpg'):
    filename = fileformat % (config.image_dir, image_path)

    # Make things better.
    image = Image.open(filename)
    image = image.resize(config.image_shape[:-1])

    if augment:
        i = np.random.randint(len(AUGMENT) + 1)

        if i < len(AUGMENT):
            image = image.transpose(AUGMENT[i])

    image = np.float32(image)
    image = image[:,:,:3]

    if augment:
        image += 20.0 * np.random.randn(*image.shape)

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

        # Batchwise preprocess images.
        x = vgg16.preprocess_input(x)

        yield x, y


def get_datagen(train_path, batch_size, validation_split=0.15):
    labels = helpers.get_labels(train_path)
    np.random.shuffle(labels)

    index = round(validation_split * len(labels))
    datagen = batch_generator(labels[index:], batch_size)
    valid_datagen = batch_generator(labels[:index], batch_size)

    return datagen, valid_datagen


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=3)

    model = build()
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)

    callbacks = list()
    callbacks.append(ModelCheckpoint(config.model_path))
    callbacks.append(ReduceLROnPlateau(factor=0.1, patience=0, verbose=1, epsilon=0.1))
    callbacks.append(MetricCallback(model, valid_datagen, 100))

    model.fit_generator(datagen, steps_per_epoch=1000, epochs=10,
                        validation_data=valid_datagen, validation_steps=100,
                        callbacks=callbacks)
