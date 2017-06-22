from PIL import Image
import numpy as np

import keras.backend as K

from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.applications import vgg16
from keras.preprocessing.image import random_rotation

import config
import helpers


AUGMENT = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
           Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
           Image.TRANSPOSE]


def f_measure(p, r, beta_sq=2.0):
    return (1.0 + beta_sq) * (p * r) / (beta_sq * p + r)


def print_metrics(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = f_measure(precision, recall)

    f_score[np.logical_and(~np.isfinite(precision), np.isfinite(recall))] = 0.0
    f_score[~np.isfinite(f_score)] = 0.0
    f_score[~np.isfinite(recall)] = np.nan

    average_f_score = np.mean(f_score[np.isfinite(f_score)])

    print()
    print('Precision:\n%s' % precision)
    print('Recall:\n%s' % recall)
    print('F_score:\n%s' % f_score)
    print('F_score:\t%s' % average_f_score)


class MetricCallback(Callback):
    def __init__(self, model, valid_datagen, steps):
        self.model = model
        self.valid_datagen = valid_datagen
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):
        TP = np.zeros(config.num_classes, dtype=np.int32)
        FP = np.zeros(config.num_classes, dtype=np.int32)
        FN = np.zeros(config.num_classes, dtype=np.int32)

        for i, x_y in enumerate(self.valid_datagen):
            if i == self.steps:
                break

            y_true = np.round(x_y[1]).astype('bool')
            y_pred = np.round(self.model.predict_on_batch(x_y[0])).astype('bool')

            TP += np.sum(np.logical_and(y_pred == True, y_true == True), axis=0)
            FP += np.sum(np.logical_and(y_pred == True, y_true == False), axis=0)
            FN += np.sum(np.logical_and(y_pred == False, y_true == True), axis=0)

        print_metrics(TP, FP, FN)


def weighted_loss(weighted=True, alpha=1.0):
    def loss(y_true, y_pred):
        left = y_true * K.log(y_pred + K.epsilon())
        right = (1.0 - y_true) * K.log(1.0 - y_pred + K.epsilon())

        log_loss = K.mean(left, axis=0) + weights * K.mean(right, axis=0)

        return K.mean(-1.0 * log_loss)

    if weighted:
        weights = np.log(get_class_weight(helpers.get_labels(config.train_path)))
    else:
        weights = np.ones(config.num_classes)

    weights = alpha * weights

    return loss


def downsample(tensor, desired_size=7):
    result = tensor

    if result.shape.as_list()[1] > desired_size:
        result = Conv2D(32, (1, 1), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

    while result.shape.as_list()[1] > desired_size:
        result = Conv2D(32, (3, 3), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)
        result = MaxPooling2D()(result)

    return result


def build(weights_path=''):
    num_classes = config.num_classes
    input_tensor = Input(shape=(224, 224, 3))

    vgg = vgg16.VGG16(input_tensor=input_tensor,
                      include_top=False,
                      classes=num_classes)

    # Freeze for training.
    for layer in vgg.layers:
        layer.frozen = True

    # Gather feature layers.
    features = [layer.output for layer in vgg.layers if layer.name in config.FEATURES]
    features += [input_tensor]

    # Use a bunch of downsampled feature maps.
    net = Concatenate()(list(map(downsample, features)))

    # Block 2.
    block2_a = net
    block2_a = Conv2D(64, (1, 1), padding='same')(block2_a)
    block2_a = BatchNormalization()(block2_a)
    block2_a = Lambda(K.relu)(block2_a)

    block2_a = Conv2D(64, (3, 3), padding='same')(block2_a)
    block2_a = BatchNormalization()(block2_a)
    block2_a = Lambda(K.relu)(block2_a)

    block2_a = Conv2D(512, (1, 1), padding='same')(block2_a)
    block2_a = BatchNormalization()(block2_a)
    block2_a = Lambda(K.relu)(block2_a)

    # Block 3.
    block3_a = block2_a

    block3_a = Conv2D(128, (1, 1), padding='same')(block3_a)
    block3_a = BatchNormalization()(block3_a)
    block3_a = Lambda(K.relu)(block3_a)

    block3_a = Conv2D(128, (3, 3), padding='same')(block3_a)
    block3_a = BatchNormalization()(block3_a)
    block3_a = Lambda(K.relu)(block3_a)

    block3_a = MaxPooling2D()(block3_a)

    block3_a = Conv2D(256, (1, 1), padding='same')(block3_a)
    block3_a = BatchNormalization()(block3_a)
    block3_a = Lambda(K.relu)(block3_a)

    block3_a = Conv2D(256, (3, 3), padding='same')(block3_a)
    block3_a = BatchNormalization()(block3_a)
    block3_a = Lambda(K.relu)(block3_a)

    block3_a = MaxPooling2D()(block3_a)

    # Block 4.
    block4_a = block3_a
    block4_a = Flatten()(block4_a)

    block4_a = Dropout(0.5)(block4_a)
    block4_a = Dense(512,
                     kernel_regularizer=l2(5e-3),
                     bias_regularizer=l2(5e-3))(block4_a)
    block4_a = BatchNormalization()(block4_a)
    block4_a = Lambda(K.relu)(block4_a)

    # Block 5.
    block5_a = block4_a
    block5_a = Dropout(0.5)(block5_a)
    block5_a = Dense(num_classes,
                     kernel_regularizer=l2(5e-3),
                     bias_regularizer=l2(5e-3),
                     activation='sigmoid')(block5_a)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[block5_a])
    model.compile(Adam(lr=1e-5), weighted_loss(), metrics=['binary_accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_image(image_path, augment=True, fileformat='%s/%s.jpg'):
    filename = fileformat % (config.image_dir, image_path)

    # Make things better.
    image = Image.open(filename).convert('RGB')
    image = image.resize(config.image_shape[:-1])

    if augment:
        i = np.random.randint(len(AUGMENT) + 1)

        if i < len(AUGMENT):
            image = image.transpose(AUGMENT[i])

    image = np.uint8(image)

    if augment:
        image = random_rotation(image, 180, row_axis=0, col_axis=1, channel_axis=2)
        image += np.uint8(5.0 * np.random.randn(*image.shape))
        image = np.clip(image, 0, 255)

    return image


def batch_generator(labels, batch_size, weighted=False):
    x = np.zeros([batch_size] + config.image_shape)
    y = np.zeros([batch_size] + [config.num_classes])

    if weighted:
        # Sample inversely proportional to data frequency.
        # class_weight = get_class_weight(labels)
        class_weight = np.ones(config.num_classes)
        class_weight = class_weight / np.sum(class_weight) * batch_size
        class_weight = np.int32(class_weight)

        # Rounding errors, balance out the rest.
        while np.sum(class_weight) < batch_size:
            min_i = 0

            # Look for least represented.
            for i in range(class_weight.shape[0]):
                if class_weight[i] < class_weight[min_i]:
                    min_i = i

            class_weight[min_i] += 1

        # Make sure we have enough for a batch.
        assert np.sum(class_weight) == batch_size

        print(class_weight)

    while True:
        index = 0

        if weighted:
            np.random.shuffle(class_weight)

            # Times each class has been seen.
            class_counts = np.zeros(class_weight.shape[0])
            class_index = 0

        # Load up a batch.
        for i in range(batch_size):
            index = np.random.randint(len(labels))

            if weighted:
                # Keep looking for the right class.
                while labels[index][1][class_index] < 1.0:
                    index = np.random.randint(len(labels))

                class_counts[class_index] += 1

                # Sufficiently found enough of the class.
                if class_counts[class_index] >= class_weight[class_index]:
                    class_index = (class_index + 1) % class_weight.shape[0]

            x[i] = load_image(labels[index][0])
            y[i] = labels[index][1]

        # Batchwise preprocess images.
        x = vgg16.preprocess_input(x)

        yield x, y


def get_datagen(train_path, batch_size, validation_split=0.2):
    labels = helpers.get_labels(train_path)
    np.random.shuffle(labels)

    index = round(validation_split * len(labels))
    datagen = batch_generator(labels[index:], batch_size, True)
    valid_datagen = batch_generator(labels[:index], batch_size)

    return datagen, valid_datagen


def get_class_weight(labels):
    counts = np.zeros(config.num_classes)

    for _, y in labels:
        counts += y

    return 1.0 / (counts / np.sum(counts))


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=2)

    model = build()
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)
    class_weight = np.log(get_class_weight(helpers.get_labels(config.train_path)))

    train_steps = 500
    validation_steps = 50
    num_epochs = 100

    callbacks = list()
    callbacks.append(ModelCheckpoint(config.model_path, verbose=1))
    callbacks.append(ReduceLROnPlateau(factor=0.5, patience=1, verbose=1, epsilon=0.01))
    callbacks.append(MetricCallback(model, valid_datagen, validation_steps))
    callbacks.append(TensorBoard(write_graph=False))

    model.fit_generator(datagen,
                        steps_per_epoch=train_steps, epochs=num_epochs,
                        validation_data=valid_datagen,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        class_weight=class_weight)
