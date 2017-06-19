from PIL import Image
import numpy as np

from sklearn.metrics import fbeta_score

import keras.backend as K

from keras.regularizers import l2
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.applications import vgg16

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

    TP_sum = TP.sum()
    FP_sum = FP.sum()
    FN_sum = FN.sum()

    average_precision = TP_sum / (TP_sum + FP_sum)
    average_recall = TP_sum / (TP_sum + FN_sum)
    average_f_score = np.mean(f_score[np.isfinite(f_score)])

    print()
    print('True Positives:\n%s' % TP)
    print('False Positives:\n%s' % FP)
    print('False Negatives:\n%s' % FN)

    print('Precision:\n%s' % precision)
    print('Recall:\n%s' % recall)
    print('F_score:\n%s' % f_score)

    print('Precision:\t%s' % average_precision)
    print('Recall:\t\t%s' % average_recall)
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

        fbeta = np.zeros(config.num_classes)
        mean_fbeta = 0
        total = 0

        for i, x_y in enumerate(self.valid_datagen):
            if i == self.steps:
                break

            y_true = np.round(x_y[1]).astype('bool')
            y_pred = np.round(self.model.predict_on_batch(x_y[0])).astype('bool')

            TP += np.sum(np.logical_and(y_pred == True, y_true == True), axis=0)
            FP += np.sum(np.logical_and(y_pred == True, y_true == False), axis=0)
            FN += np.sum(np.logical_and(y_pred == False, y_true == True), axis=0)

            total += 1
            fbeta = (total - 1.0) / total * fbeta + \
                    1.0 / total * fbeta_score(y_true, y_pred, beta=2, average=None)
            mean_fbeta = (total - 1.0) / total * mean_fbeta + \
                    1.0 / total * fbeta_score(y_true, y_pred, beta=2, average='samples')

        print_metrics(TP, FP, FN)
        print(fbeta, mean_fbeta)


def downsample(tensor, desired_size=7):
    result = tensor.output

    while result.shape.as_list()[1] > desired_size:
        result = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

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
    features = [layer for layer in vgg.layers if layer.name in config.FEATURES]

    # Use a bunch of downsampled feature maps.
    net = Concatenate()(list(map(downsample, features)))
    net = BatchNormalization()(net)

    # Block 1.
    block1_a = Conv2D(256, (1, 1), padding='same')(net)

    block1_b = Conv2D(64, (1, 1), padding='same')(block1_a)
    block1_b = BatchNormalization()(block1_b)
    block1_b = Lambda(K.relu)(block1_b)

    block1_b = Conv2D(64, (3, 3), padding='same')(block1_b)
    block1_b = BatchNormalization()(block1_b)
    block1_b = Lambda(K.relu)(block1_b)

    block1_b = Conv2D(256, (1, 1), padding='same')(block1_b)
    block1_b = Lambda(lambda x: 0.2 * x)(block1_b)

    block1_c = add([block1_a, block1_b])
    block1_c = BatchNormalization()(block1_c)
    block1_c = Lambda(K.relu)(block1_c)

    # Block 2.
    block2_a = Conv2D(64, (1, 1), padding='same')(block1_c)
    block2_a = BatchNormalization()(block2_a)
    block2_a = Lambda(K.relu)(block2_a)

    block2_a = Conv2D(64, (3, 3), padding='same')(block2_a)
    block2_a = BatchNormalization()(block2_a)
    block2_a = Lambda(K.relu)(block2_a)

    block2_a = Conv2D(256, (1, 1), padding='same')(block2_a)
    block2_a = Lambda(lambda x: 0.2 * x)(block2_a)

    block2_b = add([block1_c, block2_a])
    block2_b = BatchNormalization()(block2_b)
    block2_b = Lambda(K.relu)(block2_b)
    block2_b = AveragePooling2D((5, 5), padding='valid')(block2_b)

    # Block 3.
    block3_a = Dropout(0.5)(block2_b)
    block3_a = Conv2D(256, (1, 1), padding='same',
                      kernel_regularizer=l2(5e-5))(block3_a)
    block3_a = BatchNormalization()(block3_a)
    block3_a = Lambda(K.relu)(block3_a)

    # Block 4.
    block4_a = Flatten()(block3_a)
    block4_a = Dropout(0.5)(block4_a)
    block4_a = Dense(num_classes, activation='sigmoid',
                     kernel_regularizer=l2(5e-5))(block4_a)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[block4_a])
    model.compile(Adam(lr=1e-4), 'binary_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

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
        image += 10.0 * np.random.randn(*image.shape)

    return image


def valid_pick(one_hot, seen):
    want = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

    for i in want:
        if one_hot[i] > 0.0 and (i not in seen or len(seen) == len(want)):
            seen.add(i)
            return True

    return False


def batch_generator(labels, batch_size):
    x = np.zeros([batch_size] + config.image_shape)
    y = np.zeros([batch_size] + [config.num_classes])

    index = 0

    while True:
        seen = set()

        # Load up a batch.
        for i in range(batch_size):
            index = np.random.randint(len(labels))
            # index = (index + 1) % len(labels)

            # Force less common labels.
            while not valid_pick(labels[index][1], seen):
                # index = (index + 1) % len(labels)
                index = np.random.randint(len(labels))

            x[i] = load_image(labels[index][0])
            y[i] = labels[index][1]

        # Batchwise preprocess images.
        x = vgg16.preprocess_input(x)

        yield x, y


def get_datagen(train_path, batch_size, validation_split=0.2):
    labels = helpers.get_labels(train_path)
    np.random.shuffle(labels)

    index = round(validation_split * len(labels))
    datagen = batch_generator(labels[index:], batch_size)
    valid_datagen = batch_generator(labels[:index], batch_size)

    return datagen, valid_datagen


def get_class_weight(train_path):
    counts = np.zeros(config.num_classes)

    for _, y in helpers.get_labels(train_path):
        counts += y

    return 1.0 / (counts / np.sum(counts))


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=2)

    model = build()
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)
    class_weight = np.log(get_class_weight(config.train_path))

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
