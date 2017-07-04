import cv2
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf

import keras.backend as K

from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications import resnet50
from keras.preprocessing.image import random_rotation

import config
import helpers
import metrics


def weighted_loss(weighted=False, alpha=2.0, beta=1.0):
    def loss(y_true, y_pred):
        left = y_true * K.log(y_pred + K.epsilon())
        right = (1.0 - y_true) * K.log(1.0 - y_pred + K.epsilon())

        if weighted:
            weights = np.log(get_class_weight(helpers.get_labels(config.train_path)))
            weights = weights / np.max(weights)

            print(weights)
        else:
            weights = np.ones(config.num_classes)

        # Alpha will penalize missing a positive more.
        log_loss = alpha * weights * K.mean(left, axis=0) + beta * K.mean(right, axis=0)

        return -1.0 * K.mean(log_loss)

    return loss


def downsample(tensor, desired_size=7):
    result = tensor

    if result.shape.as_list()[1] > desired_size:
        result = Conv2D(128, (1, 1), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

        result = Conv2D(128, (3, 3), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

    while result.shape.as_list()[1] > desired_size:
        result = AveragePooling2D()(result)

    return result


def resize(tensor, h, w):
    return Lambda(lambda x: tf.image.resize_images(x, (h, w)))(tensor)


def conv_block(net, k, decay=5e-5):
    net = Conv2D(k // 2, (1, 1), padding='same', kernel_regularizer=l2(decay))(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    net = Conv2D(k * 2, (3, 3), padding='same', kernel_regularizer=l2(decay))(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    return net


def dense_block(net, k, n=3):
    for _ in range(n):
        net = Concatenate()([net, conv_block(net, k)])

    return net


def transition_block(net, k=32):
    # net = BatchNormalization()(net)
    # net = AveragePooling2D((3, 3), strides=(2, 2), padding='SAME')(net)
    # first = Dropout(0.25)(first)
    #
    # second = MaxPooling2D((3, 3), strides=(2, 2), padding='SAME')(net)
    # second = Dropout(0.25)(second)
    #
    # return Concatenate()([first, second])
    net = Conv2D(k, (3, 3), strides=(2, 2), padding='same')(net)
    net = BatchNormalization()(net)
    net = Lambda(K.relu)(net)

    return net


def build_simple(weights_path=''):
    input_tensor = Input(shape=config.image_shape)

    # 112 x 112.
    block1 = input_tensor
    block1 = conv_block(block1, 64)
    block1 = transition_block(block1, 64)

    block2 = block1
    # block2 = conv_block(block2, 16)
    # block2 = transition_block(block2, 32)

    block3 = block2
    # block3 = conv_block(block3, 32)
    # block3 = transition_block(block3, 64)

    block4 = block3
    block4 = conv_block(block4, 64)
    block4 = transition_block(block4, 64)

    # One activation map for each class.
    block5 = block4
    # block5 = conv_block(block4, 32)
    # block5 = Dropout(0.5)(block5)

    # block5 = conv_block(block5, 32)
    # block5 = Dropout(0.5)(block5)

    block5 = Conv2D(512, (1, 1), padding='same', activation='relu')(block5)
    block5 = GlobalMaxPooling2D()(block5)

    block5 = Dense(128, activation='relu')(block5)
    block5 = Dense(config.num_classes, activation='sigmoid')(block5)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[block5])
    model.compile(Adam(lr=2e-4), weighted_loss(), metrics=['binary_accuracy'])

    print(model.summary())

    if weights_path:
        model.load_weights(weights_path)

    return model


def build(weights_path=''):
    num_classes = config.num_classes
    input_tensor = Input(shape=(224, 224, 3))

    resnet = resnet50.ResNet50(input_tensor=input_tensor, include_top=False)

    for layer in resnet.layers:
        layer.trainable = False

    # Gather feature layers.
    features = [layer.output for layer in resnet.layers if layer.name in config.FEATURES]

    # Use a bunch of downsampled feature maps.
    # net = Concatenate()(list(map(downsample, features)))

    # Use the relu's from each resolution.
    relu_112, relu_55, relu_28, relu_14, relu_7 = features
    relu_56 = resize(relu_55, 56, 56)

    # 112 x 112.
    block1 = relu_112
    block1 = dense_block(block1, 8)

    # 56 x 56.
    block2 = transition_block(block1)
    block2 = Concatenate()([block2, relu_56])
    block2 = dense_block(block2, 8)

    # 28 x 28.
    block3 = transition_block(block2)
    block3 = Concatenate()([block3, relu_28])
    block3 = dense_block(block3, 8)

    # 14 x 14.
    block4 = transition_block(block3)
    block4 = Concatenate()([block4, relu_14])
    block4 = dense_block(block4, 8)

    # 7 x 7.
    block5 = transition_block(block4)
    block5 = Concatenate()([block5, relu_7])
    block5 = dense_block(block5, 8)

    # One activation map for each class.
    block6 = conv_block(block5, 128)
    block6 = Dropout(0.5)(block6)

    block6 = conv_block(block6, 256)
    block6 = Dropout(0.5)(block6)

    block6 = Conv2D(config.num_classes, (1, 1), padding='same',
                    kernel_regularizer=l2(5e-5))(block6)
    block6 = GlobalMaxPooling2D()(block6)
    block6 = BatchNormalization()(block6)
    block6 = Lambda(K.sigmoid)(block6)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[block6])
    model.compile(Adam(lr=2e-5), weighted_loss(), metrics=['binary_accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_image(image_path, augment=True, fileformat='%s/%s.jpg'):
    filename = fileformat % (config.image_dir, image_path)

    # Make things better.
    image = cv2.imread(filename)
    image = cv2.resize(image, (config.image_shape[0], config.image_shape[1]))
    image = image.astype(np.float32)

    if augment:
        image = random_rotation(image, 180, row_axis=0, col_axis=1, channel_axis=2)
        image += 5.0 * np.random.randn(*image.shape)
    #     image = np.clip(image, 0, 255)

    return image


def batch_generator(labels, batch_size, weighted):
    x = np.zeros([batch_size] + config.image_shape)
    y = np.zeros([batch_size] + [config.num_classes])

    if weighted:
        mapping = [list() for _ in range(config.num_classes)]

        for i in range(len(labels)):
            for j in range(config.num_classes):
                if labels[i][1][j] == 1.0:
                    mapping[j].append(i)

        # Sample inversely proportional to data frequency.
        class_weight = np.ones(config.num_classes)
        # class_weight = np.log(get_class_weight(labels))
        # class_weight = get_class_weight(labels)

        class_weight = class_weight / np.sum(class_weight) * batch_size
        class_weight = np.int32(class_weight)
        class_weight += np.ones(config.num_classes, dtype=np.int32)

        # Make sure we have enough for a batch.
        assert np.sum(class_weight) >= batch_size

        print(class_weight)

    while True:
        # Times each class has been seen.
        if weighted:
            class_counts = np.zeros(class_weight.shape[0])
            class_index = 0

        # Load up a batch.
        for i in range(batch_size):
            if weighted:
                class_index = np.random.randint(config.num_classes)

                # Sufficiently found enough of the class.
                while class_counts[class_index] >= class_weight[class_index]:
                    class_index = np.random.randint(config.num_classes)

                index = mapping[class_index][np.random.randint(len(mapping[class_index]))]
                class_counts[class_index] += 1
            else:
                index = np.random.randint(len(labels))

            x[i] = load_image(labels[index][0])
            y[i] = labels[index][1]

        # Batchwise preprocess images.
        # x = resnet50.preprocess_input(x)
        # x = (x - config.image_mean)

        yield x, y


def get_datagen(train_path, batch_size, validation_split=0.2):
    labels = helpers.get_labels(train_path)
    np.random.shuffle(labels)

    index = round(validation_split * len(labels))
    datagen = batch_generator(labels[index:], batch_size, True)
    valid_datagen = batch_generator(labels[:index], batch_size, True)

    return datagen, valid_datagen


def get_class_weight(labels):
    counts = np.zeros(config.num_classes)

    for _, y in labels:
        counts += y

    return 1.0 / (counts / np.sum(counts))


def find_best_threshold(y_true, y_pred, step=0.01):
    best_threshold = np.full(config.num_classes, -1.0)
    best_f_score = np.full(config.num_classes, -1.0)

    # Need to use while because for is integer only.
    threshold = step

    # Brute force for each class threshold.
    while threshold < 1.0:
        y_true_threshold = y_true >= threshold
        y_pred_threshold = y_pred >= threshold

        _, _, f_score, _ = precision_recall_fscore_support(y_true_threshold,
                                                           y_pred_threshold, 2.0)

        for i in range(config.num_classes):
            if f_score[i] > best_f_score[i]:
                best_f_score[i] = f_score[i]
                best_threshold[i] = threshold

        threshold += step

    return best_threshold, best_f_score


def get_predictions(model, datagen, num_samples=100):
    y_true = np.zeros((num_samples * config.batch_size, config.num_classes))
    y_pred = np.zeros((num_samples * config.batch_size, config.num_classes))

    for i in range(num_samples):
        x, y = next(datagen)

        # Get the slice to fill up.
        i_start = i * config.batch_size
        i_end = (i + 1) * config.batch_size

        y_true[i_start:i_end,:] = y
        y_pred[i_start:i_end,:] = model.predict_on_batch(x)

    return y_true, y_pred


def get_f_score_at_threshold(y_true, y_pred, threshold):
    y_true = y_true > threshold
    y_pred = y_pred > threshold

    _, _, f_score, _ = precision_recall_fscore_support(y_true, y_pred, 2.0)

    return f_score


def test(model, datagen, valid_datagen):
    # Cache the predictions.
    y_true, y_pred = get_predictions(model, datagen)
    y_true_valid, y_pred_valid = get_predictions(model, valid_datagen)

    threshold, f_score = find_best_threshold(y_true, y_pred)

    # Use this threshold for the validation set.
    f_score_valid = get_f_score_at_threshold(y_true_valid, y_pred_valid, threshold)

    # Just in case what would the best for validation be.
    threshold_valid_best, f_score_valid_best = find_best_threshold(y_true_valid,
                                                                   y_pred_valid)

    # Test the validation threshold on the train set (just because).
    f_score_tmp = get_f_score_at_threshold(y_true, y_pred, threshold_valid_best)

    # See what the optimal threshold would be.
    print('Optimal Train')
    print('Threshold: %s' % threshold)
    print('F_score: %s\n F_score_mean: %s' % (f_score, np.mean(f_score)))

    print('Optimal Train on Valid')
    print('F_score: %s\n F_score_mean: %s' % (f_score_valid, np.mean(f_score_valid)))

    print('Optimal Valid on Valid')
    print('Threshold: %s' % threshold_valid_best)
    print('F_score: %s\n F_score_mean: %s' % (f_score_valid_best, np.mean(f_score_valid_best)))

    print('Optimal Valid on Train')
    print('F_score: %s\n F_score_mean: %s' % (f_score_tmp, np.mean(f_score_tmp)))


def train(model, datagen, valid_datagen):
    validation_steps = 10
    train_steps = 500
    num_epochs = 100

    callbacks = list()
    callbacks.append(ModelCheckpoint(config.model_path, verbose=1))
    callbacks.append(ReduceLROnPlateau(factor=0.5, patience=3,
                                       verbose=1, epsilon=0.01))
    callbacks.append(metrics.MetricCallback(model, valid_datagen))
    callbacks.append(TensorBoard(write_graph=False))

    model.fit_generator(datagen,
                        steps_per_epoch=train_steps, epochs=num_epochs,
                        validation_data=valid_datagen,
                        validation_steps=validation_steps,
                        callbacks=callbacks)


def get_stats(datagen, num_batches=100):
    mean = np.zeros(3, dtype=np.float32)
    std = np.zeros(3, dtype=np.float32)
    n = 0

    for _ in range(num_batches):
        x, _ = next(datagen)

        n += 1

        mean = (n - 1) / n * mean + 1.0 / n * np.mean(x, axis=(0,1,2))
        std = (n - 1) / n * std + 1.0 / n * np.std(x, axis=(0,1,2))

    return mean, std


def main():
    np.random.seed(0)
    np.set_printoptions(precision=2)

    # model = build()
    model = build_simple()
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)

    # Train the model.
    train(model, datagen, valid_datagen)

    # Find the optimal thresholds.
    test(model, datagen, valid_datagen)


if __name__ == '__main__':
    main()
