import cv2
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

import keras.backend as K

from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dropout, Flatten
from keras.layers import Input, Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.applications import resnet50
from keras.preprocessing.image import random_rotation

import config
import helpers
import metrics


def weighted_loss(weighted=True, alpha=4.0):
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
        result = Conv2D(128, (1, 1), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

        result = Conv2D(128, (3, 3), padding='same')(result)
        result = BatchNormalization()(result)
        result = Lambda(K.relu)(result)

    while result.shape.as_list()[1] > desired_size:
        result = MaxPooling2D()(result)

    return result


def build(weights_path=''):
    num_classes = config.num_classes
    input_tensor = Input(shape=(224, 224, 3))

    resnet = resnet50.ResNet50(input_tensor=input_tensor, include_top=False)

    for layer in resnet.layers:
        layer.frozen = True

    # Gather feature layers.
    features = [layer.output for layer in resnet.layers if layer.name in config.FEATURES]

    # Use a bunch of downsampled feature maps.
    net = Concatenate()(list(map(downsample, features)))

    print(net)

    block1 = net

    block1 = Conv2D(256, (1, 1), padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Lambda(K.relu)(block1)
    block1 = Conv2D(256, (3, 3), padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Lambda(K.relu)(block1)

    block2 = block1
    block2 = Conv2D(256, (1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Lambda(K.relu)(block2)
    block2 = Conv2D(256, (3, 3), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Lambda(K.relu)(block2)

    block2 = MaxPooling2D()(block2)

    block3 = block2
    block3 = Conv2D(512, (1, 1), padding='same')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Lambda(K.relu)(block3)
    block3 = Conv2D(512, (3, 3), padding='same')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Lambda(K.relu)(block3)

    block3 = AveragePooling2D()(block3)

    block4 = block3
    block4 = Dropout(0.5)(block4)
    block4 = Conv2D(1024, (1, 1), padding='same',
                    kernel_regularizer=l2(5e-5))(block4)
    block4 = BatchNormalization()(block4)
    block4 = Lambda(K.relu)(block4)

    block5 = block4
    block5 = Dropout(0.5)(block5)
    block5 = Conv2D(1024, (1, 1), padding='same',
                    kernel_regularizer=l2(5e-5))(block5)
    block5 = BatchNormalization()(block5)
    block5 = Lambda(K.relu)(block5)

    # Block 4.
    block6 = block5
    block6 = Dropout(0.5)(block6)
    block6 = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid',
                    kernel_regularizer=l2(5e-5))(block6)
    block6 = Flatten()(block6)

    # Pack it up in a model.
    model = Model(inputs=[input_tensor], outputs=[block6])
    model.compile(Adam(lr=1e-5), weighted_loss(), metrics=['binary_accuracy'])

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
        image += np.random.randn(*image.shape)
        image = np.clip(image, 0, 255)

    return image


def batch_generator(labels, batch_size, weighted=False):
    x = np.zeros([batch_size] + config.image_shape)
    y = np.zeros([batch_size] + [config.num_classes])

    if weighted:
        # Sample inversely proportional to data frequency.
        # class_weight = np.ones(config.num_classes)
        # class_weight = np.log(get_class_weight(labels))
        class_weight = get_class_weight(labels)

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
            # np.random.shuffle(class_weight)

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
        x = resnet50.preprocess_input(x)

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


def find_best_threshold(y_true, y_pred, step=0.01):
    best_threshold = np.full(config.num_classes, -1.0)
    best_f_score = np.full(config.num_classes, -1.0)

    # Brute force for each class threshold.
    for threshold in range(step, 1.0, step):
        y_true_threshold = y_true >= threshold
        y_pred_threshold = y_pred >= threshold

        f_score = precision_recall_fscore_support(y_true_threshold,
                                                  y_pred_threshold, 2.0)

        for i in range(config.num_classes):
            if f_score[i] > best_f_score[i]:
                best_f_score[i] = f_score[i]
                best_threshold[i] = threshold

    return best_threshold


def get_predictions(model, datagen, num_samples=50):
    y_true = np.zeros(num_samples * config.batch_size, config.num_classes)
    y_pred = np.zeros(num_samples * config.batch_size, config.num_classes)

    for i in range(num_samples):
        x, y = next(datagen)

        # Get the slice to fill up.
        i_start = i * config.batch_size
        i_end = i * (config.batch_size + 1)

        y_true[i_start:i_end] = y
        y_pred[i_start:i_end] = model.predict_on_batch(x)

    return y_true, y_pred


def test(model, datagen, valid_datagen):
    # Cache the predictions.
    y_true, y_pred = get_predictions(model, datagen)
    y_true_valid, y_pred_valid = get_predictions(model, valid_datagen)

    threshold, f_score = find_best_threshold(y_true, y_pred)

    # Use this threshold for the validation set.
    threshold_valid, f_score_valid_best = find_best_threshold(y_true_valid,
                                                              y_pred_valid)

    y_true_valid = y_true_valid > threshold
    y_pred_valid = y_pred_valid > threshold

    f_score_valid = precision_recall_fscore_support(y_true_valid,
                                                    y_pred_valid, 2.0)

    print(threshold)
    print(f_score)
    print(np.mean(f_score))
    print(f_score_valid_best)
    print(f_score_valid)


def train(model, datagen, valid_datagen):
    class_weight = np.log(get_class_weight(helpers.get_labels(config.train_path)))

    train_steps = 500
    validation_steps = 50
    num_epochs = 100

    callbacks = list()
    callbacks.append(ModelCheckpoint(config.model_path, verbose=1))
    callbacks.append(ReduceLROnPlateau(factor=0.5, patience=3,
                                       verbose=1, epsilon=0.01))
    callbacks.append(metrics.MetricCallback(model, valid_datagen,
                                            validation_steps))
    callbacks.append(TensorBoard(write_graph=False))

    model.fit_generator(datagen,
                        steps_per_epoch=train_steps, epochs=num_epochs,
                        validation_data=valid_datagen,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        class_weight=class_weight)


def main():
    np.random.seed(0)
    np.set_printoptions(precision=2)

    model = build(config.model_path)
    datagen, valid_datagen = get_datagen(config.train_path, config.batch_size)

    # Find the optimal thresholds.
    test(model, datagen, valid_datagen)


if __name__ == '__main__':
    main()
