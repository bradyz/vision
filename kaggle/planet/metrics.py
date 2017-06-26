import numpy as np

from keras.callbacks import Callback

import config


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
