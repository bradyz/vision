import csv

import numpy as np


def to_one_hot(class_name, class_labels):
    x = np.zeros(len(class_labels), dtype=np.float32)
    x[class_labels[class_name]] = 1.0

    return x


def get_classes(filename):
    class_names = set()

    with open(filename, 'r') as csvfile:
        for _, class_labels in csv.reader(csvfile):
            for class_name in class_labels.split(' '):
                class_names.add(class_name)

    return {name: i for i, name in enumerate(sorted(class_names))}


def get_labels(filename):
    class_names = get_classes(filename)
    n = len(class_names)

    result = dict()

    with open(filename, 'r') as csvfile:
        for path, class_labels in csv.reader(csvfile):
            x = np.zeros(n, dtype=np.float32)

            for class_name in class_labels.split(' '):
                x += to_one_hot(class_name, class_names)

            result[path] = x

    return result
