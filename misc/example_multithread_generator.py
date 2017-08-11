import os
import tensorflow as tf

import multithread_generator


class MiscellaneousGenerator(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self.image_shape = [tf.float32, [32, 32, 3]]
        self.label_shape = [tf.float32, [10]]

    def get_shape(self):
        return [self.image_shape, self.label_shape]

def main():
    pass


if __name__ == '__main__':
    main()
