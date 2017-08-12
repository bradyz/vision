import os
import tensorflow as tf

import multithread_generator


class MiscellaneousGenerator(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self._dtypes = [tf.float32, tf.float32]
        self._shapes = [[32, 32, 3], [10]]

    def get_dtypes(self):
        return self._dtypes

    def get_shapes(self):
        return self._shapes


def get_runner(generator, batch_size):
    runner = multithread_generator.AsyncRunner(generator)
    inputs_op = runner.get_inputs(batch_size)

    return runner, inputs_op


def main(datagen_train, datagen_valid, batch_size, n_iterations):
    is_training_op = tf.placeholder(tf.bool, shape=[])

    runner_train, inputs_train_op = get_runner(datagen_train, batch_size)
    runner_valid, inputs_valid_op = get_runner(datagen_valid, batch_size)

    images_op, labels_op = tf.cond(is_training_op,
                                   inputs_train_op, inputs_valid_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()

        threads = list()
        threads += runner_train.create_threads(sess, coord)
        threads += runner_valid.create_threads(sess, coord)

        try:
            for _ in range(n_iterations):
                if coord.should_stop():
                    break

                sess.run(images_op, {is_training_op: True})
        except Exception as e:
            print(e)
            coord.request_stop(e)
        finally:
            coord.join(threads)


if __name__ == '__main__':
    main()
