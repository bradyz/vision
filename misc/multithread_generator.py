import tensorflow as tf


class AsyncRunner(object):
    def __init__(self, generator, cap=1024, min_after=64):
        dtypes = generator.get_dtypes()
        shapes = generator.get_shapes()

        self.queue = tf.RandomShuffleQueue(cap, min_after, dtypes, shapes)

        get_data_op = tf.py_func(generator.next, [], dtypes)
        enqueue_op = self.queue.enqueue(get_data_op)

        self.runner = tf.train.QueueRunner(self.queue, [enqueue_op])

    def get_inputs(self, batch_size):
        return self.queue.dequeue_many(batch_size)

    def create_threads(self, sess, coord):
        return self.runner.create_threads(sess, coord, daemon=True, start=True)
