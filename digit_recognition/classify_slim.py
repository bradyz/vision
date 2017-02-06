import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 32
IMG_SIZE = 28
NUM_CLASSES = 10

LEARNING_RATE = 0.001
LEARNING_STEPS = 1000


def cat_cnn(inputs):
    net = slim.conv2d(inputs, 16, [5, 5])
    net = slim.max_pool2d(net, [2, 2])
    net = slim.conv2d(net, 16, [3, 3])
    net = slim.max_pool2d(net, [2, 2])
    net = slim.flatten(net)
    net = slim.fully_connected(net, 128)
    net = slim.fully_connected(net, NUM_CLASSES, activation_fn=None)
    return net


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x, y_ = data.train.next_batch(BATCH_SIZE)
    x = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 1])

    x, y_ = tf.train.batch([x[0], y_[0]], batch_size=BATCH_SIZE)
    y = cat_cnn(x)

    slim.losses.softmax_cross_entropy(y, y_)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    accuracy = tf.contrib.metrics.accuracy(tf.argmax(y, 1), tf.argmax(y_, 1))
    tf.summary.scalar('accuracy', accuracy)

    train = slim.learning.create_train_op(
        total_loss,
        tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE))

    slim.learning.train(
        train,
        "train",
        number_of_steps=LEARNING_STEPS,
        save_summaries_secs=30,
        save_interval_secs=30)
