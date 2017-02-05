import time
from datetime import timedelta

import tensorflow as tf
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data


IMAGE_SIZE = 28
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_CHANNELS = 1
LABEL_CLASSES = 10
BATCH_SIZE = 20

total_iterations = 0

def optimize(num_iterations):
    global total_iterations

    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        acc = session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations

    end_time = time.time()
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(time_dif))))


if __name__ == "__main__":
    data = input_data.read_data_sets('data/MNIST/', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_FLAT], name='x')
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    x_pretty = pt.wrap(x_image)

    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
                conv2d(kernel=5, depth=16, name='conv1').\
                max_pool(kernel=2, stride=2).\
                conv2d(kernel=5, depth=36, name='conv2').\
                max_pool(kernel=2, stride=2).\
                flatten().\
                fully_connected(size=128, name='fc1').\
                softmax_classifier(num_classes=LABEL_CLASSES, labels=y_true)

    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    optimize(10000)
