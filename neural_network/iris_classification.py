import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris


def fully_connected(x, output_size, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable('W', [x.shape[1], output_size], tf.float32,
                initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', (output_size), tf.float32,
                initializer=tf.constant_initializer(0.0))

    return tf.nn.relu(tf.matmul(x, W) + b)


def feed_forward(inputs):
    hidden1 = fully_connected(inputs, 10, 'layer_1')
    hidden2 = fully_connected(hidden1, 10, 'layer_2')
    return fully_connected(hidden2, num_classes, 'layer_3')


def cross_entropy_loss(logits, one_hot):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
            labels=one_hot, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def generate(inputs, outputs, batch_size=16):
    num_samples = inputs.shape[0]

    while True:
        indices = np.random.permutation(np.arange(num_samples))[:batch_size]

        x = inputs.take(indices, axis=0)
        y = outputs.take(indices)

        yield x, y


if __name__ == '__main__':
    data = load_iris()
    input_size = len(data.feature_names)
    num_classes = data.target_names.shape[0]
    batch_size = 16

    with tf.Graph().as_default():
        sess = tf.Session()

        # Inputs to the graph - x, y pairs for training.
        input_placeholder = tf.placeholder(tf.float32,
                shape=(batch_size, input_size))
        label_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        label_one_hot = tf.one_hot(label_placeholder, num_classes)

        # Unnormalized class score predictions.
        logits = feed_forward(input_placeholder)

        # Human interpretable classification accuracy.
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Cross entropy loss function to minimize.
        loss = cross_entropy_loss(logits, label_one_hot)
        tf.summary.scalar('loss', loss)

        # Combine statistics.
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('log_dir', sess.graph)

        # Minimize loss function.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(1e-4)
        train = optimizer.minimize(loss, global_step=global_step)

        # Used for loading or saving sessions.
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'log_dir/model')

        for input_feed, label_feed in generate(data.data, data.target):
            # The data to be pumped into the graph.
            feed = {input_placeholder: input_feed, label_placeholder: label_feed}

            # Actually run three operations.
            _, summary, step = sess.run([train, merged, global_step],
                    feed_dict=feed)

            # Add to tensorboard.
            summary_writer.add_summary(summary, step)

            # Checkpoint.
            if step % 1000 == 0:
                saver.save(sess, 'log_dir/model')
                print(step)
