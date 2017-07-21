import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.applications import vgg16


def conv(x, k, c_out, stride=1, reg=0.0, post_scope=''):
    c_in = x.get_shape().as_list()[-1]

    with tf.variable_scope('conv%s' % post_scope):
        W = tf.get_variable('W', [k, k, c_in, c_out],
                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                regularizer=tf.contrib.layers.l2_regularizer(reg))
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0))

        Wx = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                          padding='SAME', name='Wx')

        y = Wx + b

    return y


def conv1x1(x, c_out, stride=1):
    return conv(x, 1, c_out, stride, post_scope='_1x1')


def conv3x3(x, c_out, stride=1):
    return conv(x, 3, c_out, stride, post_scope='_3x3')


def relu(x):
    with tf.name_scope('relu'):
        y = tf.nn.relu(x)

    return y


def deconv(x, c_out, k=3, stride=2, reg=1e-4):
    c_in = x.shape.as_list()[-1]
    n, h, w, _ = x.shape.as_list()

    with tf.variable_scope('deconv'):
        W = tf.get_variable('W', [k, k, c_out, c_in],
                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                regularizer=tf.contrib.layers.l2_regularizer(reg))
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0))

        # [16, 1, 1, 32] -> [16, 2, 2, 16].
        Wx = tf.nn.conv2d_transpose(x, W,
                output_shape=[n, h * stride, w * stride, c_out],
                strides=[1, stride, stride, 1])

        y = Wx + b

    return y


def get_trainable_variables():
    weights = list()
    weights_names = set()

    for op in tf.trainable_variables():
        if 'decoder' not in op.name:
            continue
        elif op.name in weights_names:
            continue

        weights.append(op)
        weights_names.add(op.name)

    print('\n'.join(map(str, sorted(weights_names))))

    return weights


def get_activations(x_op, size=(112, 112)):
    with tf.name_scope('vgg'):
        vgg = vgg16.VGG16(include_top=False, input_tensor=x_op)

    activations = [vgg.get_layer('block1_pool').output,
                   vgg.get_layer('block2_pool').output,
                   vgg.get_layer('block3_pool').output,
                   vgg.get_layer('block4_pool').output]

    with tf.name_scope('activations'):
        resized_activations = [tf.image.resize_images(op, size) for op in activations]
        resized_activations_op = tf.concat(resized_activations, axis=3)

    return resized_activations_op


def get_mean_std(x_op):
    # Shape (n, c).
    x_mean_op, x_std_op = tf.nn.moments(x_op, axes=(1, 2))

    # Shape (n, 1, 1, c).
    x_mean_op = tf.expand_dims(tf.expand_dims(x_mean_op, axis=1), axis=1)
    x_std_op = tf.expand_dims(tf.expand_dims(x_std_op, axis=1), axis=1)

    return x_mean_op, x_std_op


def adaIN(c_activations_op, s_activations_op, eps=1e-8):
    """
    Arguments:
        c_activations_op (tensor): of shape (n, h, w, c)
        s_activations_op (tensor): of shape (n, h, w, c)

    Returns:
        (tensor) of shape (n, h, w, c) c aligned with s.
    """
    with tf.name_scope('AdaIN'):
        c_mean_op, c_std_op = get_mean_std(c_activations_op)
        s_mean_op, s_std_op = get_mean_std(s_activations_op)

        # Zero centered.
        c_normalized_op = (c_activations_op - c_mean_op) / (c_std_op + eps)
        c_aligned_op = s_std_op * c_normalized_op + s_mean_op

    return c_aligned_op


def get_decoder_output(t_op):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('block1'):
            net = conv1x1(t_op, 128)
            net = conv3x3(net, 128)
            net = relu(net)

        with tf.variable_scope('block2'):
            net = conv3x3(net, 128)
            net = relu(net)

        with tf.variable_scope('block3'):
            net = deconv(net, 64)
            net = relu(net)

        with tf.variable_scope('block4'):
            net = conv3x3(net, 64)
            net = relu(net)

        with tf.variable_scope('block5'):
            net = conv1x1(net, 3)
            net = tf.nn.sigmoid(net)

    return net


def mean_squared_diff(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_loss_op(t_op, z_activations_op, s_activations_op, alpha=0.1):
    with tf.name_scope('loss'):
        content = tf.reduce_mean(tf.squared_difference(z_activations_op, t_op))

        # 1st order characteristics of feature maps.
        s_activations_mean_op, s_activations_std_op = tf.nn.moments(s_activations_op,
                                                                    axes=(1, 2))
        z_activations_mean_op, z_activations_std_op = tf.nn.moments(z_activations_op,
                                                                    axes=(1, 2))

        style = mean_squared_diff(s_activations_mean_op, z_activations_mean_op) + \
                mean_squared_diff(s_activations_std_op, z_activations_std_op)

        loss = style + alpha * content

    return loss


def main():
    c_op = tf.placeholder(tf.float32, shape=(10, 224, 224, 3))
    s_op = tf.placeholder(tf.float32, shape=(10, 224, 224, 3))

    # Content, style.
    c_activations_op = get_activations(c_op)
    s_activations_op = get_activations(s_op)

    t_op = adaIN(c_activations_op, s_activations_op)

    # Generated.
    z_op = get_decoder_output(t_op)
    z_activations_op = get_activations(z_op)

    # Content, style, variation.
    loss_op = get_loss_op(t_op, z_activations_op, s_activations_op)

    weights_op = get_trainable_variables()

    # All things training related.
    step_op = tf.Variable(0, name='step', trainable=False)
    learn_rate_op = tf.train.exponential_decay(1e-3, step_op,
                                               10000, 0.1, staircase=True)
    optimizer_op = tf.train.AdamOptimizer(learn_rate_op)
    train_op = optimizer_op.minimize(loss_op,
                                     global_step=step_op,
                                     var_list=weights_op)

    c = np.random.rand(10, 224, 224, 3)
    s = np.random.rand(10, 224, 224, 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('.', sess.graph)
        # z, _, _ = sess.run([z_op, train_op, step_op], {c_op: c, s_op: s})



if __name__ == '__main__':
    main()
