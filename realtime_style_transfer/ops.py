import tensorflow as tf


def inspiration_layer(x, s_gram, post=''):
    n, h, w, c = x.shape.as_list()
    std = 1.0 / tf.sqrt(1.0 * c / 2.0)

    with tf.variable_scope('inspiration%s' % post):
        x = tf.reshape(x, [n, h * w, c])

        W = tf.get_variable('W', [c, c],
                initializer=tf.truncated_normal_initializer(stddev=std))
        WG = tf.map_fn(lambda gram: tf.matmul(W, gram), s_gram)

        x = tf.matmul(x, WG)
        x = tf.reshape(x, [n, h, w, c])

    return x


def instance_normalization(x, post='', eps=1e-9):
    # Want to normalize across the channels.
    channels = [x.shape.as_list()[-1]]

    with tf.variable_scope('in%s' % post):
        x_mean, x_var = tf.nn.moments(x, axes=(1, 2), keep_dims=True)
        x_std = tf.sqrt(x_var + eps)

        gamma = tf.get_variable('gamma', channels,
                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', channels,
                initializer=tf.constant_initializer(0.0))

        x = (x - x_mean) / x_std
        x = gamma * x + beta

    return x


def reflect_pad(x, k):
    return tf.pad(x, [[0, 0], [k, k], [k, k], [0, 0]], mode='REFLECT')


def residual_block(x, scope='residual', repeat=2):
    k = x.shape.as_list()[-1]

    for i in range(repeat):
        with tf.variable_scope('%s/%s' % (scope, i)):
            x_id = x

            x = conv1x1(x, k // 2, post='_1')
            x = relu(x)

            x = conv3x3(x, k // 2)
            x = relu(x)

            x = conv1x1(x, k, post='_2')
            x = relu(x)

            x = x_id + x

    return x


def down_block(x, k, s_gram_op, scope=None):
    with tf.variable_scope(scope):
        x = conv3x3(x, k, 2)
        x = relu(x)

        x = inspiration_layer(x, s_gram_op)

    return x


def up_block(x, k, scope=None):
    with tf.variable_scope(scope):
        x = conv1x1(x, k)
        x = bilinear_up(x)

        x = conv3x3(x, k)
        x = relu(x)

    return x


def conv9x9(x, c_out, normalize=True, post=''):
    return conv(x, 9, c_out, normalize=normalize, post='_9x9' + post)


def conv3x3(x, c_out, stride=1, post=''):
    return conv(x, 3, c_out, stride, post='_3x3' + post)


def conv1x1(x, c_out, stride=1, post=''):
    return conv(x, 1, c_out, stride, post='_1x1' + post)


def relu(x):
    return tf.nn.relu(x)


def bilinear_up(x):
    return bilinear(x, 2)


def bilinear_down(x):
    return bilinear(x, 0.5)


def bilinear(x, scale):
    _, h, w, _ = x.shape.as_list()
    h_new, w_new = int(scale * h), int(scale * w)

    return tf.image.resize_nearest_neighbor(x, [h_new, w_new])


def conv(x, k, c_out, stride=1, normalize=True, post=''):
    c_in = x.get_shape().as_list()[-1]

    # 3x3 filter => 1 pad.
    # 5x5 filter => 2 pad.
    # 7x7 filter => 3 pad.
    if k > 1:
        x = reflect_pad(x, k // 2)

    with tf.variable_scope('conv%s' % post):
        W = tf.get_variable('W', [k, k, c_in, c_out],
                initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
        x = x + b

        if normalize:
            x = instance_normalization(x)

    return x
