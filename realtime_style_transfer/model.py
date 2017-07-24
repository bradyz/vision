import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.applications import vgg16

from PIL import Image

import config


def conv(x, k, c_out, stride=1, post_scope=''):
    c_in = x.get_shape().as_list()[-1]
    shape = [k, k, c_in, c_out]

    with tf.variable_scope('conv%s' % post_scope):
        W = tf.get_variable('W', shape,
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [c_out], initializer=tf.constant_initializer(0.0))

        Wx = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        y = Wx + b

    return y


def conv1x1(x, c_out, post='', stride=1):
    return conv(x, 1, c_out, stride, post_scope='_1x1' + post)


def conv3x3(x, c_out, post='', stride=1):
    return conv(x, 3, c_out, stride, post_scope='_3x3' + post)


def relu(x):
    with tf.name_scope('relu'):
        y = tf.nn.relu(x)

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

    return weights


def get_activations(x_op, op_name, size=(112, 112)):
    with tf.name_scope('vgg_%s' % op_name):
        x_op = x_op[:, :, :, ::-1]
        x_op = x_op - (103.939, 116.779, 123.68)

        vgg = vgg16.VGG16(include_top=False, input_tensor=x_op)

    return vgg.get_layer('block4_conv1').output


def get_mean_std(x_op):
    # Shape (n, c).
    x_mean_op, x_var_op = tf.nn.moments(x_op, axes=(1, 2))
    x_std_op = tf.sqrt(x_var_op)

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


def feature_block(x, k, scope):
    with tf.variable_scope(scope):
        x = conv3x3(x, k, '_1')
        x = relu(x)

        x = conv3x3(x, k, '_2')
        x = relu(x)

    return x


def decode_block(x, k, scope):
    with tf.variable_scope(scope):
        h = conv1x1(x, k, '_1')
        h = relu(h)

        x = conv1x1(h, k // 2, '_2')
        x = relu(x)

        x = conv3x3(x, k // 2)
        x = relu(x)

        x = conv1x1(x, k, '_3')
        x = relu(x)

        x = x + h

        x = bilinear_up(x, scope)
        x = tf.maximum(0.01 * x, x)

    return x


def bilinear_up(x, scope):
    _, h, w, _ = x.shape.as_list()

    with tf.variable_scope(scope):
        x = tf.image.resize_images(x, [2 * h, 2 * w],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return x


def get_decoder_output(t_op):
    with tf.variable_scope('decoder'):
        net = t_op
        net = feature_block(net, 32, 'block1')
        net = feature_block(net, 32, 'block2')
        net = decode_block(net, 64, 'block3')
        net = decode_block(net, 64, 'block4')
        net = decode_block(net, 128, 'block5')

        with tf.variable_scope('block6'):
            net = conv1x1(net, 3)
            net = (tf.nn.tanh(net) + 1.0) * 127.5

    return net


def mean_l2_diff(x, y, eps=1e-8):
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_loss_op(t_op, z_t_op, z_activations_op, s_activations_op):
    with tf.name_scope('loss'):
       # 1st order characteristics of feature maps.
        s_activations_mean_op, s_activations_var_op = tf.nn.moments(s_activations_op,
                                                                    axes=(1, 2))
        z_activations_mean_op, z_activations_var_op = tf.nn.moments(z_activations_op,
                                                                    axes=(1, 2))

        style = mean_l2_diff(s_activations_mean_op, z_activations_mean_op) + \
                mean_l2_diff(s_activations_var_op, z_activations_var_op)

        content = mean_l2_diff(z_t_op, t_op)

        style *= 1e-3 * 0.0
        content *= 1

        loss = style + content

    return loss, style, content


def get_summary(c_op, s_op, z_op, loss_op, style_loss_op, content_loss_op):
    tf.summary.scalar('content_loss', content_loss_op)
    tf.summary.scalar('style_loss', style_loss_op)
    tf.summary.scalar('total_loss', loss_op)

    tf.summary.image('content', tf.cast(c_op, tf.uint8), 10)
    tf.summary.image('style', tf.cast(s_op, tf.uint8), 10)
    tf.summary.image('generated', tf.cast(z_op, tf.uint8), 10)

    return tf.summary.merge_all()


def load_image(path):
    x = np.float32(Image.open(path))

    if x.ndim != 3:
        return None

    h, w, _ = config.input_shape

    if x.shape[0] <= h or x.shape[1] <= w:
        return None

    i = np.random.randint(x.shape[0] - config.input_shape[0])
    j = np.random.randint(x.shape[1] - config.input_shape[1])

    return x[i:i+h,j:j+w]


def get_datagenerator(content_dir, style_dir, batch_size):
    def get_random_valid_image(paths):
        tmp = load_image(np.random.choice(paths))

        while tmp is None:
            tmp = load_image(np.random.choice(paths))

        return tmp

    content_paths = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, x) for x in os.listdir(style_dir)]

    c_list = [get_random_valid_image(content_paths) for _ in range(batch_size)]
    s_list = [get_random_valid_image(style_paths) for _ in range(batch_size)]

    c = np.zeros([batch_size] + config.input_shape)
    s = np.zeros([batch_size] + config.input_shape)

    while True:
        for i in range(batch_size):
            c[i] = c_list[i]
            s[i] = s_list[i]

        yield c, s


def main(save_name, batch_size=10):
    s_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    s_activations_op = get_activations(s_op, 's')

    c_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    c_activations_op = get_activations(c_op, 'c')
    t_op = adaIN(c_activations_op, s_activations_op)

    # Generated.
    z_op = get_decoder_output(t_op)
    z_activations_op = get_activations(z_op, 'z')
    z_t_op = adaIN(z_activations_op, s_activations_op)

    # Content, style, variation.
    loss_op, style_loss_op, content_loss_op = get_loss_op(t_op,
                                                          z_t_op,
                                                          z_activations_op,
                                                          s_activations_op)

    # Used for saving and gradient descent.
    trainable_variables = get_trainable_variables()

    print('Weights to be trained/saved.')
    print('\n'.join(sorted(map(lambda x: x.name, trainable_variables))))

    # All things training related.
    with tf.name_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.exponential_decay(1e-4, step_op,
                                                   10000, 0.1, staircase=True)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)
        train_op = optimizer_op.minimize(loss_op,
                                         global_step=step_op,
                                         var_list=trainable_variables)

    summary_op = get_summary(c_op, s_op, z_op, loss_op, style_loss_op,
                             content_loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('.', sess.graph)
        saver = tf.train.Saver(trainable_variables)

        try:
            saver.restore(sess, save_name)
            print('loaded %s.' % save_name)
        except:
            print('failed to load %s.' % save_name)

        datagen = get_datagenerator('/root/content', '/root/style', batch_size)

        for iteration in range(50000):
            c, s = next(datagen)

            if iteration % 10 != 0:
                 sess.run(train_op, {c_op: c, s_op: s})
            else:
                 _, summary, step = sess.run([train_op, summary_op, step_op],
                                             {c_op: c, s_op: s})

                 summary_writer.add_summary(summary, step)

            if iteration + 1 % 1000 == 0:
                 saver.save(sess, save_name, global_step=step)


if __name__ == '__main__':
    np.random.seed(0)

    main('model')
