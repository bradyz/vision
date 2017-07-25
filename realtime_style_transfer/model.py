import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.applications import vgg19

from PIL import Image

import config


def conv(x, k, c_out, stride=1, post_scope=''):
    c_in = x.get_shape().as_list()[-1]
    shape = [k, k, c_in, c_out]

    with tf.variable_scope('conv%s' % post_scope):
        W = tf.get_variable('W', shape,
                initializer=tf.random_normal_initializer(stddev=0.01))
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
        y = tf.maximum(0.01 * x, x)
        # y = tf.nn.relu(x)

    return y


def get_trainable_variables():
    weights = list()
    weights_names = set()

    for op in tf.global_variables():
        if op.name in weights_names:
            continue
        elif 'decoder' not in op.name:
            continue

        weights.append(op)
        weights_names.add(op.name)

    return weights


def get_activations(x_op, op_name):
    with tf.name_scope('vgg_%s' % op_name):
        x_op = x_op[:, :, :, ::-1]
        x_op = x_op - (103.939, 116.779, 123.68)

        vgg = vgg19.VGG19(include_top=False, input_tensor=x_op)

    return vgg.get_layer('block4_conv1').output


def get_mean_std(x_op):
    # Shape (n, c).
    x_mean_op, x_var_op = tf.nn.moments(x_op, axes=(1, 2))
    x_std_op = tf.sqrt(x_var_op)

    # Shape (n, 1, 1, c).
    x_mean_op = tf.expand_dims(tf.expand_dims(x_mean_op, axis=1), axis=1)
    x_std_op = tf.expand_dims(tf.expand_dims(x_std_op, axis=1), axis=1)

    return x_mean_op, x_std_op


def adaIN(c_activations_op, s_activations_op, eps=1e-6):
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
        x = conv3x3(x, k, '_2')
        x = relu(x)

    return x


def residual_block(x, scope):
    k = x.shape.as_list()[-1]

    with tf.variable_scope(scope):
        h = x

        x = conv1x1(x, k // 4, '_1')
        x = conv3x3(x, k // 4)
        x = conv1x1(x, k, '_2')

        x = x + h
        x = relu(x)

    return x


def decode_block(x, scope):
    with tf.variable_scope(scope):
        x = bilinear_up(x, scope)

    return x


def bilinear_up(x):
    _, h, w, _ = x.shape.as_list()

    return tf.image.resize_images(x, [2 * h, 2 * w],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def get_decoder_output(t_op):
    with tf.variable_scope('decoder'):
        net = t_op

        with tf.variable_scope('block1'):
            net = conv3x3(net, 256)
            net = relu(net)
            net = bilinear_up(net)

        with tf.variable_scope('block2'):
            net = conv3x3(net, 256, '_1')
            net = relu(net)
            net = conv3x3(net, 256, '_2')
            net = relu(net)
            net = conv3x3(net, 256, '_3')
            net = relu(net)
            net = conv3x3(net, 128, '_4')
            net = relu(net)
            net = bilinear_up(net)

        with tf.variable_scope('block3'):
            net = conv3x3(net, 128, '_1')
            net = relu(net)
            net = conv3x3(net, 64, '_2')
            net = relu(net)
            net = bilinear_up(net)

        with tf.variable_scope('block4'):
            net = conv3x3(net, 64, '_1')
            net = relu(net)
            net = conv3x3(net, 3, '_2')
            net = (tf.nn.tanh(net) + 1) * 255.0 / 2.0

    return net


def mean_l2_diff(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_loss_op(t_op, z_t_op, z_activations_op, s_activations_op, c_activations_op):
    with tf.name_scope('loss'):
       # 1st order characteristics of feature maps.
        s_activations_mean_op, s_activations_var_op = tf.nn.moments(s_activations_op,
                                                                    axes=(1, 2))
        z_activations_mean_op, z_activations_var_op = tf.nn.moments(z_activations_op,
                                                                    axes=(1, 2))

        style = mean_l2_diff(s_activations_mean_op, z_activations_mean_op) + \
                mean_l2_diff(s_activations_var_op, z_activations_var_op)

        content = mean_l2_diff(z_activations_op, c_activations_op)

        style *= 1e-1
        content *= 1

        loss = style + content

    return loss, style, content


def get_summary(c_op, s_op, z_op, loss_op, style_loss_op, content_loss_op,
                c_activations_op, s_activations_op, z_activations_op,
                grad_var_op, learn_rate_op):
    tf.summary.scalar('learn_rate', learn_rate_op)
    tf.summary.scalar('content_loss', content_loss_op)
    tf.summary.scalar('style_loss', style_loss_op)
    tf.summary.scalar('total_loss', loss_op)

    tf.summary.image('content', tf.cast(c_op, tf.uint8), 10)
    tf.summary.image('style', tf.cast(s_op, tf.uint8), 10)
    tf.summary.image('generated', tf.cast(z_op, tf.uint8), 10)

    c_activations_op = tf.transpose(c_activations_op[:1], (3, 1, 2, 0))
    s_activations_op = tf.transpose(s_activations_op[:1], (3, 1, 2, 0))
    z_activations_op = tf.transpose(z_activations_op[:1], (3, 1, 2, 0))

    tf.summary.image('c_activations', tf.cast(c_activations_op, tf.float32), 20)
    tf.summary.image('s_activations', tf.cast(s_activations_op, tf.float32), 20)
    tf.summary.image('z_activations', tf.cast(z_activations_op, tf.float32), 20)

    for dx_x in grad_var_op:
        dx, x = dx_x

        relative_grad = tf.div(tf.abs(dx), abs(x) + 1e-7)
        relative_grad_dead = tf.reduce_mean(tf.to_float(relative_grad > 1e-5))

        tf.summary.histogram(x.name, x)
        tf.summary.histogram('grad/' + x.name, relative_grad)
        tf.summary.scalar('grad/' + x.name, relative_grad_dead)

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


def main(save_name, batch_size=4):
    s_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    s_activations_op = get_activations(s_op, 's')

    c_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    c_activations_op = get_activations(c_op, 'c')
    t_op = adaIN(c_activations_op, c_activations_op)

    # Generated.
    z_op = get_decoder_output(t_op)
    z_activations_op = get_activations(z_op, 'z')
    z_t_op = adaIN(z_activations_op, s_activations_op)

    # Content, style, variation.
    loss_op, style_loss_op, content_loss_op = get_loss_op(t_op,
                                                          z_t_op,
                                                          z_activations_op,
                                                          s_activations_op,
                                                          c_activations_op)

    # Used for saving and gradient descent.
    trainable_variables = get_trainable_variables()

    print('Weights to be trained/saved.')
    print('\n'.join(sorted(map(lambda x: x.name, trainable_variables))))

    # All things training related.
    with tf.name_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.exponential_decay(1e-4, step_op,
                                                   5000, 0.1, staircase=True)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        grad_var_op = optimizer_op.compute_gradients(loss_op, var_list=trainable_variables)
        clipped_grad_var_op = [(tf.clip_by_value(g, -.1, .1), v) for g, v in grad_var_op]
        train_op = optimizer_op.apply_gradients(clipped_grad_var_op, global_step=step_op)
        # train_op = optimizer_op.minimize(loss_op,
        #                                  global_step=step_op,
        #                                  var_list=trainable_variables)

    summary_op = get_summary(c_op, s_op, z_op, loss_op, style_loss_op,
                             content_loss_op,
                             c_activations_op, s_activations_op, z_activations_op,
                             grad_var_op, learn_rate_op)

    with tf.Session() as sess:
        tf.set_random_seed(42)
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('.', sess.graph)
        saver = tf.train.Saver(trainable_variables)

        try:
            saver.restore(sess, save_name)
            print('loaded %s.' % save_name)
        except:
            print('failed to load %s.' % save_name)

        datagen = get_datagenerator('/root/content', '/root/style', batch_size)

        for iteration in range(200000):
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
