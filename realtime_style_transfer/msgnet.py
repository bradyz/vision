import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.applications import vgg19

from PIL import Image

import config


def conv(x, k, c_out, stride=1, post=''):
    c_in = x.get_shape().as_list()[-1]

    # x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    std = 1.0 / tf.sqrt(1.0 * k * k * c_in)

    with tf.variable_scope('conv%s' % post):
        W = tf.get_variable('W', [k, k, c_in, c_out],
                initializer=tf.random_uniform_initializer(-std, std, seed=0))
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0))

        Wx = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        y = tf.nn.bias_add(Wx, b)

    return y


def conv7x7(x, c_out, post=''):
    return conv(x, 7, c_out, post='_7x7' + post)


def conv3x3(x, c_out, post=''):
    return conv(x, 3, c_out, post='_3x3' + post)


def conv1x1(x, c_out, stride=1, post=''):
    return conv(x, 1, c_out, stride, post='_1x1' + post)


def relu(x):
    with tf.name_scope('relu'):
        # y = tf.maximum(0.01 * x, x)
        y = tf.nn.relu(x)

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
        x_op = x_op - (103.939, 116.78, 123.68)

        vgg = vgg19.VGG19(include_top=False, input_tensor=x_op)

    return vgg.layers


def residual_block(x, k, scope='residual_down'):
    with tf.variable_scope(scope):
        x_id = x
        x_id = conv1x1(x_id, k, 2, post='res')

        x = conv1x1(x, k // 4, post='_1')
        x = relu(x)
        x = conv1x1(x, k // 4, 2, post='_2')
        x = relu(x)
        x = conv1x1(x, k, post='_3')

        x = x_id + x
        x = relu(x)

    return x


def residual_block_up(x, k, scope='residual_up'):
    with tf.variable_scope(scope):
        x_id = x
        x_id = bilinear_up(x_id)
        x_id = conv1x1(x_id, k, post='res')

        x = conv1x1(x, k // 4, post='_1')
        x = relu(x)
        x = conv1x1(x, k // 4, post='_2')
        x = relu(x)
        x = bilinear_up(x)
        x = conv1x1(x, k, post='_3')

        x = x_id + x
        x = relu(x)

    return x


def bilinear_up(x):
    _, h, w, _ = x.shape.as_list()

    return tf.image.resize_images(x, [2 * h, 2 * w],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def inspiration_layer(x, s_gram, post=''):
    n, h, w, c = x.shape.as_list()

    with tf.variable_scope('inspiration%s' % post):
        x = tf.reshape(x, [n, h * w, c])

        W = tf.get_variable('W', [c, c],
                initializer=tf.contrib.layers.xavier_initializer())

        WG = tf.map_fn(lambda gram: tf.matmul(W, gram), s_gram)
        x = tf.matmul(x, WG)
        x = tf.reshape(x, [n, h, w, c])

    return x


def get_decoder_output(c_op, s_gram_list):
    with tf.variable_scope('decoder'):
        net = c_op

        with tf.variable_scope('block1'):
            net = conv7x7(net, 64)
            net = relu(net)
            net = inspiration_layer(net, s_gram_list[0])
            net = residual_block(net, 128)

        with tf.variable_scope('block2'):
            net = inspiration_layer(net, s_gram_list[1])
            net = residual_block(net, 256)

        with tf.variable_scope('block3'):
            net = inspiration_layer(net, s_gram_list[2])
            net = residual_block(net, 512)

        with tf.variable_scope('block4'):
            net = inspiration_layer(net, s_gram_list[3])

        with tf.variable_scope('block5'):
            net = residual_block_up(net, 256)

        with tf.variable_scope('block6'):
            net = residual_block_up(net, 128)

        with tf.variable_scope('block7'):
            net = residual_block_up(net, 64)
            net = conv7x7(net, 3)

    return net


def mean_l2_diff(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_loss_op(z_op, z_layer_list, z_gram_list, c_layer_list, s_gram_list):
    with tf.name_scope('loss'):
        content_loss_op = 0.0

        for z_layer_op, c_layer_op in zip(z_layer_list, c_layer_list):
            content_loss_op += mean_l2_diff(z_layer_op, c_layer_op)

        style_loss_op = 0.0

        for z_gram_op, s_gram_op in zip(z_gram_list, s_gram_list):
            style_loss_op += mean_l2_diff(z_gram_op, s_gram_op)

        loss_op = style_loss_op + content_loss_op

    return loss_op, style_loss_op, content_loss_op


def get_summary(c_op, c_layer_list, s_op, s_gram_list,
                z_op, z_layer_list, z_gram_list, layer_names,
                loss_op, style_loss_op, content_loss_op,
                learn_rate_op, grad_var_op):
    tf.summary.scalar('learn_rate', learn_rate_op)

    tf.summary.scalar('content_loss', content_loss_op)
    tf.summary.scalar('style_loss', style_loss_op)
    tf.summary.scalar('total_loss', loss_op)

    tf.summary.image('content', tf.cast(c_op, tf.uint8), 10)
    tf.summary.image('style', tf.cast(s_op, tf.uint8), 10)
    tf.summary.image('generated', tf.cast(tf.clip_by_value(z_op, 0.0, 255.0), tf.uint8), 10)

    for i, layer_name in enumerate(layer_names):
        c_layer_op = tf.transpose(c_layer_list[0][:1], (3, 1, 2, 0))
        z_layer_op = tf.transpose(z_layer_list[0][:1], (3, 1, 2, 0))

        tf.summary.image('c_activations/' + layer_name, tf.to_float(c_layer_op), 30)
        tf.summary.image('z_activations/' + layer_name, tf.to_float(z_layer_op), 30)

    for i, layer_name in enumerate(layer_names):
        tf.summary.image('s_gram/' + layer_name, tf.expand_dims(s_gram_list[i], axis=-1))
        tf.summary.image('z_gram/' + layer_name, tf.expand_dims(z_gram_list[i], axis=-1))

    for dx_x in grad_var_op:
        dx, x = dx_x

        relative_grad = tf.div(tf.abs(dx), abs(x) + 1e-7)
        relative_grad_alive = tf.reduce_mean(tf.to_float(relative_grad > 1e-5))

        tf.summary.scalar('grad/' + x.name, relative_grad_alive)
        tf.summary.histogram('weight/' + x.name, x)
        tf.summary.histogram('abs_grad/' + x.name, abs(dx))
        tf.summary.histogram('rel_grad/' + x.name, relative_grad)

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

        # for i in range(batch_size):
        #     c_list[i] = get_random_valid_image(content_paths)
        #     s_list[i] = get_random_valid_image(style_paths)


def get_layers(activations, layer_names):
    return [op.output for op in activations if op.name in layer_names]


def gram_matrix(x):
    n, h, w, c = x.shape.as_list()

    u = tf.transpose(x, [0, 3, 1, 2])
    u = tf.reshape(u, [n, c, h * w])

    return tf.matmul(u, tf.transpose(u, [0, 2, 1]))


def main(save_name, batch_size=2):
    layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

    c_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    c_activations = get_activations(c_op, 'c')
    c_layer_list = get_layers(c_activations, layer_names)

    s_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    s_activations = get_activations(s_op, 's')
    s_gram_list = [gram_matrix(op) for op in get_layers(s_activations, layer_names)]

    # Generated.
    z_op = get_decoder_output(c_op, s_gram_list)
    z_activations = get_activations(z_op, 'z')
    z_layer_list = get_layers(z_activations, layer_names)
    z_gram_list = [gram_matrix(op) for op in z_layer_list]

    # Content, style, variation.
    loss_op, style_loss_op, content_loss_op = get_loss_op(
            z_op, z_layer_list, z_gram_list, c_layer_list, s_gram_list)

    # Used for saving and gradient descent.
    trainable_variables = get_trainable_variables()

    print('Weights to be trained/saved.')
    print('\n'.join(sorted(map(lambda x: x.name, trainable_variables))))

    # All things training related.
    with tf.name_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.exponential_decay(1e-8, step_op,
                                                   100000, 0.1, staircase=True)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        grad_var_op = optimizer_op.compute_gradients(loss_op,
                                                     var_list=trainable_variables)
        train_op = optimizer_op.apply_gradients(grad_var_op,
                                                global_step=step_op)


    summary_op = get_summary(c_op, c_layer_list, s_op, s_gram_list,
                             z_op, z_layer_list, z_gram_list, layer_names,
                             loss_op, style_loss_op, content_loss_op,
                             learn_rate_op, grad_var_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('.', sess.graph)
        saver = tf.train.Saver(trainable_variables)

        try:
            saver.restore(sess, save_name)
            print('loaded %s.' % save_name)
        except:
            print('failed to load %s.' % save_name)

        datagen = get_datagenerator('/root/code/content', '/root/code/style', batch_size)

        for iteration in range(1000000):
            c, s = next(datagen)

            if iteration % 10 != 0:
                 sess.run(train_op, {c_op: c, s_op: s})
            else:
                 _, summary, step = sess.run([train_op, summary_op, step_op],
                                             {c_op: c, s_op: s})

                 summary_writer.add_summary(summary, step)

            if (iteration + 1) % 1000 == 0:
                 saver.save(sess, save_name, global_step=step)


if __name__ == '__main__':
    np.random.seed(0)

    main('model')
