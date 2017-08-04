import os

import numpy as np
import tensorflow as tf

import vgg19

from PIL import Image

from matplotlib import pyplot as plt

import config
import summary


def conv(x, k, c_out, stride=1, normalize=True, post=''):
    c_in = x.get_shape().as_list()[-1]

    # 3x3 filter, want 1 pad to fit.
    # 5x5 filter, want 2 pad to fit.
    # 7x7 filter, want 3 pad to fit.
    if k > 1:
        x = reflect_pad(x, k // 2)

    with tf.variable_scope('conv%s' % post):
        W = tf.get_variable('W', [k, k, c_in, c_out],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0))

        Wx = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
        y = Wx + b

        if normalize:
            y = instance_normalization(y)

    return y


def conv9x9(x, c_out, normalize, post=''):
    return conv(x, 9, c_out, normalize=normalize, post='_9x9' + post)


def conv3x3(x, c_out, stride=1, post=''):
    return conv(x, 3, c_out, stride, post='_3x3' + post)


def conv1x1(x, c_out, stride=1, post=''):
    return conv(x, 1, c_out, stride, post='_1x1' + post)


def relu(x):
    return tf.nn.relu(x)


def reflect_pad(x, k):
    return tf.pad(x, [[0, 0], [k , k], [k , k], [0, 0]], mode='REFLECT')


def gram_matrix(x):
    n, h, w, c = x.shape.as_list()

    u = tf.reshape(x, [n, h * w, c])
    u_transpose = tf.transpose(u, [0, 2, 1])

    return tf.matmul(u_transpose, u) / (h * w * c)


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


def get_activations(vgg, x_op, scope):
    with tf.name_scope('preprocess_vgg'):
        x_op = x_op[:, :, :, ::-1]
        x_op = x_op - (103.939, 116.78, 123.68)

    return vgg.feed_forward(x_op, scope)


def instance_normalization(x, post='', eps=1e-4):
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


def residual_block(x, k, repeat=2, post=''):
    for i in range(repeat):
        with tf.variable_scope('residual%s_%d' % (post, i)):
            x_id = x

            x = conv3x3(x, k, post='_1')
            x = relu(x)
            x = conv3x3(x, k, post='_2')

            x = x_id + x

    return x


def bilinear(x, scale):
    _, h, w, _ = x.shape.as_list()
    h_new, w_new = int(scale * h), int(scale * w)

    return tf.image.resize_nearest_neighbor( x, [h_new, w_new])


def bilinear_up(x):
    return bilinear(x, 2)


def bilinear_down(x):
    return bilinear(x, 0.5)


def down_block(x, k, post=''):
    x = conv3x3(x, k, 2, post=post)
    x = relu(x)

    return x


def up_block(x, k, post=''):
    x = conv3x3(x, k, post=post)
    x = relu(x)
    x = bilinear_up(x)

    return x


def inspiration_layer(x, s_gram, post=''):
    n, h, w, c = x.shape.as_list()

    with tf.variable_scope('inspiration%s' % post):
        x = tf.reshape(x, [n, h * w, c])

        std = 1.0 / tf.sqrt(1.0 * c / 2.0)

        W = tf.get_variable('W', [c, c],
                initializer=tf.truncated_normal_initializer(stddev=std))

        # Could adding a bias help?
        # b = tf.get_variable('b', [c],
        #         initializer=tf.constant_initializer(0.0))
        # WG = tf.map_fn(lambda gram: tf.matmul(W, gram) + b, s_gram)

        WG = tf.map_fn(lambda gram: tf.matmul(W, gram), s_gram)

        x = tf.matmul(x, WG)
        x = tf.reshape(x, [n, h, w, c])
        x = instance_normalization(x)

    return x


def get_decoder_output(c_op, s_gram_list):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('block1'):
            block1 = (c_op / 255.0)
            block1 = conv9x9(block1, 32, True)
            block1 = relu(block1)

            block1 = down_block(block1, 64, post='_1')
            block1 = inspiration_layer(block1, s_gram_list[0], '_1')

            block1 = down_block(block1, 128, post='_2')
            block1 = inspiration_layer(block1, s_gram_list[1], '_2')

            block1 = down_block(block1, 256, post='_3')
            block1 = inspiration_layer(block1, s_gram_list[2], '_3')

            block1 = residual_block(block1, 256, repeat=5)

        with tf.variable_scope('block2'):
            block2 = up_block(block1, 128, post='_1')
            block2 = up_block(block2, 64, post='_2')
            block2 = up_block(block2, 32, post='_3')
            block2 = conv9x9(block2, 3, False)

        with tf.variable_scope('predictions'):
            block3 = tf.sigmoid(block2) * 255.0

    net_layers = {'block1': block1,
                  'block2': block2,
                  'block3': block3}

    return block3, net_layers


def mean_l2_diff(x, y):
    return tf.reduce_mean(tf.square(x - y))


def get_loss_op(z_op, z_layer_list, z_gram_list,
                c_layer_list, s_gram_list, gram_weights):
    with tf.name_scope('loss'):
        content_loss_op = 0.0

        for z_layer_op, c_layer_op in zip(z_layer_list, c_layer_list):
            content_loss_op += mean_l2_diff(z_layer_op, c_layer_op)

        style_loss_op = 0.0

        for z_gram, s_gram, weight in zip(z_gram_list, s_gram_list, gram_weights):
            style_loss_op += weight * mean_l2_diff(z_gram, s_gram)

        style_loss_op = 1e0 * style_loss_op
        content_loss_op = 1e0 * content_loss_op

        loss_op = style_loss_op + content_loss_op

    return loss_op, style_loss_op, content_loss_op


def get_summary(c_op, c_layer_list, s_op, s_gram_list,
                z_op, z_layer_list, z_gram_list, z_net_layers, layer_names,
                loss_op, style_loss_op, content_loss_op,
                learn_rate_op, grad_var_op):
    tf.summary.scalar('learn_rate', learn_rate_op)

    tf.summary.scalar('content_loss', content_loss_op)
    tf.summary.scalar('style_loss', style_loss_op)
    tf.summary.scalar('total_loss', loss_op)

    tf.summary.image('content', tf.cast(c_op, tf.uint8), 10)
    tf.summary.image('style', tf.cast(s_op, tf.uint8), 10)
    tf.summary.image('generated', tf.cast(z_op, tf.uint8), 10)

    tf.summary.histogram('colors/content', c_op)
    tf.summary.histogram('colors/style', s_op)
    tf.summary.histogram('colors/generated', z_op)

    for i, layer_name in enumerate(layer_names):
        summary.view_activation('c_activation' + layer_name, c_layer_list[i])
        summary.view_activation('z_activation' + layer_name, z_layer_list[i])

    for layer_name, z_net_layer_op in z_net_layers.items():
        summary.view_activation('z_layer/' + layer_name, z_net_layer_op)

        tf.summary.histogram('activation/' + layer_name, z_net_layer_op)

    for i, layer_name in enumerate(layer_names):
        summary.view_gram('s_gram/' + layer_name, s_gram_list[i])
        summary.view_gram('z_gram/' + layer_name, z_gram_list[i])

        tf.summary.histogram('s_gram/' + layer_name, s_gram_list[i])
        tf.summary.histogram('z_gram/' + layer_name, z_gram_list[i])

    for dx_x in grad_var_op:
        dx, x = dx_x

        relative_grad = learn_rate_op * tf.div(tf.abs(dx), abs(x) + 1e-7)
        relative_grad_alive = tf.reduce_mean(
                tf.to_float(relative_grad >= 1e-7))

        tf.summary.scalar('grad/' + x.name, relative_grad_alive)
        tf.summary.histogram('weight/' + x.name, x)
        tf.summary.histogram('abs_grad/' + x.name, abs(dx))
        tf.summary.histogram('rel_grad/' + x.name, relative_grad)

    return tf.summary.merge_all()


def load_image(path, resize=None):
    x = Image.open(path)

    if len(x.getbands()) != 3:
        return None

    if resize:
        return np.float32(x.resize(resize))

    x = np.float32(x)
    h, w, _ = config.input_shape

    if x.shape[0] <= h or x.shape[1] <= w:
        return None

    i = np.random.randint(x.shape[0] - config.input_shape[0])
    j = np.random.randint(x.shape[1] - config.input_shape[1])

    return x[i:i+h,j:j+w]


def get_datagenerator(content_dir, style_dir, batch_size):
    def get_random_valid_image(paths, resize=(512, 512)):
        tmp = load_image(np.random.choice(paths), resize)

        while tmp is None:
            tmp = load_image(np.random.choice(paths), resize)

        return tmp

    content_paths = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, x) for x in os.listdir(style_dir)]
    s_list = [get_random_valid_image(style_paths) for _ in range(len(style_paths))]

    c = np.zeros([batch_size] + config.input_shape)
    s = np.zeros([batch_size] + config.input_shape)

    while True:
        for i in range(batch_size):
            c[i] = get_random_valid_image(content_paths)
            s[i] = s_list[np.random.randint(len(s_list))]

        yield c, s


def get_layers(activations, layer_names):
    return [activations[name] for name in layer_names]


def main(save_name, batch_size=4):
    layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    gram_weights = [1e-2, 1e1, 1e1, 8e0, 1e1]

    vgg = vgg19.VGG19('imagenet-vgg-verydeep-19.mat')

    # The content source and activations.
    c_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    c_activations = get_activations(vgg, c_op, 'c')
    c_layer_list = get_layers(c_activations, layer_names)

    # The style source and gram matrices.
    s_op = tf.placeholder(tf.float32, shape=[batch_size] + config.input_shape)
    s_activations = get_activations(vgg, s_op, 's')
    s_gram_list = [gram_matrix(op) for op in get_layers(s_activations, layer_names)]

    # Generated, activations, gram matrices.
    z_op, z_net_layers = get_decoder_output(c_op, s_gram_list)
    z_activations = get_activations(vgg, z_op, 'z')
    z_layer_list = get_layers(z_activations, layer_names)
    z_gram_list = [gram_matrix(op) for op in z_layer_list]

    # Content, style.
    loss_op, style_loss_op, content_loss_op = get_loss_op(
            z_op, [z_layer_list[3]], z_gram_list,
            [c_layer_list[3]], s_gram_list, gram_weights)

    # Used for saving and gradient descent.
    train_vars = get_trainable_variables()

    print('Weights to be trained/saved.')
    print('\n'.join(sorted(map(lambda x: x.name, train_vars))))

    # All things training related.
    with tf.name_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.exponential_decay(1e-3, step_op,
                                                   100000, 0.1, staircase=True)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)
        grad_var_op = optimizer_op.compute_gradients(loss_op, var_list=train_vars)
        train_op = optimizer_op.apply_gradients(grad_var_op, global_step=step_op)

    summary_op = get_summary(c_op, c_layer_list, s_op, s_gram_list,
                             z_op, z_layer_list, z_gram_list, z_net_layers, layer_names,
                             loss_op, style_loss_op, content_loss_op,
                             learn_rate_op, grad_var_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('.', sess.graph)
        saver = tf.train.Saver(train_vars)

        try:
            saver.restore(sess, 'model_decent_v7')
            print('Loaded %s.' % save_name)
        except Exception as e:
            print(e)
            print('Failed to load %s.' % save_name)

        datagen = get_datagenerator('/home/user/data/train2014',
                                    '/home/user/data/styles', batch_size)

        for iteration in range(config.num_steps):
            c, s = next(datagen)

            if iteration % config.checkpoint_steps != 0:
                sess.run(train_op, {c_op: c, s_op: s})
            else:
                _, summary, step = sess.run([train_op, summary_op, step_op],
                        {c_op: c, s_op: s})

                summary_writer.add_summary(summary, step)

            if (iteration + 1) % config.save_steps == 0:
                saver.save(sess, 'model_decent_v8')


if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(0)

    # run_loop()
    main('')
