import util

import numpy as np
import tensorflow as tf


def _maxpool(x, kernel):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1],
            strides=[1, kernel, kernel, 1], padding='SAME')


def _conv(x, kernel, in_channels, out_channels, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable('W', [kernel, kernel, in_channels, out_channels],
                initializer=tf.random_normal_initializer(stddev=1e-4))
        b = tf.get_variable('b', [out_channels],
                initializer=tf.constant_initializer(0.0))

        Wx = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        return _maxpool(tf.nn.relu(Wx + b), 2)


def _deconv(x, kernel, in_channels, out_channels, scope):
    batch_size, height, width, _ = map(lambda dim: dim.value, x.get_shape())

    with tf.variable_scope(scope):
        W = tf.get_variable('W', [kernel, kernel, out_channels, in_channels],
                initializer=tf.random_normal_initializer(stddev=1e-4))

        b = tf.get_variable('b', [out_channels],
                initializer=tf.constant_initializer(0.0))

        # [16, 1, 1, 32] -> [16, 2, 2, 16].
        Wx = tf.nn.conv2d_transpose(x, W,
                output_shape=[batch_size, height * 2, width * 2, out_channels],
                strides=[1, 2, 2, 1])

        z = Wx + b

        # Leaky relu.
        return tf.maximum(z, z * 0.1)


def encoder(input_op):
    # Double feature size per layer.
    conv1 = _conv(input_op, 3, 3, 8, 'conv1')
    conv2 = _conv(conv1, 3, 8, 16, 'conv2')
    conv3 = _conv(conv2, 3, 16, 32, 'conv3')
    conv4 = _conv(conv3, 3, 32, 64, 'conv4')

    # Two channels for mean, stddev.
    conv5 = _conv(conv4, 1, 64, 2 * util.latent_n, 'conv5')

    mean_op = conv5[:,:,:,util.latent_n:]
    stddev_op = conv5[:,:,:,:util.latent_n]

    return mean_op, stddev_op


def sampled_z(mean_op, stddev_op):
    samples = tf.random_normal([util.batch_size, 1, 1, util.latent_n])
    sampled_z_op = mean_op + (stddev_op * samples)

    return sampled_z_op


def decoder(sampled_z_op):
    deconv5 = _deconv(sampled_z_op, 1, util.latent_n, 64, 'deconv5')
    deconv6 = _deconv(deconv5, 3, 64, 32, 'deconv6')
    deconv7 = _deconv(deconv6, 3, 32, 16, 'deconv7')
    deconv8 = _deconv(deconv7, 3, 16, 8, 'deconv8')
    deconv9 = _deconv(deconv8, 3, 8, 3, 'deconv9')

    return deconv9


def reconstruction_loss(x, x_):
    return tf.reduce_mean(tf.squared_difference(x, x_))


def kl_divergence_loss(mean_op, stddev_op):
    mean_loss = tf.square(mean_op)
    stddev_loss = stddev_op - tf.log(stddev_op + 1e-8)

    return 0.5 * tf.reduce_mean(mean_loss + stddev_loss)


def get_data(dataset, batch_size=util.batch_size):
    data = np.zeros(shape=(batch_size, 32, 32, 3))

    for i in range(batch_size):
        example = dataset[b'data'][i]
        example = example.reshape((3, 32, 32))
        example = example.transpose([1, 2, 0])

        data[i,:,:,:] = example

    return data


def train(dataset):
    with tf.Graph().as_default():
        sess = tf.Session()

        input_op = tf.placeholder(tf.float32, [util.batch_size] + util.image_shape)

        mean_op, stddev_op = encoder(input_op)
        sampled_z_op = sampled_z(mean_op, stddev_op)
        decoded_op = decoder(sampled_z_op)

        recon_loss_op = reconstruction_loss(input_op, decoded_op)
        vae_loss_op = kl_divergence_loss(mean_op, stddev_op)
        total_loss_op = recon_loss_op + util.vae_alpha * vae_loss_op

        tf.summary.scalar('recon_loss', recon_loss_op)
        tf.summary.scalar('vae_loss', vae_loss_op)
        tf.summary.scalar('total_loss', total_loss_op)
        tf.summary.image('inputs', input_op, util.batch_size // 4)
        tf.summary.image('decoded', decoded_op, util.batch_size // 4)
        merged_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('log_dir', sess.graph)

        # Minimize loss function.
        optimizer = tf.train.AdamOptimizer(1e-3)
        step_op = tf.Variable(0, name='step', trainable=False)
        train_op = optimizer.minimize(total_loss_op, global_step=step_op)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for _ in range(100000):
            feed = {input_op: get_data(dataset)}

            _, summary, step = sess.run([train_op, merged_op, step_op],
                    feed_dict=feed)

            summary_writer.add_summary(summary, step)


if __name__ == '__main__':
    # Keys: filenames, labels, batch_label, data.
    dataset = util.load_pickle('cifar-10-batches-py/data_batch_1')

    train(dataset)
