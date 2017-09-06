import os

import numpy as np
import tensorflow as tf

import msgnet
import summary
import dataprovider
import config


def get_train_op(loss_op, train_vars, learn_rate=1e-3, decay=0.1):
    with tf.name_scope('training'):
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.exponential_decay(learn_rate, step_op,
                                                   25000, decay, staircase=True)
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)
        grad_var_op = optimizer_op.compute_gradients(loss_op, var_list=train_vars)
        train_op = optimizer_op.apply_gradients(grad_var_op, global_step=step_op)

    return train_op, learn_rate_op, step_op, grad_var_op


def get_summary(c_op, s_op, s_gram_list,
                z_op, z_gram_list, layer_names,
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
        summary.view_gram('s_gram/' + layer_name, s_gram_list[i])
        summary.view_gram('z_gram/' + layer_name, z_gram_list[i])

        tf.summary.histogram('s_gram/' + layer_name, s_gram_list[i])
        tf.summary.histogram('z_gram/' + layer_name, z_gram_list[i])

    summary.gradient_summary(grad_var_op, learn_rate_op)

    return tf.summary.merge_all()


def main(log_dir, content_dir, style_dir,
         content_layers, style_layers, gram_weights,
         vgg_weights, model_name, input_shape,
         batch_size):
    save_path = os.path.join(log_dir, model_name)

    extractor = msgnet.ContentStyleExtractor(content_layers, style_layers, vgg_weights)

    # The content source and activations.
    c_op = tf.placeholder(tf.float32, shape=[batch_size] + input_shape)
    c_content_layer_list, _ = extractor.get_content_style(c_op, 'c')

    # The style source and gram matrices.
    s_op = tf.placeholder(tf.float32, shape=[batch_size] + input_shape)
    _, s_gram_list = extractor.get_content_style(s_op, 's')

    # Generated, activations, gram matrices.
    z_op = msgnet.MSGNet.feed_forward(c_op, s_gram_list)
    z_content_layer_list, z_gram_list = extractor.get_content_style(z_op, 'z')

    # Content, style.
    loss_op, style_loss_op, content_loss_op = msgnet.get_loss_op(
            z_content_layer_list, z_gram_list,
            c_content_layer_list, s_gram_list,
            gram_weights)

    # Used for saving and gradient descent.
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'MSGNet')

    print('Weights to be trained/saved.')
    print('\n'.join(sorted(map(lambda x: x.name, train_vars))))

    train_op, learn_rate_op, step_op, grad_var_op = get_train_op(loss_op, train_vars)

    summary_op = get_summary(c_op, s_op, s_gram_list,
                             z_op, z_gram_list, extractor.style_layers,
                             loss_op, style_loss_op, content_loss_op,
                             learn_rate_op, grad_var_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver(train_vars)

        try:
            saver.restore(sess, save_path)
            print('Loaded %s.' % save_path)
        except Exception as e:
            print(e)
            print('Failed to load %s.' % save_path)

        datagen = dataprovider.get_datagenerator(content_dir, style_dir,
                                                 input_shape, batch_size)

        for _ in range(config.num_steps):
            c, s = next(datagen)

            step, _ = sess.run([step_op, train_op], {c_op: c, s_op: s})

            if step % config.checkpoint_steps == 0:
                _, summary = sess.run([train_op, summary_op], {c_op: c, s_op: s})

                summary_writer.add_summary(summary, step)

            if step % config.save_steps == 0:
                saver.save(sess, save_path)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    main(config.log_dir, config.content_dir, config.style_dir,
         config.content_layers, config.style_layers, config.gram_weights,
         config.vgg_weights, config.model_name, config.input_shape,
         config.batch_size)
