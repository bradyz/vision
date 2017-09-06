import tensorflow as tf


def view_activation(name, layer_op):
    # ":1" to just view the first samples' layers.
    with tf.name_scope('summary/activation/'):
        c_layer_op = tf.transpose(layer_op[:1], (3, 1, 2, 0))
        c_layer_op = tf.to_float(c_layer_op)

    return tf.summary.image(name, c_layer_op, 30)


def view_gram(name, gram_op):
    return tf.summary.image(name, tf.expand_dims(gram_op, axis=-1))


def weight_summary(grad_var_op):
    summaries = list()

    for dx, x in grad_var_op:
        if dx is None:
            continue

        summaries += [tf.summary.histogram('weight/' + x.name, x)]

    return summaries

def gradient_summary(grad_var_op, learn_rate_op, eps=1e-7):
    summaries = list()

    with tf.name_scope('summary/gradients'):
        for dx, x in grad_var_op:
            if dx is None:
                continue

            rel_dx = learn_rate_op * tf.div(tf.abs(dx), abs(x) + eps)
            rel_dx_alive = tf.reduce_mean(tf.to_float(rel_dx >= eps))

            summaries += [tf.summary.scalar('dx/' + x.name, rel_dx_alive),
                          tf.summary.histogram('dx/abs/' + x.name, abs(dx)),
                          tf.summary.histogram('dx/rel/' + x.name, rel_dx)]

    return summaries
