import tensorflow as tf


def view_activation(name, layer_op):
    # ":1" to just view the first samples' layers.
    c_layer_op = tf.transpose(layer_op[:1], (3, 1, 2, 0))
    c_layer_op = tf.to_float(c_layer_op)

    return tf.summary.image(name, c_layer_op, 30)


def view_gram(name, gram_op):
    return tf.summary.image(name, tf.expand_dims(gram_op, axis=-1))
