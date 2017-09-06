import tensorflow as tf

import vgg19
import ops


class MSGNet(object):
    def feed_forward(c_op, s_gram_op_list, scope='MSGNet'):
        with tf.variable_scope(scope):
            with tf.variable_scope('block1'):
                net = (c_op / 255.0) - 0.5
                net = ops.conv9x9(net, 32)
                net = ops.relu(net)

            net = ops.down_block(net, 64, s_gram_op_list[0], 'block2')
            net = ops.down_block(net, 128, s_gram_op_list[1], 'block3')
            net = ops.down_block(net, 256, s_gram_op_list[2], 'block4')
            net = ops.down_block(net, 512, s_gram_op_list[3], 'block5')

            net = ops.residual_block(net, 'block6', repeat=5)

            net = ops.up_block(net, 256, 'block7')
            net = ops.up_block(net, 128, 'block8')
            net = ops.up_block(net, 64, 'block9')
            net = ops.up_block(net, 32, 'block10')

            with tf.variable_scope('block11'):
                net = ops.conv9x9(net, 3, normalize=False)
                net = tf.sigmoid(net) * 255.0

        return net


class ContentStyleExtractor(object):
    def __init__(self, content_layers, style_layers, vgg_weights):
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.network = vgg19.VGG19(vgg_weights)

    def get_content_style(self, x_op, scope=None):
        with tf.name_scope('preprocess'):
            x_op = self.network.preprocess(x_op)

        activations = self.network.feed_forward(x_op, scope)

        # Content features.
        content_layer_list = list()

        for layer_name in self.content_layers:
            content_layer_list.append(activations[layer_name])

        # Style features.
        gram_list = list()

        for layer_name in self.style_layers:
            gram_list.append(gram_matrix(activations[layer_name]))

        return content_layer_list, gram_list


def gram_matrix(x):
    n, h, w, c = x.shape.as_list()

    u = tf.reshape(x, [n, h * w, c])
    u_transpose = tf.transpose(u, [0, 2, 1])

    return tf.matmul(u_transpose, u) / (h * w * c)


def mean_l2_diff(x, y):
    return tf.reduce_mean(tf.square(x - y))


def get_loss_op(z_layer_list, z_gram_list, c_layer_list, s_gram_list, gram_weights):
    with tf.name_scope('loss'):
        content_loss_op = 0.0

        for z_layer_op, c_layer_op in zip(z_layer_list, c_layer_list):
            content_loss_op += mean_l2_diff(z_layer_op, c_layer_op)

        style_loss_op = 0.0

        for z_gram, s_gram, weight in zip(z_gram_list, s_gram_list, gram_weights):
            style_loss_op += weight * mean_l2_diff(z_gram, s_gram)

        style_loss_op = 1e-1 * style_loss_op
        content_loss_op = 1e0 * content_loss_op

        loss_op = style_loss_op + content_loss_op

    return loss_op, style_loss_op, content_loss_op
