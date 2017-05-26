import PIL
import keras

from keras import backend as K
from matplotlib import pyplot as plt

import numpy as np


def load_image(path):
    return np.float32(np.array(PIL.Image.open(path))) / 255.0


def show_image(image, title):
    plt.cla()

    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    plt.pause(0.01)


def preprocess(x):
    x[:, :, :, 0] -= 123.68 / 255.0
    x[:, :, :, 1] -= 116.78 / 255.0
    x[:, :, :, 2] -= 103.93 / 255.0

    return x


def postprocess(x):
    x[:, :, :, 0] += 123.68 / 255.0
    x[:, :, :, 1] += 116.78 / 255.0
    x[:, :, :, 2] += 103.93 / 255.0
    x = np.clip(x, 0.0, 1.0)

    return x


def to_tensor(x):
    return x.reshape((1,) + x.shape)


def to_constant(x):
    return K.constant(x, shape=x.shape)


def gram_matrix(x_op):
    _, w, h, f = map(lambda dim: dim.value, x_op.shape)

    x_flatten_op = K.reshape(x_op, (-1, w * h, f))
    x_flatten_t_op = K.permute_dimensions(K.transpose(x_flatten_op), (2, 0, 1))

    return K.batch_dot(x_flatten_t_op, x_flatten_op)


def get_content_loss(c, vgg, layer):
    activation = vgg.get_layer(layer).output
    get_activation = K.function([vgg.layers[0].input], [activation])

    x_op = activation
    c_op = to_constant(get_activation([to_tensor(c)])[0])

    return K.sum(K.square(x_op - c_op))


def get_style_loss(s, vgg, layer):
    activation = vgg.get_layer(layer).output
    get_activation = K.function([vgg.layers[0].input], [activation])

    x_op = activation
    s_op = to_constant(get_activation([to_tensor(s)])[0])

    return K.mean(K.square(gram_matrix(x_op) - gram_matrix(s_op)))


def get_total_loss(c, s, vgg, n=5, alpha=0.05, beta=0.008):
    content_loss = 0.0
    style_loss = 0.0

    for i in range(1, n+1):
        layer = 'block%d_conv1' % i

        content_loss += get_content_loss(c, vgg, layer)
        style_loss += get_style_loss(s, vgg, layer)

    content_loss /= n
    style_loss /= n

    return alpha * content_loss + beta * style_loss


def get_gradient(c, s):
    x_op = keras.layers.Input((224, 224, 3))
    vgg = keras.applications.vgg16.VGG16(include_top=False, input_tensor=x_op)

    loss_op = get_total_loss(c, s, vgg)

    gradient_func = K.function([x_op], K.gradients(loss_op, [x_op]))

    return lambda x: gradient_func([x])[0]


def optimize(c, s, h=1e-5, noise=False):
    if noise:
        x = to_tensor(np.random.randn(224, 224, 3))
    else:
        x = to_tensor(np.copy(c))

    plt.subplot(223)
    show_image(x[0], "Iteration: 0")

    get_dx = get_gradient(c, s)

    for i in range(1, 10000):
        x = preprocess(x)

        dx = get_dx(x)

        x = x - h * dx

        x = postprocess(x)

        plt.subplot(223)
        show_image(x[0], "Iteration: %d" % i)


if __name__ == '__main__':
    plt.ion()
    plt.show(block=False)

    content_image = load_image('content.jpg')
    style_image = load_image('style.jpg')

    plt.subplot(221)
    show_image(content_image, 'Content Image')

    plt.subplot(224)
    show_image(style_image, 'Style Image')

    plt.subplot(223)
    optimize(content_image, style_image, noise=False)
