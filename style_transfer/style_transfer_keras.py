import PIL
import keras

from keras import backend as K
from matplotlib import pyplot as plt

import numpy as np


def load_image(path):
    return np.clip(np.float32(np.array(PIL.Image.open(path))), 0.0, 255.0) / 255.0


def show_image(image, title):
    plt.cla()

    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    plt.pause(0.01)


def to_tensor(x):
    return x.reshape((1,) + x.shape)


def to_constant(x):
    return K.constant(x, shape=x.shape)


def get_content_loss(x_op, c_op):
    return K.mean(K.square(x_op - c_op))


def get_style_loss(x_op, s_op):
    return K.mean(K.square(x_op * x_op - s_op * s_op))


foo = None
bar = None


def get_total_loss(c, s, vgg, alpha=10000.0, layer='block5_conv2'):
    activation = vgg.get_layer(layer).output

    get_activation = K.function([vgg.layers[0].input], [activation])

    c_activation_op = to_constant(get_activation([to_tensor(c)])[0])
    s_activation_op = to_constant(get_activation([to_tensor(s)])[0])

    content_loss = get_content_loss(activation, c_activation_op)
    style_loss = get_style_loss(activation, s_activation_op)

    global foo, bar
    foo = K.function([vgg.layers[0].input], [content_loss])
    bar = K.function([vgg.layers[0].input], [style_loss])

    return content_loss + alpha * style_loss


def get_gradient(c, s):
    x_op = keras.layers.Input((224, 224, 3))
    vgg = keras.applications.vgg16.VGG16(include_top=False, input_tensor=x_op)

    loss_op = get_total_loss(c, s, vgg)

    gradient_func = K.function([x_op], K.gradients(loss_op, [x_op]))

    return lambda x: gradient_func([x])[0]


def optimize(c, s, h=1e-3, noise=False):
    if noise:
        x = to_tensor(np.random.randn(224, 224, 3))
    else:
        x = to_tensor(np.copy(c))

    get_dx = get_gradient(c, s)

    for i in range(10000):
        x = x - h * get_dx(x)
        x = np.clip(x, 0.0, 1.0)

        plt.subplot(221)
        show_image(x[0], "Iteration: %d" % i)


if __name__ == '__main__':
    plt.ion()
    plt.show(block=False)

    content_image = load_image('content.jpg')
    style_image = load_image('style.jpg')

    plt.subplot(222)
    show_image(content_image, 'Content Image')

    plt.subplot(212)
    show_image(style_image, 'Style Image')

    plt.subplot(221)
    optimize(content_image, style_image)
