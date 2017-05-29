import PIL
import keras

from keras import backend as K
from matplotlib import pyplot as plt

import numpy as np


MEAN_IMAGE = np.array((123.68, 116.78, 103.93)) / 255.0


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
    return x[:,:,:,:] - MEAN_IMAGE


def postprocess(x):
    return np.clip(x + MEAN_IMAGE, 0.0, 1.0)


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
    c_op = to_constant(get_activation([preprocess(to_tensor(c))])[0])

    return K.sum(K.square(x_op - c_op))


def get_style_loss(s, vgg, layer):
    activation = vgg.get_layer(layer).output
    get_activation = K.function([vgg.layers[0].input], [activation])

    x_op = activation
    s_op = to_constant(get_activation([to_tensor(s)])[0])

    return K.mean(K.square(gram_matrix(x_op) - gram_matrix(s_op)))


def get_total_loss(c, s, vgg, n=5, alpha=0.5, beta=0.05):
    content_loss = 0.0
    style_loss = 0.0

    for i in range(1, n+1):
        layer = 'block%d_conv1' % i

        content_loss += get_content_loss(c, vgg, layer)
        style_loss += get_style_loss(s, vgg, layer)

    content_loss /= n
    style_loss /= n

    return alpha * content_loss + beta * style_loss


def get_deep_dream_loss(c, vgg, alpha=1.0, beta=2.0, content_layer='input_1'):
    layer = vgg.layers[-1]

    d = layer.output.shape.as_list()[-1]
    d = np.random.randint(d)

    content_loss = get_content_loss(c, vgg, content_layer)
    deep_dream_loss = -K.sum(K.square(layer.output[:,:,:,d]))

    return alpha * content_loss + beta * deep_dream_loss


def get_gradient(x_op, loss_op):
    gradient_op = K.gradients(loss_op, [x_op])[0]
    gradient_op = gradient_op / (K.sqrt(K.sum(K.square(gradient_op)) + 1e-8))

    gradient_func = K.function([x_op], [gradient_op])

    return lambda x: gradient_func([x])[0]


def optimize(x_initial, get_dx, h=1.0, mu=0.9, n=10000):
    x = np.copy(to_tensor(x_initial))
    v = np.zeros(shape=(x.shape))

    for i in range(1, n):
        x = preprocess(x)

        dx = get_dx(x + mu * v)
        v = mu * v - h * dx

        x = x + v

        x = postprocess(x)

        show_image(x[0], "Iteration: %d" % i)


def style_transfer(content_path, style_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    plt.subplot2grid((3, 2), (0, 0))
    show_image(content_image, 'Content Image')

    plt.subplot2grid((3, 2), (0, 1))
    show_image(style_image, 'Style Image')

    plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)

    x_op = keras.layers.Input((224, 224, 3))
    vgg = keras.applications.vgg16.VGG16(include_top=False, input_tensor=x_op)
    loss_op = get_total_loss(content_image, style_image, vgg)

    optimize(content_image, get_gradient(x_op, loss_op))


def deep_dream(content_image):
    content_image = load_image(content_image)

    plt.subplot(121)
    show_image(content_image, 'Content Image')

    plt.subplot(122)

    x_op = keras.layers.Input((224, 224, 3))
    vgg = keras.applications.vgg16.VGG16(include_top=False, input_tensor=x_op)
    loss_op = get_deep_dream_loss(content_image, vgg)

    optimize(content_image, get_gradient(x_op, loss_op))


if __name__ == '__main__':
    np.random.seed(1337)

    plt.ion()
    plt.show(block=False)

    deep_dream('content.jpg')
    style_transfer('content.jpg', 'cubism1.jpg')
