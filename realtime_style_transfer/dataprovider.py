import os
import numpy as np

from PIL import Image


def load_image(path, input_shape, use_whole):
    h, w, _ = input_shape
    x = Image.open(path)

    if len(x.getbands()) != 3:
        return None

    if use_whole:
        x = x.resize([h, w])
        x = np.float32(x)

        return x

    x = np.float32(x)

    if x.shape[0] < h or x.shape[1] < w:
        return None

    i = np.random.randint(x.shape[0] - input_shape[0] + 1)
    j = np.random.randint(x.shape[1] - input_shape[1] + 1)

    return x[i:i+h,j:j+w]


def get_image(paths, input_shape, use_whole=False):
    tmp = load_image(np.random.choice(paths), input_shape, use_whole)

    while tmp is None:
        tmp = load_image(np.random.choice(paths), input_shape, use_whole)

    return tmp


def get_datagenerator(content_dir, style_dir, input_shape, batch_size):
    content_paths = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, x) for x in os.listdir(style_dir)]
    s_list = [get_image(style_paths, input_shape, True) for _ in range(len(style_paths))]

    c = np.zeros([batch_size] + input_shape)
    s = np.zeros([batch_size] + input_shape)

    while True:
        for i in range(batch_size):
            c[i] = get_image(content_paths, input_shape)
            s[i] = s_list[np.random.randint(len(s_list))]

        yield c, s
