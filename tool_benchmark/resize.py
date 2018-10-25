import time

import numpy as np

import torch
import cv2
import tqdm

from PIL import Image


def resize_PIL(x, h_w):
    return Image.fromarray(x).resize((h_w, h_w), Image.ANTIALIAS)


def resize_torch(x, h_w):
    return torch.nn.functional.interpolate(torch.FloatTensor(x).unsqueeze(0), h_w)


def resize_cv2(x, h_w):
    return cv2.resize(x, (h_w, h_w))


def benchmark(x, func, h_w, n):
    tick = time.time()

    for i in tqdm.tqdm(range(n)):
        func(x[i], h_w)

    return time.time() - tick


def benchmark_np(h_w=256, n=10000):
    x = np.uint8(np.random.rand(n, h_w, h_w) * 255.0)

    a = benchmark(x, resize_PIL, h_w, n)
    b = benchmark(x, resize_torch, h_w, n)
    c = benchmark(x, resize_cv2, h_w, n)

    print(a / n)
    print(b / n)
    print(c / n)


if __name__ == '__main__':
    benchmark_np()
