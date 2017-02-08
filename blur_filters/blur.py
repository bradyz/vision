import sys
from random import random

import cv2
import numpy as np

from matplotlib import pyplot as plt


def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    return


def average(image, x, y, width=3):
    rows, cols = image.shape

    total = 0
    elements = 0

    for dx in range(-(width/2), (width/2)+1):
        for dy in range(-(width/2), (width/2)+1):
            if x+dx < rows and x+dx >= 0 and y+dy < cols and y+dy >= 0:
                total += image[x+dx][y+dy]
                elements += 1

    return float(total) / elements


def blur(image):
    new_image = np.copy(image)

    for (i, j), value in np.ndenumerate(new_image):
        new_image[i][j] = average(image, i, j)

    return new_image


def add_noise(image, noise=100):
    new_image = np.copy(image)

    for (i, j), value in np.ndenumerate(new_image):
        new_image[i][j] = max(0, value + (random()-0.5) * noise)

    return new_image


# Kernel must have odd dimensions.
def star(image, kernel):
    def convolve(x, y):
        total = 0

        for dx in range(-(width/2), (width/2)+1):
            for dy in range(-(width/2), (width/2)+1):
                if x+dx < rows and x+dx >= 0 and y+dy < cols and y+dy >= 0:
                    total += image[x+dx][y+dy] * kernel[dx+width/2][dy+width/2]

        return total

    rows, cols = image.shape
    width, _ = kernel.shape
    new_image = np.copy(image)

    for (i, j), _ in np.ndenumerate(new_image):
        new_image[i][j] = convolve(i, j)

    return new_image


if __name__ == '__main__':
    image_path = sys.argv[1]

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_image = add_noise(image)
    blurred_image = blur(noisy_image)

    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    kernel = 1.0 / 16 * kernel
    gaussian_image = star(noisy_image, kernel)

    plt.axis('off')
    fig = plt.figure()

    plots = list()

    plots.append(fig.add_subplot(2, 2, 1))
    plt.imshow(image, cmap='gray')

    plots.append(fig.add_subplot(2, 2, 2))
    plt.imshow(noisy_image, cmap='gray')

    plots.append(fig.add_subplot(2, 2, 3))
    plt.imshow(blurred_image, cmap='gray')

    plots.append(fig.add_subplot(2, 2, 4))
    plt.imshow(gaussian_image, cmap='gray')

    for plot in plots:
        plot.set_xticks([])
        plot.set_yticks([])

    plt.show()
