from PIL import Image

from matplotlib import pyplot as plt
from matplotlib import patches

import numpy as np
import cv2

from bounding_box_labeler import Labeler


def intersects(x, dx, y, dy, nx, ndx, ny, ndy):
    if x + dx < nx:
        return False
    elif x > nx + ndx:
        return False
    elif y + dy < ny:
        return False
    elif y > ny + ndy:
        return False
    return True


def get_features():
    features = list()

    features.append(((0, 0, 0, 0),
                     (0, 1, 1, 0),
                     (0, 1, 1, 0),
                     (0, 0, 0, 0)))

    features.append(((0, 0, 0, 0),
                     (0, 1, 1, 0),
                     (0, 0, 0, 0),
                     (0, 0, 0, 0)))

    features.append(((0, 0, 0, 0),
                     (0, 0, 0, 0),
                     (0, 1, 1, 0),
                     (0, 0, 0, 0)))

    features.append(((1, 1, 0, 0),
                     (1, 1, 0, 0),
                     (0, 0, 1, 1),
                     (0, 0, 1, 1)))

    features.append(((0, 0, 1, 1),
                     (0, 0, 1, 1),
                     (1, 1, 0, 0),
                     (1, 1, 0, 0)))

    features = list(map(lambda x: np.float32(2 * np.array(x) - 1), features))

    return np.float32(features)


def get_activations(image, features):
    activations = list(map(lambda x: np.sum(np.multiply(image, x)), features))
    activations += [1.0]

    return np.float32(activations)


def downsample(image):
    image = Image.fromarray(image)
    image = image.resize((4, 4), Image.ANTIALIAS)
    image = np.array(image)

    return image


def gather_training_data(n=10, negative_samples=3, label_file='labels.txt'):
    video = cv2.VideoCapture(0)

    # Burn off initial crap.
    video.read()

    data = list()

    for i in range(n):
        _, raw_image = video.read()

        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        for _ in range(5):
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        height, width = gray_image.shape

        labeler = Labeler(gray_image, 'b')
        labeler.show()
        x, y, dx, dy = labeler.get_data()

        positive_image = gray_image[y:y+dy,x:x+dx]

        positive_filename = 'data/positive_%d' % i

        # Add positive sample.
        data.append(('%s.npy' % positive_filename, 1))
        np.save(positive_filename, positive_image)

        for j in range(negative_samples):
            found = False

            while not found:
                nx = np.random.randint(width)
                ny = np.random.randint(height)

                ndx = np.random.randint(width - nx)
                ndy = np.random.randint(height - ny)

                if ndx < 20:
                    continue
                elif ndy < 20:
                    continue

                if not intersects(x, dx, y, dy, nx, ndx, ny, ndy):
                    found = True

            negative_image = gray_image[ny:ny+ndy,nx:nx+ndx]

            negative_filename = 'data/negative_%d_%d' % (i, j)

            # Add negative sample.
            data.append(('%s.npy' % negative_filename, 0))
            np.save(negative_filename, negative_image)

    with open(label_file, 'w+') as fd:
        for row in data:
            fd.write(','.join(map(str, row)))
            fd.write('\n')


def optimize(x, y, iterations=10000, h=1e-6):
    n, m = x.shape

    w = np.random.randn(m)

    for _ in range(iterations):
        hinge_loss = np.maximum(0.0, 1.0 - y * np.dot(x, w))

        if sum(hinge_loss > 0) == 0:
            break

        dw = np.multiply(x[hinge_loss > 0,:], -y[hinge_loss > 0, np.newaxis])
        dw = np.mean(dw, axis=0)

        w = w - h * dw

    return w


def train(label_file='labels.txt'):
    x = list()
    y = list()

    features = get_features()

    with open(label_file, 'r') as fd:
        for line in fd.read().split('\n'):
            if line:
                filename, label = line.split(',')

                image = np.load(filename)
                image = downsample(image)

                x.append(get_activations(image, features))
                y.append(label)

    x = np.array(x, dtype='float')
    y = 2.0 * np.array(y, dtype='float') - 1.0

    return optimize(x, y)


def test(coefficients):
    video = cv2.VideoCapture(0)

    # Burn off initial crap.
    video.read()
    video.read()

    features = get_features()

    while True:
        _, raw_image = video.read()

        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        for _ in range(5):
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        height, width = gray_image.shape

        best = None
        best_score = None

        for x in range(0, width, 200):
            for y in range(0, height, 200):
                for dx in range(200, width - x, 200):
                    for dy in range(200, height - y, 200):
                        candidate_image = gray_image[y:y+dy,x:x+dx]
                        candidate_image = downsample(candidate_image)

                        activations = get_activations(candidate_image, features)
                        score = np.dot(coefficients, activations)

                        if best_score is None or score > best_score:
                            best_score = score
                            best = (x, y, dx, dy)

        x, y, dx, dy = best

        cv2.rectangle(gray_image, (x, y), (x + dx, y + dy), 0, thickness=2)
        cv2.imshow('Face Detection', gray_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    np.random.seed(1337)

    # gather_training_data()
    coefficients = train()
    test(coefficients)
