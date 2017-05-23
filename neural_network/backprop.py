import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return np.multiply(sigmoid(y), 1.0 - sigmoid(y))


def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


def forward_pass(x, W1, b1, W2, b2, W3, b3):
    # Layer 1
    y1 = np.dot(W1, x) + b1
    z1 = sigmoid(y1)

    # Layer 2.
    y2 = np.dot(W2, z1) + b2
    z2 = sigmoid(y2)

    # Layer 3.
    y = np.dot(W3, z2) + b3

    return (x, y1, z1, y2, z2), y

def check_shape(gradient, value):
    warning = "Expected: %s, Actual: %s" % (gradient.shape, value.shape)
    assert gradient.shape == value.shape, warning


def generate(function):
    example = np.matrix(np.random.rand(1) * 10)
    return example, function(example)


def sample(function, W1, b1, W2, b2, W3, b3, num_samples=100):
    plt.clf()

    examples = list()
    labels = list()
    predictions = list()

    for _ in range(num_samples):
        example, label = generate(function)
        _, prediction = forward_pass(example, W1, b1, W2, b2, W3, b3)

        examples.append(example.tolist()[0][0])
        labels.append(label.tolist()[0][0])
        predictions.append(prediction.tolist()[0][0])

    plt.plot(examples, labels, "ro")
    plt.plot(examples, predictions, "bo")

    plt.pause(0.01)


def train(function, W1, b1, W2, b2, W3, b3, h=1e-4, batch_size=64, n=10000000):
    for i in range(n):
        if i % 100 == 0:
            validation_n = 100
            validation_loss = 0.0

            for _ in range(validation_n):
                x_, y_ = generate(function)
                _, y_valid = forward_pass(x_, W1, b1, W2, b2, W3, b3)
                validation_loss += mean_squared_error(y_valid, y_)

            validation_loss /= validation_n

            print("Validation loss %.3f" % validation_loss)

            sample(function, W1, b1, W2, b2, W3, b3)

            continue

        dldW3 = np.zeros((W3.shape))
        dldb3 = np.zeros((b3.shape))

        dldW2 = np.zeros((W2.shape))
        dldb2 = np.zeros((b2.shape))

        dldW1 = np.zeros((W1.shape))
        dldb1 = np.zeros((b1.shape))

        for _ in range(batch_size):
            example, label = generate(function)

            activations, y = forward_pass(example, W1, b1, W2, b2, W3, b3)
            x, y1, z1, y2, z2 = activations

            # Loss is 1/2 * (y_pred - y_true)^2.
            loss = mean_squared_error(y, label)

            if np.isnan(loss):
                print("NaN encountered at iteration %d." % i)
                break

            # Derivative of loss with respect to prediction.
            dldy = np.array(y - label).reshape(1, 1)

            # Layer 3.
            dydW3 = np.transpose(np.matrix(z2))
            dydb3 = np.matrix(1)

            dldW3 += np.dot(dldy, dydW3)
            dldb3 += np.dot(dldy, dydb3)

            check_shape(dldW3, W3)
            check_shape(dldb3, b3)

            # Layer 2.
            dydz2 = np.matrix(W3)
            dz2dy2 = dsigmoid(y2)

            dldz2 = np.dot(np.transpose(dydz2), dldy)
            dldy2 = np.multiply(dz2dy2, dldz2)

            dy2dW2 = np.transpose(np.matrix(z1))
            dy2db2 = np.matrix(1)

            dldW2 += np.dot(dldy2, dy2dW2)
            dldb2 += np.dot(dldy2, dy2db2)

            check_shape(dldW2, W2)
            check_shape(dldb2, b2)

            # Layer 1.
            dy2dz1 = W2
            dz1dy1 = dsigmoid(y1)

            dldz1 = np.dot(np.transpose(dy2dz1), dldy2)
            dldy1 = np.multiply(dz1dy1, dldz1)

            dy1dW1 = x
            dy1db1 = 1

            dldW1 += np.dot(dldy1, dy1dW1)
            dldb1 += np.dot(dldy1, dy1db1)

            check_shape(dldW1, W1)
            check_shape(dldb1, b1)

        # Apply gradients.
        W3 -= h * dldW3
        b3 -= h * dldb3

        W2 -= h * dldW2
        b2 -= h * dldb2

        W1 -= h * dldW1
        b1 -= h * dldb1


if __name__ == "__main__":
    n_1 = 8
    n_2 = 8

    W1 = np.random.randn(n_1, 1) * 1e-2
    b1 = np.zeros((n_1, 1))

    W2 = np.random.randn(n_2, n_1) * 1e-2
    b2 = np.zeros((n_2, 1))

    W3 = np.random.randn(1, n_2) * 1e-2
    b3 = np.zeros((1, 1))

    plt.ion()
    line = lambda x: -3.5 * x + 10.0
    train(line, W1, b1, W2, b2, W3, b3)
