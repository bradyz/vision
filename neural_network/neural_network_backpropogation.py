import numpy as np

from matplotlib import pyplot as plt


SIGMOID = False


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return np.multiply(sigmoid(x), 1.0 - sigmoid(x))


def relu(x):
    return np.clip(np.maximum(x, 0.0), -100.0, 100.0)


def drelu(x):
    return 1.0 * (x > 0.0)


def activation(x):
    if SIGMOID:
        return sigmoid(x)
    return relu(x)


def dactivation(x):
    if SIGMOID:
        return dsigmoid(x)
    return drelu(x)


def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


def forward_pass(x, W1, b1, W2, b2, W3, b3):
    # Layer 1
    y1 = np.dot(W1, x) + b1
    z1 = activation(y1)

    # Layer 2.
    y2 = np.dot(W2, z1) + b2
    z2 = activation(y2)

    # Layer 3.
    y = np.dot(W3, z2) + b3

    return (x, y1, z1, y2, z2), y

def check_shape(gradient, value):
    warning = "Expected: %s, Actual: %s" % (gradient.shape, value.shape)
    assert gradient.shape == value.shape, warning


def check(parameter, actual, x, label, W1, b1, W2, b2, W3, b3, h=1e-8, tol=1e-3):
    expected = np.zeros(actual.shape)

    m, n = parameter.shape

    for i in range(m):
        for j in range(n):
            tmp = parameter[i, j]

            parameter[i, j] = tmp + h
            _, y = forward_pass(x, W1, b1, W2, b2, W3, b3)
            loss_1 = mean_squared_error(y, label)

            parameter[i, j] = tmp - h
            _, y = forward_pass(x, W1, b1, W2, b2, W3, b3)
            loss_2 = mean_squared_error(y, label)

            expected[i, j] = (loss_1 - loss_2) / (2.0 * h)

            parameter[i, j] = tmp

    numer = np.square(expected - actual).sum()
    denom = np.square(actual).sum() + 1e-5

    return numer / denom < tol


def gradient_check(x, label, W1, b1, W2, b2, W3, b3,
        dldW1, dldb1, dldW2, dldb2, dldW3, dldb3):

    assert check(W3, dldW3, x, label, W1, b1, W2, b2, W3, b3), "W3"
    assert check(W2, dldW2, x, label, W1, b1, W2, b2, W3, b3), "W2"
    assert check(W1, dldW1, x, label, W1, b1, W2, b2, W3, b3), "W1"
    assert check(b3, dldb3, x, label, W1, b1, W2, b2, W3, b3), "b3"
    assert check(b2, dldb2, x, label, W1, b1, W2, b2, W3, b3), "b2"
    assert check(b1, dldb1, x, label, W1, b1, W2, b2, W3, b3), "b1"


def generate(function):
    example = np.matrix(np.random.rand(1) * 100.0)
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


def train(function, W1, b1, W2, b2, W3, b3, h=1e-5, batch_size=32, n=10000000):
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

        for batch in range(batch_size):
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
            dz2dy2 = dactivation(y2)

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
            dz1dy1 = dactivation(y1)

            dldz1 = np.dot(np.transpose(dy2dz1), dldy2)
            dldy1 = np.multiply(dz1dy1, dldz1)

            dy1dW1 = x
            dy1db1 = 1

            dldW1 += np.dot(dldy1, dy1dW1)
            dldb1 += np.dot(dldy1, dy1db1)

            check_shape(dldW1, W1)
            check_shape(dldb1, b1)

            # Occasionally check if gradients are okay.
            if i % 99 == 0 and batch == 0:
                gradient_check(x, label, W1, b1, W2, b2, W3, b3,
                        dldW1, dldb1, dldW2, dldb2, dldW3, dldb3)

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
    eps = 1e-5

    W1 = np.random.randn(n_1, 1) * eps
    b1 = np.zeros((n_1, 1))

    W2 = np.random.randn(n_2, n_1) * eps
    b2 = np.zeros((n_2, 1))

    W3 = np.random.randn(1, n_2) * eps
    b3 = np.zeros((1, 1))

    plt.ion()
    line = lambda x: 12.0 * np.cos(np.power(x, 0.4)) + 10.0
    train(line, W1, b1, W2, b2, W3, b3)
