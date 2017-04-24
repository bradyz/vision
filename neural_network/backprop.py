import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return np.multiply(y, 1.0 - y)


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
    example = np.matrix(np.random.rand(1))
    return example, function(example)


def train(function, W1, b1, W2, b2, W3, b3, n_iterations=1000000, h=1e-6):
    for i in range(n_iterations):
        example, label = generate(function)

        activations, y = forward_pass(example, W1, b1, W2, b2, W3, b3)
        x, y1, z1, y2, z2 = activations

        # Loss is 1/2 * (y_pred - y_true)^2.
        loss = mean_squared_error(y, label)

        if np.isnan(loss):
            print("NaN encountered at iteration %d." % i)
            break
        elif i % 10000 == 0:
            validation_n = 100
            validation_loss = 0.0

            for _ in range(validation_n):
                x_, y_ = generate(function)
                _, y_valid = forward_pass(x_, W1, b1, W2, b2, W3, b3)
                validation_loss += mean_squared_error(y_valid, y_)

            validation_loss /= validation_n

            print("Validation loss %.3f" % validation_loss)

        # Derivative of loss with respect to prediction.
        dldy = np.array(y - label).reshape(1, 1)

        # Layer 3.
        dydW3 = np.transpose(np.matrix(z2))
        dydb3 = np.matrix(1)

        dldW3 = np.dot(dldy, dydW3)
        dldb3 = np.dot(dldy, dydb3)

        check_shape(dldW3, W3)
        check_shape(dldb3, b3)

        # Layer 2.
        dydz2 = np.matrix(W3)
        dz2dy2 = dsigmoid(y2)

        dldz2 = np.dot(np.transpose(dydz2), dldy)
        dldy2 = np.multiply(dz2dy2, dldz2)

        dy2dW2 = np.transpose(np.matrix(z1))
        dy2db2 = np.matrix(1)

        dldW2 = np.dot(dldy2, dy2dW2)
        dldb2 = np.dot(dldy2, dy2db2)

        check_shape(dldW2, W2)
        check_shape(dldb2, b2)

        # Layer 1.
        dy2dz1 = W2
        dz1dy1 = dsigmoid(y1)

        dldz1 = np.dot(np.transpose(dy2dz1), dldy2)
        dldy1 = np.multiply(dz1dy1, dldz1)

        dy1dW1 = x
        dy1db1 = 1

        dldW1 = np.dot(dldy1, dy1dW1)
        dldb1 = np.dot(dldy1, dy1db1)

        check_shape(dldW1, W1)
        check_shape(dldb1, b1)

        # Apply gradients.
        W3 -= h * dldW3
        b3 -= h * dldb3

        W2 -= h * dldW2
        b2 -= h * dldb2

        W1 -= h * dldW1
        b1 -= h * dldb1
        #
if __name__ == "__main__":
    W1 = np.random.randn(3, 1) * 1e-2
    b1 = np.random.randn(3, 1) * 1e-2

    W2 = np.random.randn(2, 3) * 1e-2
    b2 = np.random.randn(2, 1) * 1e-2

    W3 = np.random.randn(1, 2) * 1e-2
    b3 = np.random.randn(1, 1) * 1e-2

    line = lambda x: 12.5 * x + 10
    train(line, W1, b1, W2, b2, W3, b3)
