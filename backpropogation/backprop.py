import numpy as np


STEP_SIZE = 1e-2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return np.multiply(y, 1.0 - y)


def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


def neural_net(n_inputs, n_hidden, n_outputs, n_units=5, weight=1e-2):
    layers = list()

    A_1 = np.random.randn(n_units, n_inputs) * weight
    b_1 = np.random.randn(n_units, 1) * weight
    layers.append([A_1, b_1])

    for _ in range(n_hidden):
        A_i = np.random.randn(n_units, n_units) * weight
        b_i = np.random.randn(n_units, 1) * weight
        layers.append([A_i, b_i])

    A_2 = np.random.randn(n_outputs, n_units) * weight
    b_2 = np.random.randn(n_outputs, 1) * weight
    layers.append([A_2, b_2])

    return layers


def forward_pass(model, example):
    activations = [None for _ in model]

    x = example

    for i, layer in enumerate(model):
        A, b = layer
        activations[i] = {"input": x,
                          "output": sigmoid(np.dot(A, x) + b)}

        x = activations[i]["output"]

    return activations[-1]["output"], activations


def train(function, model, n_iterations=100000):
    right = 0
    total = 0

    for _ in range(n_iterations):
        example = np.random.randint(2, size=(2, 1))
        label = function(example)

        prediction, activations = forward_pass(model, example)

        loss = mean_squared_error(prediction, label)
        if np.isnan(loss) or np.isinf(loss):
            break

        right += np.sum(np.round(prediction) == label)
        total += 1
        print("Accuracy: %.4f" % (right / total))

        delta = np.matrix(prediction - label)

        for level in range(len(model)-1, -1, -1):
            A, b = model[level]

            x = activations[level]["input"]
            y = activations[level]["output"]

            # The gradient with respect to function.
            delta = np.multiply(delta, dsigmoid(y))

            # Update weights.
            A -= STEP_SIZE * delta.dot(x.T)
            b -= STEP_SIZE * delta

            model[level] = [A, b]

            # The gradient with respect to the input.
            delta = np.dot(A.T, delta)


if __name__ == "__main__":
    model = neural_net(2, 1, 1)

    xor = lambda x: x[0] ^ x[1]
    train(xor, model)
