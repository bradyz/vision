import numpy as np


STEP_SIZE = 1e-2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return np.multiply(sigmoid(x), (1.0 - sigmoid(x)))


def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


def neural_net(n_inputs, n_hidden, n_outputs, n_units=10, weight=1e-2):
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
    activations = list()

    x = example

    for i in range(len(model)):
        A, b = model[i]

        # Last layer is linear regression.
        if i == len(model) - 1:
            activations.append({"input": x,
                                "output": np.dot(A, x) + b})
        else:
            activations.append({"input": x,
                                "output": sigmoid(np.dot(A, x) + b)})

        x = activations[-1]["output"]

    return activations[-1]["output"], activations


def train(function, model, n_iterations=10000):
    for _ in range(n_iterations):
        example = np.random.rand(1, 1)
        label = function(example)

        prediction, activations = forward_pass(model, example)
        loss = mean_squared_error(prediction, label)

        print("Loss: %s" % loss)
        print("Prediction: %s" % (prediction[0][0]))
        print("Label: %s" % label[0][0])

        if np.isnan(loss) or np.isinf(loss):
            break

        delta = np.matrix(loss)

        for level in range(len(model)-1, -1, -1):
            A, b = model[level]
            x = activations[level]["input"]

            # Update weights.
            A = A - STEP_SIZE * delta.dot(x.T)
            b = b - STEP_SIZE * delta

            # Update gradient. Last layer does not have an activation.
            if level == len(model) - 1:
                delta = np.multiply(A.T.dot(delta), x)
            else:
                delta = np.multiply(A.T.dot(delta), dsigmoid(x))

            model[level] = [A, b]


if __name__ == "__main__":
    model = neural_net(1, 2, 1)

    hard_function = lambda x: np.sin(x)
    train(hard_function, model)
