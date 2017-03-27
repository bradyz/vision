import numpy as np


N_NODES = 100
N_BATCHES = 10000
N_BATCH_SIZE = 32
STEP_SIZE = 1e-6


def mean_squared_error(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


def neural_net(n_inputs, n_hidden, n_outputs):
    layers = list()

    A_1 = np.random.rand(N_NODES, n_inputs) * 1e-2
    b_1 = np.random.rand(N_NODES, 1) * 1e-2
    layers.append([A_1, b_1])

    for i in range(n_hidden):
        A_i = np.random.rand(N_NODES, N_NODES) * 1e-2
        b_i = np.random.rand(N_NODES, 1) * 1e-2
        layers.append([A_i, b_i])

    A_2 = np.random.rand(n_outputs, N_NODES) * 1e-2
    b_2 = np.random.rand(n_outputs, 1) * 1e-2
    layers.append([A_2, b_2])

    return layers


def forward_pass(model, examples):
    predictions = np.zeros(examples.shape)
    activations = list()

    for i in range(examples.shape[0]):
        activation = list()
        x = np.matrix(examples[i])

        for layer in model:
            A, b = layer

            activation.append({"input": x, "output": np.dot(A, x) + b})
            x = activation[-1]["output"]

        predictions[i] = x
        activations.append(activation)

    return predictions, activations


def train(function, model):
    for i in range(N_BATCHES):
        examples = np.random.rand(N_BATCH_SIZE, 1) * 10
        labels = function(examples)

        predictions, activations = forward_pass(model, examples)

        loss = mean_squared_error(predictions, labels)

        gradient = np.matrix(loss)

        for level in range(len(model)-1, -1, -1):
            A, b = model[level]

            x_ = activations[0][level]["input"]
            for i in range(1, N_BATCH_SIZE):
                x_ += activations[i][level]["input"]
            x_ = np.matrix(np.mean(x_))

            dA = gradient.T.dot(x_)
            db = gradient.T

            # import pdb; pdb.set_trace()

            # Update gradient.
            gradient = gradient.dot(A)

            A = A - STEP_SIZE * dA
            b = b - STEP_SIZE * db

            # Update weights.
            model[level] = [A, b]

        print(loss)
        if loss == float('inf'):
            break


if __name__ == "__main__":
    easy_function = lambda x: x * 42 + 8
    hard_function = lambda x: np.sin(x)
    model = neural_net(1, 3, 1)

    train(easy_function, model)
