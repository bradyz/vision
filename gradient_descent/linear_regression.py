from matplotlib import pyplot as plt
import numpy as np


NB_SAMPLES = 1000
NB_BATCHES = 10000
NB_DIMENSIONS = 20


def generate_training_data(nb_samples, nb_dimensions, beta=7.5):
    # True parameters.
    seed = np.random.randn(nb_dimensions)

    # Artificial data.
    examples = np.random.randn(nb_samples, nb_dimensions) * 10
    labels = np.zeros(nb_samples)

    # Generate labels with random noise.
    for i in range(nb_samples):
        labels[i] = examples[i].dot(seed) + np.random.randn() * beta

    return examples, labels, seed


def train(examples, labels, batch_size=8, plot=False):
    # Plotting only works for 1-dimensional regression.
    if plot:
        assert examples.shape[1] == 1, "Can't view multiple dimensions."

        plt.ion()
        plt.show(block=False)

    # Samples, features.
    m, n = examples.shape
    losses = list()

    # Model weights.
    theta = np.random.randn(n)

    offset = 0
    for i in range(NB_BATCHES):
        # Examples, labels, predictions.
        x = np.zeros((batch_size, n))
        y = np.zeros((batch_size))
        y_ = np.zeros(batch_size)

        # Collect the training batch.
        for j in range(batch_size):
            x[j] = examples[offset]
            y[j] = labels[offset]
            offset = (offset + 1) % m

            # Shuffle if the dataset is exhausted.
            if offset == 0:
                permutation = np.random.permutation(m)
                examples = examples[permutation]
                labels = labels[permutation]

        # Compute predictions.
        for j in range(batch_size):
            y_[j] = theta.dot(x[j])

        # Checkpoint losses.
        if i % 100 == 0:
            losses.append(np.mean(np.square(y_ - y)))
            print("Batch: ", i, "Loss: ", losses[-1])

            if plot:
                # Generate line of best fit given the estimator theta.
                sample_x = np.linspace(np.min(examples), np.max(examples), 100)
                sample_y = [sample * theta for sample in sample_x]

                # Clear all.
                plt.clf()

                # Plot the line of best fit.
                plt.subplot(211)
                plt.title("Training Data")
                plt.xlim([np.min(examples), np.max(examples)])
                plt.ylim([np.min(labels), np.max(labels)])
                plt.plot(examples, labels, "r.")
                plt.plot(sample_x, sample_y, "b-")

                # Print batch loss.
                plt.subplot(212)
                plt.title("Batch Loss (Mean Squared Error)")
                plt.plot(losses, "c-")

                # Something required for plt.ion().
                plt.pause(0.01)

        # Calculate the gradient.
        dtheta = np.zeros(n)
        for j in range(batch_size):
            # Gradient with respect to theta.
            dtheta += 1 / batch_size * (theta.dot(x[j]) - y[j]) * x[j]

        # Update parameters.
        theta = theta - 1e-5 * dtheta

    if plot:
        while True:
            plt.pause(0.01)

    return theta


if __name__ == "__main__":
    # Higher dimensional.
    examples, labels, _ = generate_training_data(NB_SAMPLES, NB_DIMENSIONS)
    train(examples, labels)

    # Something we can visualize.
    examples, labels, _ = generate_training_data(NB_SAMPLES, 1)
    train(examples, labels, plot=True)
