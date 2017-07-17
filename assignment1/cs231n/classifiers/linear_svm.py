import numpy as np
from random import shuffle
from past.builtins import xrange


EPSILON = 1e-8


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)

    y_hat = np.dot(X, W)

    loss = 0.0

    for i in range(y_hat.shape[0]):
        for j in range(y_hat.shape[1]):
            if y[i] == j:
                continue

            delta = y_hat[i,j] - y_hat[i,y[j]] + 1.0

            if delta < EPSILON:
                continue

            loss += delta

            dW[:,j] += X[i,:]
            dW[:,y[j]] -= X[i,:]

    dW /= X.shape[0]

    loss += reg * np.sum(W * W)
    loss /= X.shape[0]

    return loss, dW
