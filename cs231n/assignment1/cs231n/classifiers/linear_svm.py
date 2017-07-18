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

            delta = y_hat[i,j] - y_hat[i,y[i]] + 1.0

            if delta < EPSILON:
                continue

            loss += delta

            dW[:,j] += X[i,:]
            dW[:,y[i]] -= X[i,:]

    dW /= X.shape[0]
    dW += reg * W

    loss /= X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    n = y.shape[0]
    m = W.shape[1]

    y_hat = np.dot(X, W)
    y_star = np.expand_dims(y_hat[np.arange(n),y], axis=1)

    mask = (y_hat != y_star)
    delta = np.maximum(0.0, y_hat - y_star + 1.0)

    loss = 0.0
    loss += np.sum(mask * delta)
    loss /= n
    loss += 0.5 * reg * np.sum(W * W)

    # dW = np.zeros(W.shape)
    # dW += np.dot(X.T, (mask * delta) > EPSILON)
    # dW /= n
    # dW += reg * W

    return loss, dW
