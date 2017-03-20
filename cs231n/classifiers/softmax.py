import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        f_i = X[i].dot(W)
        # Normalization trick to avoid numerical instability
        # Shift the values of so that the highest number is 0
        f_i -= np.max(f_i)

        sum_i = np.sum(np.exp(f_i))
        loss += -f_i[y[i]] + np.log(sum_i)

        for j in xrange(num_classes):
            p = np.exp(f_i[j])/sum_i
            dW[:, j] += (p - (j == y[i])) * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)

    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f) / sum_f
    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW
