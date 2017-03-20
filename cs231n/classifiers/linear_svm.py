import numpy as np
from random import shuffle


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        diff_count = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            if j != y[i]:
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                if margin > 0:
                    loss += margin
                    diff_count += 1
                    # Gradient update for incorrect rows
                    dW[:, j] += X[i]
        # Gradient update for correct row
        dW[:, y[i]] -= diff_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # same with loss
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Gradient regularization that carries through per
    # https://piazza.com/class/i37qi08h43qfv?cid=118
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # Implement a vectorized version of the structured SVM loss, storing the
    # result in loss.
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1) # delta = 1
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train # get mean
    loss += 0.5 * reg * np.sum(W * W) # regularization


    # Implement a vectorized version of the gradient for the structured SVM
    # loss, storing the result in dW.
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1

    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] = -incorrect_counts

    dW = X.T.dot(X_mask) / num_train # average out weights
    dW += reg * W # regularize the weights

    return loss, dW
