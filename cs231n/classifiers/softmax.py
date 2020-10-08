from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # 让最大值为零，避免指数爆炸
        correct_class_expscore = np.exp(scores[y[i]])
        total_expscore = np.sum(np.exp(scores))
        loss += - np.log(correct_class_expscore / total_expscore)
        for j in range(num_classes):
            i_j_element_expscore = np.exp(scores[j])
            if j != y[i]:
                dW[:, j] += X[i] * i_j_element_expscore / total_expscore
        dW[:, y[i]] += - X[i] + X[i] * correct_class_expscore / \
            total_expscore  # 在一行中只有一个正确的列，因此可以放在i循环中

    '''
    scores = X.dot(W)
    for i in range(num_train):
        f = scores[i] - np.max(scores[i])  # avoid numerical instability
        softmax = np.exp(f)/np.sum(np.exp(f))
        loss += -np.log(softmax[y[i]])
        # Weight Gradients
        for j in range(num_classes):
            dW[:, j] += X[i] * softmax[j]
        dW[:, y[i]] -= X[i] #把所有的都放入softmax项，再在y[i]项减去X[i]'''

    loss /= num_train
    loss += reg * np.sum(W**2)

    dW /= num_train
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_class_expscore = np.reshape(np.exp(
        scores[np.arange(num_train), y[np.arange(num_train)]]), [num_train, 1])
    # reshap和keepdims保证（别人的）梯度代码可用
    total_expscore = np.sum(np.exp(scores), axis=1, keepdims=True)

    loss = np.sum(-np.log(correct_class_expscore/total_expscore))
    loss /= num_train
    loss += reg * np.sum(W**2)

    # Weight Gradient
    softmax_matrix = np.exp(scores) / total_expscore
    softmax_matrix[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax_matrix) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
