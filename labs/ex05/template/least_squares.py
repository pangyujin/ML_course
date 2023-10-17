# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx @ w
    # reshape
    e = e.reshape(e.shape[0],)

    mse = e.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # least squares
    # returns mse, and optimal weights
    # ***************************************************
    optimal_weights = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, optimal_weights)
    
    return optimal_weights, mse
