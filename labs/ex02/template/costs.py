# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # compute loss by MSE
    # ***************************************************
    N = y.shape[0]
    return  1 / (2*N) * ((y - tx @w) ** 2).sum()

def compute_loss_mae(y, tx, w):
    error = y - tx @ w
    return abs(error)
