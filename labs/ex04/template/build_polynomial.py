# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # ***************************************************
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    ans = np.ones(x.shape[0]).reshape(x.shape[0], 1)
    for i in range(degree):
        column_i = (x ** (i+1)).reshape(x.shape[0], 1)
        ans = np.append(ans, column_i, axis=1)
    return ans
