import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # compute subgradient gradient vector for MAE
    # ***************************************************
    error = y - tx @ w
    N = y.shape[0]
    subg_func_mae = lambda e: 1 if e > 0 else (0 if e == 0 else -1)
    subgradient = -1 / N * tx.transpose() @ np.array(list(map(subg_func_mae, error)))

    return subgradient
