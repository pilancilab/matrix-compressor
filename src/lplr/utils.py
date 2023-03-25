import numpy as np


def normalize_and_shift_wrt_inner_prod(X: np.ndarray = None, Y: np.ndarray = None):
    """
    :param X: Target matrix
    :param Y: Approximation matrix
    :return: Best approximation of X up to scaling and shift of Y
    """

    assert X.shape == Y.shape, "Dimension mismatch!"

    n = X.shape[0]
    d = X.shape[1]
    M = np.ones_like(X)

    t1 = np.linalg.norm(Y, ord="fro") ** 2 - np.sum(np.multiply(Y, M)) ** 2 / (n * d)
    t2 = np.sum(np.multiply(X, Y)) - np.sum(np.multiply(Y, M)) * np.sum(
        np.multiply(X, M)
    ) / (n * d)
    alpha = t2 / t1

    beta = (np.sum(np.multiply(X, M)) - np.sum(np.multiply(Y, M)) * alpha) / (n * d)

    return alpha * Y + beta * M


def error(target, estimated):
    tfro = np.linalg.norm(target, "fro")
    return np.linalg.norm(estimated - target, "fro") / tfro
