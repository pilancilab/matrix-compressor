import numpy as np
import torch


def normalize_and_shift_wrt_inner_prod(X: torch.Tensor = None, Y: torch.Tensor = None):
    """
    :param X: Target matrix
    :param Y: Approximation matrix
    :return: Best approximation of X up to scaling and shift of Y
    """

    assert X.shape == Y.shape, "Dimension mismatch!"

    n = X.shape[0]
    d = X.shape[1]
    M = torch.ones_like(X)

    t1 = torch.linalg.norm(Y, ord="fro") ** 2 - torch.sum(torch.multiply(Y, M)) ** 2 / (
        n * d
    )
    t2 = torch.sum(torch.multiply(X, Y)) - torch.sum(torch.multiply(Y, M)) * torch.sum(
        torch.multiply(X, M)
    ) / (n * d)
    alpha = t2 / t1

    beta = (
        torch.sum(torch.multiply(X, M)) - torch.sum(torch.multiply(Y, M)) * alpha
    ) / (n * d)

    return alpha * Y + beta * M


