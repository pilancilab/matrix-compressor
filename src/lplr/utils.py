import numpy as np
import torch
from loguru import logger

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
    logger.trace(f"t1 = {t1:.5f}")
    t2 = torch.sum(torch.multiply(X, Y)) - torch.sum(torch.multiply(Y, M)) * torch.sum(
        torch.multiply(X, M)
    ) / (n * d)
    
    logger.trace(f"t2 = {t2:.5f}")
    alpha = t2 / t1
    logger.trace(f"alpha = {alpha:.5f}")
    beta = (
        torch.sum(torch.multiply(X, M)) - torch.sum(torch.multiply(Y, M)) * alpha
    ) / (n * d)
    logger.trace(f"beta = {beta:.5f}")
    return alpha * Y + beta * M



