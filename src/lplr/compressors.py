import numpy as np
import torch
import math
from .quantizers import quantize
from .utils import normalize_and_shift_wrt_inner_prod
from . import SEED
from loguru import logger


def uncompressed(X: torch.Tensor = None) -> torch.Tensor:
    """No compressor"""
    return X


def lplr(
    X: torch.Tensor = None, r: int = None, B1: int = 8, B2: int = 8
) -> torch.Tensor:
    """
    :param X: Target matrix
    :param r: Inherent rank
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    logger.trace(f"Using seed = {SEED}")
    torch.random.manual_seed(SEED)

    # Sketch the column space of X with S matrix
    S = torch.randn(X.shape[1], r, device=X.device, dtype=X.dtype) / math.sqrt(r)  # Gaussian sketching matrix

    logger.debug(f"S.dtype = {S.dtype}, X.dtype={X.dtype}")

    # Quantize the sketched matrix and get the first low-rank factor
    logger.trace(f"X.dtype = {X.dtype}")
    Z = quantize(X=X @ S, B=B1)
    logger.trace(f"Z.dtype = {Z.dtype}")

    # Get the second low-rank factor
    W = torch.linalg.pinv(Z.float()).type(X.dtype) @ X

    W = quantize(W, B=B2)

    # Return the scaled and shifted output
    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def direct_svd_quant(
    X: torch.Tensor, r: int = None, B1: int = 8, B2: int = 8
) -> torch.Tensor:
    """
    Compute the full SVD and naively quantize each low rank factor
    :param X: Target matrix
    :param r: Target rank (intrinsic rank)
    :param B1: Bit-budget for the first low-rank factor
    :param B2: Bit-budget for the second low-rank factor
    :return: A (naive) low-precision low-rank approximation of X
    """
    torch.random.manual_seed(SEED)

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)
    U = U[:, 0:r]
    S = S[0:r]
    VT = VT[0:r, :]

    # Normalize and quantize the first low-rank factor
    Z = U @ torch.diag(S)
    Z = quantize(Z, B=B1)

    # Normalize and quantize the second low-rank factor
    W = VT
    W = quantize(W, B=B2)

    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def iterative_lplr(
    X: torch.Tensor = None, r: int = None, K: int = None, B1: int = 8, B2: int = 8
):
    """
    :param X: Target matrix
    :param r: Target rank (per-iteration)
    :param K: Total number of iterations (Target rank = r * K)
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    Xres = torch.copy(X)
    X_app = torch.zeros_like(X, device=X.device)

    for k in range(K):
        Xq = lplr(X=Xres, r=r, B1=B1, B2=B2)
        X_app += Xq
        Xres -= Xq

    return normalize_and_shift_wrt_inner_prod(X, X_app)
