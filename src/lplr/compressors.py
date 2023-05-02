import numpy as np
import torch
import math
from .quantizers import quantize
from .utils import normalize_and_shift_wrt_inner_prod
from loguru import logger


def uncompressed(X: torch.Tensor = None) -> torch.Tensor:
    """No compressor"""
    return X


def lplr(
    X: torch.Tensor = None,
    r: int = None,
    B1: int = 8,
    B2: int = 8,
    normalize_and_shift=False,
    seed=None,
    q = 0,
) -> torch.Tensor:
    """
    :param X: Target matrix
    :param r: Inherent rank
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :param q: No. of power iterations
    :return: Low-precision Low-rank approximation
    """
    if seed is not None:
        logger.trace(f"Using seed = {seed}")
        torch.random.manual_seed(seed)
    orig_dtype = X.dtype
    X = X.float()
    # Sketch the column space of X with S matrix
    S = torch.randn(X.shape[1], r, device=X.device, dtype=X.dtype) / math.sqrt(r)  # Gaussian sketching matrix

    logger.trace(f"S.dtype = {S.dtype}, X.dtype={X.dtype}")

    # Quantize the sketched matrix and get the first low-rank factor
    logger.trace(f"X.dtype = {X.dtype}")

    Y = X @ S
    for k in range(q):
        print(f"Running power iteration: {k}/{q}")
        Y = X @ (X.T @ Y)

    Z = quantize(X=Y, B=B1, preserve_original_dtype=True)
    logger.trace(f"Z.dtype = {Z.dtype}")

    if torch.isnan(Z).any().item():
        logger.error(f"NaNs encountered in right sketched matrix")

    # Get the second low-rank factor
    pinv = torch.linalg.pinv(Z.float()).type(X.dtype)
    if torch.isnan(pinv).any().item():
        logger.error(f"NaNs encountered in pinv")

    W = pinv @ X

    if torch.isnan(W).any().item():
        logger.error(f"NaNs encountered in pinv @ X")

    W = quantize(W, B=B2, preserve_original_dtype=True)

    if torch.isnan(W).any().item():
        logger.error(f"NaNs encountered in Q(pinv @ X)")
    # Return the scaled and shifted output

    out = Z @ W
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, Z @ W)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")
    out = out.type(orig_dtype)
    return out


def direct_svd_quant(
    X: torch.Tensor, r: int = None, B1: int = 8, B2: int = 8, f: float = None, eps=1e-5
) -> torch.Tensor:
    """
    Compute the full SVD and naively quantize each low rank factor
    :param X: Target matrix
    :param r: Target rank (intrinsic rank)
    :param B1: Bit-budget for the first low-rank factor
    :param B2: Bit-budget for the second low-rank factor
    :return: A (naive) low-precision low-rank approximation of X
    """

    if r is None and f is None:
        raise ValueError("Atleast one of target rank or fraction must be provided.")

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)
    if r is None:
        rank = torch.sum(S >= eps)
        r = int(torch.ceil(f * rank).item())
        logger.info(
            f"Quantizing to rank = {r} using fraction = {f:.3f} with original rank = {rank}"
        )
    U = U[:, 0:r]
    S = S[0:r]
    VT = VT[0:r, :]

    # Normalize and quantize the first low-rank factor
    Z = U @ torch.diag(S)
    Z = quantize(Z, B=B1)

    # Normalize and quantize the second low-rank factor
    W = VT
    W = quantize(W, B=B2)

    # return normalize_and_shift_wrt_inner_prod(X, Z @ W)
    return Z @ W


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
