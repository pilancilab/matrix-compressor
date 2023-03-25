import numpy as np
from .quantizers import quantize
from .utils import normalize_and_shift_wrt_inner_prod
from . import SEED


def uncompressed(X: np.ndarray = None) -> np.ndarray:
    """No compressor"""
    return X


def lplr(X: np.ndarray = None, r: int = None, B1: int = 8, B2: int = 8) -> np.ndarray:
    """
    :param X: Target matrix
    :param r: Inherent rank
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """
    np.random.seed(SEED)

    # Sketch the column space of X with S matrix
    S = np.random.randn(X.shape[1], r) / np.sqrt(r)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = quantize(X=X @ S, B=B1)

    # Get the second low-rank factor
    W = np.linalg.pinv(Z) @ X
    W = quantize(W, B=B2)

    # Return the scaled and shifted output
    return normalize_and_shift_wrt_inner_prod(X, Z @ W)
    # return Z @ W


def direct_svd_quant(
    X: np.ndarray = None, r: int = None, B1: int = 8, B2: int = 8
) -> np.ndarray:
    """
    Compute the full SVD and naively quantize each low rank factor
    :param X: Target matrix
    :param r: Target rank (intrinsic rank)
    :param B1: Bit-budget for the first low-rank factor
    :param B2: Bit-budget for the second low-rank factor
    :return: A (naive) low-precision low-rank approximation of X
    """
    np.random.seed(SEED)

    # Compute full SVD
    U, S, VT = np.linalg.svd(X.astype(float), full_matrices=False)
    U = U[:, 0:r]
    S = S[0:r]
    VT = VT[0:r, :]

    # Normalize and quantize the first low-rank factor
    Z = U @ np.diag(S)
    Z = quantize(Z, B=B1)

    # Normalize and quantize the second low-rank factor
    W = VT
    W = quantize(W, B=B2)

    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def iterative_lplr(
    X: np.ndarray = None, r: int = None, K: int = None, B1: int = 8, B2: int = 8
):
    """
    :param X: Target matrix
    :param r: Target rank (per-iteration)
    :param K: Total number of iterations (Target rank = r * K)
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    Xres = np.copy(X)
    X_app = np.zeros_like(X)

    for k in range(K):
        Xq = lplr(X=Xres, r=r, B1=B1, B2=B2)
        X_app += Xq
        Xres -= Xq

    return normalize_and_shift_wrt_inner_prod(X, X_app)
