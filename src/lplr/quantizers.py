import numpy as np
import torch
from loguru import logger


def quantize(X: torch.Tensor = None, B: int = 16) -> torch.Tensor:
    """
    Element-wise matrix quantization for general bit-budget
    :param X: Matrix to be quantized
    :param B: Bit-budget per coordinate for quantization
    :return: Quantized matrix
    """

    M = 2**B  # No. of quantization points per dimension
    res = 2 / (M - 1)  # Resolution

    # Normalize the coordinates of the quantizer input to [-1,1]
    X_min = X.min().item()
    X_max = X.max().item()
    orig_dtype = X.dtype
    # logger.trace(f"X_min = {X_min}, X_min.dtype = {X_min.dtype}, X_min.device = {X_min.device}")
    # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")
    X = torch.from_numpy(np.interp(X.to("cpu").numpy(), (X_min, X_max), (-1, 1))).to(X.device)
    # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    fst_pt = -1  # First quantization point
    L_idx = torch.floor((X - fst_pt) / res)  # Lower index for each entry
    L = fst_pt + L_idx * res  # Matrix of lower quantization points
    U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

    # Nearest neighbor quantization
    Q = torch.zeros_like(X)
    Q[X < -1] = -1  # Value less than lower limit of dynamic range
    Q[X > 1] = 1  # Value more than upper limit of dynamic range
    mask0 = torch.abs(X) <= 1  # Value within dynamic range
    mask = torch.abs(X - L) <= res / 2
    Q[mask * mask0] = L[mask * mask0]
    mask = torch.abs(U - X) <= res / 2
    Q[mask * mask0] = U[mask * mask0]

    # Re-normalize the quantized matrix back to its input scale
    Qr = torch.from_numpy(np.interp(Q.to("cpu").numpy(), (Q.min().item(), Q.max().item()), (X_min, X_max))).to(X.device).type(orig_dtype)

    logger.trace(f"Qr.dtype = {Qr.dtype}, Qr.device = {Qr.device}")

    return Qr