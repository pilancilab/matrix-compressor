import numpy as np

def quantize(X: np.ndarray = None, B: int = 16):
    """
    Element-wise matrix quantization for general bit-budget
    :param X: Matrix to be quantized
    :param B: Bit-budget per coordinate for quantization
    :return: Quantized matrix
    """

    M = 2**B  # No. of quantization points per dimension
    res = 2 / (M - 1)  # Resolution

    # Normalize the coordinates of the quantizer input to [-1,1]
    X_min = X.min()
    X_max = X.max()
    X = np.interp(X, (X_min, X_max), (-1, 1))

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    fst_pt = -1  # First quantization point
    L_idx = np.floor((X - fst_pt) / res)  # Lower index for each entry
    L = fst_pt + L_idx * res  # Matrix of lower quantization points
    U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

    # Nearest neighbor quantization
    Q = np.zeros_like(X)
    Q[X < -1] = -1  # Value less than lower limit of dynamic range
    Q[X > 1] = 1  # Value more than upper limit of dynamic range
    mask0 = np.abs(X) <= 1  # Value within dynamic range
    mask = np.abs(X - L) <= res / 2
    Q[mask * mask0] = L[mask * mask0]
    mask = np.abs(U - X) <= res / 2
    Q[mask * mask0] = U[mask * mask0]

    # Re-normalize the quantized matrix back to its input scale
    return np.interp(Q, (Q.min(), Q.max()), (X_min, X_max))
