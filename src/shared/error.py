from loguru import logger
import torch


def relative_tensor_error(target: torch.Tensor, estimated: torch.Tensor, ord=None) -> float:
    if target.ndim == 2:
        ord = "fro"
    t_norm = torch.linalg.norm(target, ord)
    if t_norm < 1e-5:
        logger.error(f"Low target norm: {t_norm:.6f}, division by zero may cause nans")
    diff_norm = torch.linalg.norm(estimated - target, ord)
    res = diff_norm / t_norm
    if torch.isnan(res):
        logger.error(
            f"NaN obtained for delta norm {diff_norm:.5f}"
            f", target norm: {t_norm:.5f}"
            f", estimate norm: {torch.linalg.norm(estimated, 'fro'):.5f}"
        )
    return res
