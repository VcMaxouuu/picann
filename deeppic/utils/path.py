r"""Regularisation paths.

A phased fit does not jump straight to the calibrated :math:`\lambda`: it walks
up to it, each phase starting from the weights the previous one reached. Early
phases are barely penalised and let the network find its shape, later ones prune
what the level no longer supports.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["geometric_path"]


def geometric_path(
    lambda_max: Tensor | float,
    n_phases: int,
    ratio: float = 1e-2,
) -> Tensor:
    r"""Build the increasing geometric grid of levels a phased fit walks.

    The grid runs from :math:`\mathrm{ratio} \times \lambda_{\max}` to
    :math:`\lambda_{\max}`, in equal ratios. Geometric rather than linear because
    what a penalty does to the weights depends on the order of magnitude of the
    level, not on its absolute value.

    :param lambda_max: level the last phase runs at, usually the calibrated one.
    :param n_phases: number of levels; ``1`` returns ``lambda_max`` alone.
    :param ratio: fraction of ``lambda_max`` the first phase runs at.
    :return: levels in increasing order, of shape ``(n_phases,)``, ending at
        ``lambda_max``.
    :raises ValueError: if ``n_phases`` is not positive, or if ``ratio`` does not
        lie in :math:`(0, 1]`.
    """
    if n_phases < 1:
        raise ValueError(f"n_phases must be positive, got {n_phases}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must lie in (0, 1], got {ratio}")

    levels = torch.as_tensor(lambda_max, dtype=torch.get_default_dtype())
    if n_phases == 1:
        return levels.reshape(1)

    exponents = torch.linspace(1.0, 0.0, n_phases, dtype=levels.dtype)
    return levels * ratio**exponents
