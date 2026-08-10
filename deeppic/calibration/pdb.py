from __future__ import annotations

import torch
from torch import Tensor

from ..optim import Loss

__all__ = ["monte_carlo_null", "lambda_pdb"]

#: Elements per matrix-product block, chosen so a chunk stays well inside cache/VRAM.
_BLOCK_ELEMENTS = 1 << 23


@torch.no_grad()
def monte_carlo_null(
    X: Tensor,
    loss: Loss,
    n_mc: int = 1_000,
    generator: "torch.Generator | None" = None,
) -> Tensor:
    r"""Draw the null sample of the zero-thresholding statistic, shape ``(n_mc,)``.

    Parameters
    ----------
    X : Tensor
        Design matrix, shape ``(n, p)``.  Its columns should be standardised.
    loss : Loss
        Supplies both the null model (:meth:`~Loss.sample_null`) and the
        closed-form statistic (:meth:`~Loss.zero_thresholding`).
    n_mc : int, default 1000
        Number of Monte-Carlo replicates.
    generator : torch.Generator, optional
        For reproducibility.

    Returns
    -------
    Tensor
        The `n_mc` samples of the zero-thresholding statistic
    """
    n, p = X.shape
    block = max(1, min(n_mc, _BLOCK_ELEMENTS // max(n, p, 1)))

    # A generator is bound to a device, so draw where it lives and move the
    # sample afterwards. The null stream is then identical whatever device
    # the statistic is evaluated on, which keeps a seeded run reproducible
    # across CPU, CUDA and MPS.
    draw_device = X.device if generator is None else generator.device

    out = []
    drawn = 0
    while drawn < n_mc:
        m = min(block, n_mc - drawn)
        Z = loss.sample_null(
            n, m, generator=generator, device=draw_device, dtype=X.dtype
        )
        out.append(loss.zero_thresholding(X, Z.to(X.device)))
        drawn += m
    return torch.cat(out)


@torch.no_grad()
def lambda_pdb(
    X: Tensor,
    loss: Loss,
    alpha: float = 0.05,
    n_mc: int = 1_000,
    generator: "torch.Generator | None" = None,
) -> float:
    r"""The calibrated threshold :math:`\lambda^{\rm PDB} = q_{1-\alpha}(\Lambda)`.

    Using `\lambda^{\rm PDB}` as regularization parameter, ensures that under
    :math:`H_0` no feature is selected with probability :math:`1-\alpha`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha must lie strictly between 0 and 1, got {alpha}. It is the "
            "probability of selecting at least one feature under the pure-noise model."
        )
    Lambda = monte_carlo_null(X, loss, n_mc=n_mc, generator=generator)
    # On the host in double precision: the sample is small, the quantile is
    # a reduction where accuracy matters, and accelerators such as MPS do
    # not support float64 at all.
    return float(torch.quantile(Lambda.cpu().double(), 1.0 - alpha))
