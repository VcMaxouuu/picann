r"""Calibration of the regularisation level.

The prescribed :math:`\lambda` is the level at which the penalty dominates the
loss when no feature carries signal. It is estimated by simulation: draw
:math:`M` target vectors under :math:`H_0`, compute the gradient each of them
induces on the linear coefficients, and take the :math:`(1 - \alpha)` empirical
quantile of the sup-norms

.. math::
    \lambda = Q_{1 - \alpha}
    \left( \| \nabla_\beta \|_\infty^{(1)}, \dots,
    \| \nabla_\beta \|_\infty^{(M)} \right).

A coefficient then enters the model only if its gradient exceeds what pure
noise produces, at level :math:`\alpha`.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor

from deeppic.loss.loss import Loss

__all__ = ["calibrate_lambda"]


def calibrate_lambda(
    loss: Loss,
    X: Tensor,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    batch_size: int | None = None,
    generator: Generator | None = None,
) -> Tensor:
    r"""Compute the level :math:`\lambda` at which the penalty dominates the loss
    when no feature carries signal.

    Draws are independent, so they are evaluated in batches: one batch holds
    intermediates of shape ``(batch_size, n)``, which is what bounds the memory.

    :param loss: loss whose null distribution is simulated.
    :param X: design matrix, of shape ``(n, p)``, standardised to columns of
        zero mean and unit variance.
    :param alpha: level of the quantile; :math:`\lambda` dominates the null
        gradient in a proportion :math:`1 - \alpha` of the draws.
    :param n_simulations: number of Monte-Carlo draws :math:`M`.
    :param batch_size: number of draws evaluated at once; ``None`` evaluates them
        all in a single batch.
    :param generator: pseudo-random generator, for reproducible draws.
    :return: prescribed regularisation level, as a 0-dim tensor.
    :raises ValueError: if ``alpha`` does not lie in :math:`(0, 1)`, or if
        ``n_simulations`` or ``batch_size`` is not positive.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n = X.shape[0]
    step = n_simulations if batch_size is None else batch_size
    norms = torch.empty(n_simulations, dtype=X.dtype, device=X.device)
    for start in range(0, n_simulations, step):
        batch = min(step, n_simulations - start)
        y = loss.sample_null(batch * n, generator=generator).to(X).view(batch, n)
        norms[start : start + batch] = loss.null_gradient(X, y).abs().amax(dim=-1)
    return torch.quantile(norms, 1.0 - alpha)
