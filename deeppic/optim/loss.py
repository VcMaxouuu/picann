from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

_TINY = 1e-12

__all__ = ["Loss", "SqrtMSELoss", "ExpBinaryLoss", "CoxPHLoss"]


def _logaddexp(a: Tensor, b: Tensor) -> Tensor:
    r""":math:`\log(e^a + e^b)`, evaluated around the larger term."""
    return torch.maximum(a, b) + torch.log1p(torch.exp(-(a - b).abs()))


def _logcumsumexp(x: Tensor) -> Tensor:
    r"""Stable :math:`\log \sum_{i \le k} e^{x_i}` for a vector ``x``.

    Written from primitive operations rather than
    :func:`torch.logcumsumexp`, which several accelerator backends do not
    implement (MPS, as of torch 2.4).

    Doubling scan: after step :math:`d` entry :math:`k` holds the log-sum of
    the ``2d`` terms ending at :math:`k`, so :math:`\lceil \log_2 n \rceil`
    steps cover every prefix. Each combination is done around its own larger
    term, which keeps the result accurate however wide the spread of ``x``;
    shifting by a single global maximum instead would lose an early prefix
    entirely once the spread exceeds the exponential range (~88 in float32).
    The pad value is the smallest finite number of the dtype, a neutral
    element here: adding it back changes nothing and cannot produce a NaN.
    """
    padding = torch.finfo(x.dtype).min
    out = x
    step = 1
    while step < x.shape[0]:
        shifted = torch.cat([x.new_full((step,), padding), out[:-step]])
        out = _logaddexp(out, shifted)
        step *= 2
    return out

class Loss(nn.Module, ABC):
    r"""
    Base class for data-fidelity objective functions.

    Represents the objective term :math:`D(\mu(x), y)`, measuring the
    discrepancy between the model prediction :math:`\mu(x)` and the
    observed data :math:`y`.

    Lower values correspond to better agreement, with the minimum attained
    when :math:`\mu(x)` best matches :math:`y`.
    """

    @abstractmethod
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """f(output, target), a scalar. Gradients come from autodiff."""

    @abstractmethod
    def zero_thresholding(self, X: Tensor, Y: Tensor) -> Tensor:
        r"""Closed-form gradient of the objective with respect to the first-layer
        parameters under the null hypothesis. If ``Y`` has shape ``(n, m)``,
        the gradient is computed independently for each of the ``m`` columns,
        yielding ``m`` gradients.
        """

    @abstractmethod
    def sample_null(
        self,
        n: int,
        M: int,
        generator: "torch.Generator | None" = None,
        *,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> Tensor:
        """Draw ``M`` independent responses of length ``n`` from the null model."""


    @abstractmethod
    def mu_null(self, y: Tensor) -> Tensor:
        r"""Returns the maximum-likelihood constant predictor
        :math:`\hat{\mu}_{\mathrm{MLE}}`.

        This is the constant value minimizing the data-fidelity objective
        over all constant predictors.
        """


class SqrtMSELoss(Loss):
    r"""Square-root MSE

    .. math::  f(\theta) = \sqrt{\tfrac1n \sum_n (y_n - \mu_\theta(x_n))^2}

    Parameters
    ----------
    eps : float, default 1e-8
        Floor on the mean squared error inside the square root.  At an interpolating
        fit the residual vanishes and :math:`\sqrt{R}` is non-smooth; the floor keeps
        the gradient finite.  It is not a statistical device and does not enter the
        calibration.
    """
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        mse = (target - output).pow(2).mean()
        return torch.sqrt(mse.clamp_min(self.eps**2))

    def zero_thresholding(self, X: Tensor, Y: Tensor) -> Tensor:
        r"""Computes the pivotal statistic

        .. math::

            \frac{\lVert X^\top \tilde{Y} \rVert_\infty}
                {\sqrt{n}\,\lVert \tilde{Y} \rVert_2},

        where :math:`\tilde{Y}` is obtained by centering each column of ``Y``.
        """
        n = X.shape[0]
        Y = Y.reshape(n, -1)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        num = (X.transpose(0, 1) @ Yc).abs().amax(dim=0)
        den = (n**0.5) * Yc.norm(dim=0).clamp_min(_TINY)
        return num / den

    def sample_null(
        self,
        n: int,
        M: int,
        generator: "torch.Generator | None" = None,
        *,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> Tensor:
        r"""Draw :math:`Z \sim N(0, I_n)`, shape ``(n, M)``.
        """
        return torch.randn(n, M, generator=generator, device=device, dtype=dtype)

    def mu_null(self, y: Tensor) -> Tensor:
        return y.mean()



class ExpBinaryLoss(Loss):
    r"""Weighted-score binary loss

    .. math:: f(\theta) = \sqrt{\tfrac2n \sum_n \exp\!\left(-\tfrac{s_i \mu_\theta(x_n)}{2}\right).

    with :math:`s_i = 2y_i-1\in\{-1,+1\}`.

    Parameters
    ----------
    clamp : float, default 30.0
        Upper clamp on the exponent guarding overflow.Beyond the clamp the
        gradient is exactly zero: a grossly misclassified point stops
        contributing rather than dominating.
    """
    def __init__(self, clamp: float = 30.0) -> None:
        super().__init__()
        self.clamp = float(clamp)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        s = 2.0 * target - 1.0
        return 2.0 * torch.exp((-0.5 * s * output).clamp(max=self.clamp)).mean()

    def zero_thresholding(self, X: Tensor, Y: Tensor) -> Tensor:
        r"""Computes the pivotal statistic

        .. math::

            \frac{\lVert X^\top \tilde{Y} \rVert_\infty}
                {n\sqrt{\bar Y(1-\bar Y)}},

        where :math:`\tilde{Y}` is obtained by centering each column of ``Y``.
        """
        n = X.shape[0]
        Y = Y.reshape(n, -1)
        pbar = Y.mean(dim=0, keepdim=True)
        Yc = Y - pbar
        num = (X.transpose(0, 1) @ Yc).abs().amax(dim=0)
        den = n * torch.sqrt((pbar * (1.0 - pbar)).clamp_min(_TINY)).squeeze(0)
        return num / den

    def sample_null(
        self,
        n: int,
        M: int,
        generator: "torch.Generator | None" = None,
        *,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> Tensor:
        r"""Draw :math:`Y_{im} \sim \mathrm{Bernoulli}(0.5)`, shape ``(n, M)``.
        """
        return torch.bernoulli(torch.full((n, M), 0.5, device=device, dtype=dtype), generator=generator)

    def mu_null(self, y: Tensor) -> Tensor:
        p = y.mean().clamp(_TINY, 1.0 - _TINY)
        return torch.log(p / (1.0 - p))



class CoxPHLoss(Loss):
    r"""Negative Cox partial log-likelihood, with Breslow handling of ties.

    .. math::
        f(\theta) = -\frac{1}{d} \sum_{i:\,\delta_i = 1}
            \left[ \mu_\theta(x_i)
            - \log\!\!\sum_{j:\,t_j \ge t_i} e^{\mu_\theta(x_j)} \right],

    where :math:`d` is the number of observed events. The target packs the
    follow-up times and the event indicators as a ``(n, 2)`` tensor
    ``[time, event]``, with ``event = 1`` for an observed event and ``0``
    for a right-censored observation.

    The partial likelihood is invariant to a constant shift of
    :math:`\mu`, so the null predictor is :math:`\mu \equiv 0` and the
    baseline hazard never needs to be estimated.
    """

    @staticmethod
    def _unpack(target: Tensor) -> tuple[Tensor, Tensor]:
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError(
                "The Cox target must have shape (n, 2) = [time, event], "
                f"got {tuple(target.shape)}."
            )
        return target[:, 0], target[:, 1]

    @staticmethod
    def _tie_blocks(sorted_times: Tensor) -> tuple[Tensor, Tensor]:
        """Return, for times sorted in descending order, the block index of
        each observation and the last position of each tie block."""
        _, counts = torch.unique_consecutive(sorted_times, return_counts=True)
        last = counts.cumsum(0) - 1
        block = torch.repeat_interleave(
            torch.arange(counts.numel(), device=sorted_times.device), counts
        )
        return block, last

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        time, event = self._unpack(target)
        order = torch.argsort(time, descending=True)
        mu = output[order]
        delta = event[order]

        # Risk-set denominators: all observations with t_j >= t_i, so every
        # member of a tie block shares the block's full cumulative sum.
        log_risk = _logcumsumexp(mu)
        block, last = self._tie_blocks(time[order])
        denominator = log_risk[last[block]]

        n_events = delta.sum().clamp_min(1.0)
        return -((mu - denominator) * delta).sum() / n_events

    def null_residuals(self, target: Tensor) -> Tensor:
        r"""Martingale residuals under the null model, in the input order.

        .. math::
            M_i = \delta_i - \hat{H}(t_i),

        with :math:`\hat{H}` the Nelson-Aalen estimator (Breslow ties).
        Up to the factor :math:`-1/d`, this is the gradient of the loss
        with respect to :math:`\mu` at :math:`\mu = 0`.
        """
        time, event = self._unpack(target)
        order = torch.argsort(time, descending=True)
        delta = event[order]

        block, last = self._tie_blocks(time[order])
        n_blocks = last.numel()
        at_risk = (last + 1).to(delta.dtype)
        events_per_block = torch.zeros(
            n_blocks, dtype=delta.dtype, device=delta.device
        ).index_add_(0, block, delta)

        # Cumulative hazard at each block's time: sum of d_b / n_b over the
        # blocks with smaller or equal times, i.e. later in descending order.
        hazard = events_per_block / at_risk
        cumulative = hazard.flip(0).cumsum(0).flip(0)

        residuals = torch.empty_like(delta)
        residuals[order] = delta - cumulative[block]
        return residuals

    def zero_thresholding(self, X: Tensor, Y: Tensor) -> Tensor:
        r"""Sup-norm of the null gradient of the smooth part,

        .. math::
            \frac{\lVert X^\top M \rVert_\infty}{d},

        with :math:`M` the null martingale residuals and :math:`d` the
        number of events. ``Y`` is a single ``(n, 2)`` target.
        """
        M = self.null_residuals(Y)
        n_events = Y[:, 1].sum().clamp_min(1.0)
        return (X.transpose(0, 1) @ M).abs().amax() / n_events

    def sample_null(
        self,
        n: int,
        M: int,
        generator: "torch.Generator | None" = None,
        *,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> Tensor:
        """Not available: the null law of (time, event) carries nuisance
        parameters (baseline hazard, censoring). The Cox estimator
        calibrates by permutation instead, conditionally on the observed
        outcomes."""
        raise NotImplementedError(
            "CoxPHLoss has no pivotal null model to sample from; calibrate "
            "by permutation of the observed (time, event) pairs instead."
        )

    def mu_null(self, y: Tensor) -> Tensor:
        return torch.zeros((), dtype=y.dtype, device=y.device)
