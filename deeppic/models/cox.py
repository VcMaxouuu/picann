"""Sparse Cox proportional-hazards model."""

from __future__ import annotations

import torch
from torch import Tensor

from ..optim import CoxPHLoss, Loss
from .estimator import SparseEstimator

__all__ = ["SparseCoxPH"]


class SparseCoxPH(SparseEstimator):
    r"""Cox proportional-hazards model with calibrated variable selection.

    Minimises the negative Cox partial log-likelihood (Breslow ties) plus
    a penalty on the first weight matrix; see
    :class:`~deeppic.models.SparseEstimator` for the shared parameters and
    fitted attributes. The network output is the log relative hazard:

    .. math::
        h(t \mid x) = h_0(t) \, e^{\mu_\theta(x)} .

    The target is ``(n, 2)``, one row ``[time, event]`` per observation
    with ``event = 1`` for an observed event and ``0`` for right
    censoring; ``fit`` also accepts the pair ``(times, events)``.

    Unlike the regression and classification cases, the null law of the
    outcome carries nuisance parameters (baseline hazard, censoring), so
    the threshold is calibrated by permutation instead of Monte-Carlo
    sampling: under :math:`H_0` the outcomes are independent of ``X``, so
    permuting the observed ``(time, event)`` pairs against the rows of
    ``X`` gives an exact conditional null for the zero-thresholding
    statistic.

    The partial likelihood is invariant to a constant shift of the
    output, so the output bias is not identifiable and ``bias`` defaults
    to ``False`` here; hidden layers keep theirs.

    Examples
    --------
    >>> model = SparseCoxPH(hidden_dims=(64, 32)).fit(X, (times, events))
    >>> model.selected_
    >>> risk = model.predict(X_new)
    """

    def __init__(
        self,
        hidden_dims=None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(hidden_dims=hidden_dims, bias=bias, **kwargs)

    def _build_loss(self) -> Loss:
        return CoxPHLoss()

    def _check_target(self, y) -> Tensor:
        if isinstance(y, (tuple, list)) and len(y) == 2:
            y = torch.stack(
                [self._as_tensor(y[0]), self._as_tensor(y[1])], dim=1
            )
        y = self._as_tensor(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "y must have shape (n, 2) = [time, event], or be a "
                f"(times, events) pair; got shape {tuple(y.shape)}."
            )

        time, event = y[:, 0], y[:, 1]
        if not torch.all(torch.isfinite(time)):
            raise ValueError("All follow-up times must be finite.")
        if not torch.all((event == 0.0) | (event == 1.0)):
            raise ValueError("Event indicators must contain only 0 and 1.")
        if event.sum() == 0:
            raise ValueError(
                "The Cox partial likelihood is undefined without any "
                "observed event."
            )
        return y

    def _calibrate(
        self, X: Tensor, y: Tensor, generator: "torch.Generator | None"
    ) -> float:
        """Permutation quantile of the zero-thresholding statistic."""
        residuals = self.loss.null_residuals(y)
        n = residuals.numel()
        n_events = y[:, 1].sum().clamp_min(1.0)

        # A generator is bound to a device: permute where it lives, then
        # move the whole block once. A seeded run is thus reproducible
        # whatever device the statistic is evaluated on.
        draw_device = residuals.device if generator is None else generator.device
        drawn = residuals.to(draw_device)
        permuted = torch.stack(
            [
                drawn[torch.randperm(n, generator=generator, device=draw_device)]
                for _ in range(self.n_mc)
            ],
            dim=1,
        ).to(X.device)
        stats = (X.transpose(0, 1) @ permuted).abs().amax(dim=0) / n_events
        return float(torch.quantile(stats.cpu().double(), 1.0 - self.alpha))

    def predict(self, X) -> Tensor:
        """Log relative hazards, shape ``(n_samples,)``.

        Defined up to an additive constant absorbed by the baseline
        hazard; differences between individuals are meaningful.
        """
        return self._decision(X)

    def predict_partial_hazard(self, X) -> Tensor:
        """Relative hazards :math:`e^{\\mu(x)}`, shape ``(n_samples,)``."""
        return torch.exp(self._decision(X))
