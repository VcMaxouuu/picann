"""Sparse binary classification."""

from __future__ import annotations

import torch
from torch import Tensor

from ..optim import ExpBinaryLoss, Loss
from .estimator import SparseEstimator

__all__ = ["SparseClassifier"]


class SparseClassifier(SparseEstimator):
    r"""Binary classification with calibrated variable selection.

    Minimises the exponential binary loss plus a penalty on the first
    weight matrix; see :class:`~deeppic.models.SparseEstimator` for the
    shared parameters and fitted attributes. Labels are ``{0, 1}``. At the
    optimum the network output is the log-odds
    :math:`\mu(x) = \log \frac{p(x)}{1 - p(x)}`, which
    :meth:`predict_proba` maps through the sigmoid.

    Examples
    --------
    >>> model = SparseClassifier(hidden_dims=(64, 32)).fit(X, y)
    >>> model.selected_
    >>> proba = model.predict_proba(X_new)
    """

    def _build_loss(self) -> Loss:
        return ExpBinaryLoss()

    def _check_target(self, y) -> Tensor:
        y = super()._check_target(y)
        if not torch.all((y == 0.0) | (y == 1.0)):
            raise ValueError("y must contain only the labels 0 and 1.")
        return y

    def decision_function(self, X) -> Tensor:
        """Log-odds scores, shape ``(n_samples,)``."""
        return self._decision(X)

    def predict_proba(self, X) -> Tensor:
        """Probabilities of the label ``1``, shape ``(n_samples,)``."""
        return torch.sigmoid(self._decision(X))

    def predict(self, X) -> Tensor:
        """Predicted labels in ``{0, 1}``, shape ``(n_samples,)``."""
        return (self._decision(X) > 0).long()
