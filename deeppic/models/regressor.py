"""Sparse regression."""

from __future__ import annotations

from torch import Tensor

from ..optim import Loss, SqrtMSELoss
from .estimator import SparseEstimator

__all__ = ["SparseRegressor"]


class SparseRegressor(SparseEstimator):
    r"""Regression with calibrated variable selection.

    Minimises the square-root MSE plus a penalty on the first weight
    matrix; see :class:`~deeppic.models.SparseEstimator` for the shared
    parameters and fitted attributes. The square-root loss is what makes
    the calibrated threshold pivotal: :math:`\lambda^{\rm PDB}` does not
    depend on the noise level.

    Examples
    --------
    >>> model = SparseRegressor(hidden_dims=(64, 32)).fit(X, y)
    >>> model.selected_
    >>> y_hat = model.predict(X_new)
    """

    def _build_loss(self) -> Loss:
        return SqrtMSELoss()

    def predict(self, X) -> Tensor:
        """Predicted responses, shape ``(n_samples,)``."""
        return self._decision(X)
