r"""Variable-selection model for a binary classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from deeppic.loss.loss import BinaryLoss
from deeppic.models.base import SelectionMLP

__all__ = ["BinaryClassifier"]


class BinaryClassifier(SelectionMLP):
    r"""Variable-selection model for a binary classification.

    Fits the pivotal :class:`~deeppic.loss.loss.BinaryLoss` through a logistic
    link, so that the linear predictor :math:`\eta` is the log-odds of class
    ``1``. The network, the penalty and the phased fit are the ones of
    :class:`~deeppic.models.base.SelectionMLP`.

    The targets are the labels ``0`` and ``1``, one per observation, of shape
    ``(n,)``: no one-hot encoding.

    :param input_dim: number of input variables :math:`p`.
    :param hidden_dims: width of each hidden layer, from the first to the last;
        ``None`` or empty reduces the network to a logistic model.
    :param kwargs: forwarded to :class:`~deeppic.models.base.SelectionMLP`,
        which documents the level, the penalty and the phases.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_dim, hidden_dims, BinaryLoss(), **kwargs)

    def _check_data(self, X: Tensor, y: Tensor) -> Tensor:
        r"""Check the data as :class:`~deeppic.models.base.SelectionMLP` does,
        then that every target is a label ``0`` or ``1``.

        :param X: raw design matrix.
        :param y: labels, as integers, booleans or floats.
        :return: labels, of shape ``(n,)``, in the dtype and on the device of
            ``X``.
        :raises ValueError: if the base check fails, or if ``y`` holds a value
            other than ``0`` and ``1``.
        """
        y = super()._check_data(X, y)
        if not ((y == 0.0) | (y == 1.0)).all():
            raise ValueError("y must hold the labels 0 and 1 only")
        return y

    def predict_proba(self, X: Tensor) -> Tensor:
        r"""Predict the probability of class ``1``.

        :param X: raw design matrix, of shape ``(n, p)``, standardised on the
            way in with the statistics of the fit.
        :return: :math:`P(Y = 1 \mid x) = (1 + e^{-\eta})^{-1}`, of shape
            ``(n,)``.
        """
        return super().predict(X)

    def predict(self, X: Tensor) -> Tensor:
        r"""Predict the class.

        :param X: raw design matrix, of shape ``(n, p)``, standardised on the
            way in with the statistics of the fit.
        :return: ``1`` where :meth:`predict_proba` exceeds :math:`1/2`, ``0``
            elsewhere, as integers of shape ``(n,)``.
        """
        return (self.predict_proba(X) > 0.5).long()
