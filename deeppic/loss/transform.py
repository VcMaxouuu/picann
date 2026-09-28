r"""Transformations applied to the value of a loss.

A transformation acts on the *scalar* value of a base loss, once reduced over
the batch, to change the statistical behaviour of the objective without changing
what the model predicts.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

__all__ = [
    "Transform",
    "IdentityTransform",
    "SquareRootTransform",
    "ExponentialTransform",
]


class Transform(Module):
    r"""Base class for transformations of a loss value.

    A transformation is a scalar function :math:`h`, applied to the reduced loss
    and differentiable on the range the loss takes. Values are 0-dim tensors,
    not Python floats, so that the transformation stays part of the autograd
    graph. They are applied elementwise, so a stack of independent losses is
    transformed in one call. Subclasses must override :meth:`forward` and
    :meth:`derivative`.
    """

    def forward(self, loss: Tensor) -> Tensor:
        r"""Apply the transformation :math:`h`.

        :param loss: scalar loss value, as a 0-dim tensor.
        :return: :math:`h(\ell)`, as a 0-dim tensor.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def derivative(self, loss: Tensor) -> Tensor:
        r"""Apply the derivative :math:`h'` of the transformation.

        :param loss: scalar loss value, as a 0-dim tensor.
        :return: :math:`h'(\ell)`, as a 0-dim tensor.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return str(self)


class IdentityTransform(Transform):
    r"""Identity transformation :math:`h(\ell) = \ell`, leaving the loss as is."""

    def forward(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h(\ell) = \ell`.

        :param loss: scalar loss value.
        :return: ``loss`` itself, not a copy.
        """
        return loss

    def derivative(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h'(\ell) = 1`.

        :param loss: scalar loss value.
        :return: one, with the dtype and device of ``loss``.
        """
        return torch.ones_like(loss)


class SquareRootTransform(Transform):
    r"""Square-root transformation :math:`h(\ell) = \sqrt{\ell}`.
    """

    def forward(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h(\ell) = \sqrt{\ell}`.

        :param loss: scalar loss value, expected non-negative; a negative value
            gives ``nan``.
        :return: square root of ``loss``.
        """
        return torch.sqrt(loss)

    def derivative(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h'(\ell) = \frac{1}{2 \sqrt{\ell}}`.

        :param loss: scalar loss value, expected strictly positive; zero gives
            ``inf`` and a negative value ``nan``.
        :return: derivative at ``loss``.
        """
        return 0.5 * torch.rsqrt(loss)


class ExponentialTransform(Transform):
    r"""Exponential transformation :math:`h(\ell) = e^{\ell}`.
    """

    def forward(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h(\ell) = e^{\ell}`.

        :param loss: scalar loss value.
        :return: exponential of ``loss``.
        """
        return torch.exp(loss)

    def derivative(self, loss: Tensor) -> Tensor:
        r"""Apply :math:`h'(\ell) = e^{\ell}`.

        :param loss: scalar loss value.
        :return: derivative at ``loss``, equal to :math:`h(\ell)`.
        """
        return torch.exp(loss)
