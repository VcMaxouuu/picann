r"""Link functions applied to the linear output of the model.

A link maps the linear predictor to the quantity the loss compares to the
targets. It acts elementwise on a tensor, unlike a transformation of the
loss value.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

__all__ = ["Link", "IdentityLink", "LogisticLink", "LogLink"]


class Link(Module):
    r"""Base class for link functions applied to the linear output.

    - :meth:`forward` maps the linear predictor :math:`\eta` to the mean
      :math:`\mu = g(\eta)`;
    - :meth:`inverse` maps :math:`\mu` back to :math:`\eta`;
    - :meth:`derivative` gives :math:`g'(\eta)`.

    All three act elementwise and preserve the shape of their input. Subclasses
    must override them.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the link :math:`g`.

        :param x: linear predictor.
        :return: :math:`g(x)`, same shape as ``x``.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def derivative(self, x: Tensor) -> Tensor:
        r"""Apply the derivative :math:`g'` of the link.

        :param x: linear predictor.
        :return: :math:`g'(x)`, same shape as ``x``.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def inverse(self, x: Tensor) -> Tensor:
        r"""Apply the inverse link :math:`g^{-1}`.

        :param x: mean.
        :return: :math:`g^{-1}(x)`, same shape as ``x``.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return str(self)


class IdentityLink(Link):
    r"""Identity link :math:`g(\eta) = \eta`."""

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g(\eta) = \eta`.

        :param x: linear predictor.
        :return: ``x`` itself, not a copy.
        """
        return x

    def derivative(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g'(\eta) = 1`.

        :param x: linear predictor.
        :return: tensor of ones with the shape, dtype and device of ``x``.
        """
        return torch.ones_like(x)

    def inverse(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g^{-1}(\mu) = \mu`.

        :param x: mean.
        :return: ``x`` itself, not a copy.
        """
        return x


class LogisticLink(Link):
    r"""Logistic link :math:`g(\eta) = (1 + e^{-\eta})^{-1}`.

    Maps the linear predictor to a probability in :math:`(0, 1)`; its inverse is
    the logit :math:`\log \frac{\mu}{1 - \mu}`.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g(\eta) = (1 + e^{-\eta})^{-1}`.

        :param x: linear predictor.
        :return: probability in :math:`(0, 1)`, same shape as ``x``.
        """
        return torch.sigmoid(x)

    def derivative(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g'(\eta) = g(\eta) \, (1 - g(\eta))`.

        :param x: linear predictor.
        :return: derivative in :math:`(0, 1/4]`, same shape as ``x``.
        """
        p = torch.sigmoid(x)
        return p * (1.0 - p)

    def inverse(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g^{-1}(\mu) = \log \frac{\mu}{1 - \mu}`.

        :param x: probability, expected in :math:`(0, 1)`; ``0`` and ``1`` give
            :math:`\mp \infty`, values outside :math:`[0, 1]` give ``nan``.
        :return: linear predictor, same shape as ``x``.
        """
        return torch.logit(x)


class LogLink(Link):
    r"""Log link, mapping the linear predictor to a positive mean.

    :meth:`forward` applies the corresponding mean function
    :math:`g(\eta) = e^{\eta}`. Canonical for Poisson regression.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g(\eta) = e^{\eta}`.

        :param x: linear predictor.
        :return: positive mean, same shape as ``x``; overflows to ``inf`` for
            large ``x``.
        """
        return torch.exp(x)

    def derivative(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g'(\eta) = e^{\eta}`.

        :param x: linear predictor.
        :return: derivative, equal to :math:`g(x)`, same shape as ``x``.
        """
        return torch.exp(x)

    def inverse(self, x: Tensor) -> Tensor:
        r"""Apply :math:`g^{-1}(\mu) = \log \mu`.

        :param x: mean, expected strictly positive; ``0`` gives
            :math:`-\infty` and negative values give ``nan``.
        :return: linear predictor, same shape as ``x``.
        """
        return torch.log(x)
