r"""Losses built from a link, a base loss and a transformation of its value."""

from __future__ import annotations

import torch
from torch import Generator, Tensor
from torch.nn import Module

from .link import IdentityLink, Link, LogisticLink
from .transform import IdentityTransform, SquareRootTransform, Transform

__all__ = ["Loss", "SqrtMSELoss", "BinaryLoss"]


class Loss(Module):
    r"""Base class for losses.

    A loss composes three pieces: a link applied to the linear output of the
    model, a base loss comparing the targets to the resulting prediction, and a
    transformation of the reduced loss value. Subclasses must override
    :meth:`raw_loss` and fix the link and the transformation they need.

    :param link: link applied to the linear output.
    :param transform: transformation applied to the reduced base loss.
    """

    def __init__(self, link: Link, transform: Transform) -> None:
        super().__init__()
        self.link = link
        self.transform = transform

    def __str__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(link={self.link}, transform={self.transform})"

    def __repr__(self) -> str:
        return self.__str__()

    def forward(self, lin: Tensor, y: Tensor) -> Tensor:
        r"""Evaluate the loss.

        :param lin: linear output of the model.
        :param y: targets, same shape as ``lin``.
        :return: transformed loss, as a 0-dim tensor.
        """
        y_pred = self.link(lin)
        raw = self.raw_loss(y, y_pred)
        return self.transform(raw)

    def raw_loss(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute the base loss, before transformation.

        :param y: targets.
        :param y_pred: predictions, the linear output passed through the link.
        :return: base loss averaged over the samples, as a 0-dim tensor, or one
            value per row for targets of shape ``(batch, n)``.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def raw_loss_derivative(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute the derivative of the base loss with respect to the predictions.

        :param y: targets.
        :param y_pred: predictions, the linear output passed through the link.
        :return: :math:`\partial \ell / \partial \mu_i` for every sample, same
            shape as ``y_pred``. The reduction of :meth:`raw_loss` is included, so
            the mean over the :math:`n` samples contributes a factor :math:`1/n`.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError

    def null_mle(self, y: Tensor) -> Tensor:
        r"""Fit the linear predictor under the null hypothesis.

        Under :math:`H_0` no feature carries signal, so the model reduces to its
        intercept and the fit is the same constant for every sample. Cancelling
        the derivative of the loss in that constant gives :math:`\hat{\mu}
        = \bar{y}`, hence :math:`\hat{\eta} = g^{-1}(\bar{y})`. A loss whose
        intercept-only fit is not the mean of the targets overrides this.

        :param y: targets drawn under the null hypothesis, of shape ``(n,)`` or
            ``(batch, n)``.
        :return: fitted constant linear predictor, of shape ``(1,)`` or
            ``(batch, 1)``.
        """
        return self.link.inverse(y.mean(dim=-1, keepdim=True))

    def null_gradient(self, X: Tensor, y: Tensor) -> Tensor:
        r"""Compute the gradient with respect to the coefficients under :math:`H_0`.

        The linear predictor is the null fit of :meth:`null_mle`, and the chain
        rule through the transformation, the link and the base loss gives the
        per-sample residual

        .. math::
            r_i = h'(\ell) \, g'(\hat{\eta}) \,
            \frac{\partial \ell}{\partial \mu_i},

        from which :math:`\nabla_\beta = X^\top r`. Its distribution over
        repeated draws of ``y`` is what sets the pivotal regularisation level.

        The gradient with respect to the intercept is not returned: it is the
        first-order condition satisfied by :meth:`null_mle`, hence zero.

        Draws are independent, so a batch of targets of shape ``(batch, n)``
        yields the ``batch`` gradients in one pass.

        :param X: design matrix, of shape ``(n, p)``.
        :param y: targets drawn under the null hypothesis, of shape ``(n,)`` or
            ``(batch, n)``, typically from :meth:`sample_null`.
        :return: gradient with respect to the coefficients, of shape ``(p,)`` or
            ``(batch, p)``.
        """
        lin = self.null_mle(y).expand_as(y)
        y_pred = self.link(lin)
        raw = self.raw_loss(y, y_pred)
        residual = (
            self.transform.derivative(raw).unsqueeze(-1)
            * self.link.derivative(lin)
            * self.raw_loss_derivative(y, y_pred)
        )
        return residual @ X

    def sample_null(self, size: int, generator: Generator | None = None) -> Tensor:
        r"""Draw targets under the null hypothesis.

        Under :math:`H_0` the linear output carries no signal, so the targets
        follow the response distribution of the loss at a null linear predictor,
        :math:`\mu = g(0)`. Used to calibrate the regularisation level by
        simulation.

        :param size: number of samples to draw.
        :param generator: pseudo-random generator, for reproducible draws.
        :return: targets, of shape ``(size,)``.
        :raises NotImplementedError: if the subclass does not override this method.
        """
        raise NotImplementedError


class SqrtMSELoss(Loss):
    r"""Square root of the mean squared error.

    .. math::
        \sqrt{\frac{1}{n} \sum_i (y_i - \mu_i)^2}, \qquad \mu_i = \eta_i.

    Identity link and square-root transformation: the square-root Lasso
    objective, pivotal with respect to the noise scale :math:`\sigma`.
    """

    def __init__(self) -> None:
        super().__init__(link=IdentityLink(), transform=SquareRootTransform())

    def raw_loss(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute the mean squared error.

        :param y: targets.
        :param y_pred: predictions, equal to the linear output.
        :return: mean squared error over the samples, as a 0-dim tensor, or one
            value per row for targets of shape ``(batch, n)``.
        """
        return ((y - y_pred) ** 2).mean(dim=-1)

    def raw_loss_derivative(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute :math:`\frac{\partial \ell}{\partial \mu_i}
        = \frac{2}{n} (\mu_i - y_i)`.

        :param y: targets.
        :param y_pred: predictions, equal to the linear output.
        :return: derivative for every sample, same shape as ``y_pred``.
        """
        return 2.0 * (y_pred - y) / y_pred.shape[-1]

    def sample_null(self, size: int, generator: Generator | None = None) -> Tensor:
        r"""Draw :math:`y_i \sim \mathcal{N}(0, 1)`, the Gaussian noise left
        under :math:`H_0`.

        :param size: number of samples to draw.
        :param generator: pseudo-random generator, for reproducible draws.
        :return: targets, of shape ``(size,)``.
        """
        return torch.randn(size, generator=generator)


class BinaryLoss(Loss):
    r"""Pivotal loss for binary targets.

    .. math::
        \frac{1}{n} \sum_i 2 y_i \sqrt{\frac{1 - \mu_i}{\mu_i}}
        + 2 (1 - y_i) \sqrt{\frac{\mu_i}{1 - \mu_i}},
        \qquad \mu_i = \frac{1}{1 + e^{-\eta_i}}.

    Logistic link, and no transformation of the loss value.
    For :math:`y_i \in \{0, 1\}` the summand is :math:`2 e^{-(2 y_i - 1) \eta_i / 2}`,
    the exponential loss of boosting on the half-margin. It decays
    to zero on a well-classified sample and grows exponentially on
    a misclassified one.

    :param eps: bound keeping :math:`\mu_i` in
        :math:`[\varepsilon, 1 - \varepsilon]`. A saturated sigmoid returns
        exactly ``0`` or ``1``, for which the summand of the opposite class is
        :math:`0 \times \infty`, i.e. ``nan``.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__(link=LogisticLink(), transform=IdentityTransform())
        self.eps = eps

    def raw_loss(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute the pivotal binary loss.

        :param y: targets in :math:`[0, 1]`, usually ``0`` or ``1``.
        :param y_pred: predicted probabilities, clamped to
            :math:`[\varepsilon, 1 - \varepsilon]`.
        :return: loss averaged over the samples, as a 0-dim tensor, or one value
            per row for targets of shape ``(batch, n)``.
        """
        mu = y_pred.clamp(self.eps, 1.0 - self.eps)
        odds = mu / (1.0 - mu)
        pointwise = 2.0 * y * torch.rsqrt(odds) + 2.0 * (1.0 - y) * torch.sqrt(odds)
        return pointwise.mean(dim=-1)

    def raw_loss_derivative(self, y: Tensor, y_pred: Tensor) -> Tensor:
        r"""Compute :math:`\frac{\partial \ell}{\partial \mu_i} = \frac{1}{n}
        \frac{(1 - y_i) \sqrt{\mu_i / (1 - \mu_i)}
        - y_i \sqrt{(1 - \mu_i) / \mu_i}}{\mu_i (1 - \mu_i)}`.

        :param y: targets in :math:`[0, 1]`, usually ``0`` or ``1``.
        :param y_pred: predicted probabilities, clamped as in :meth:`raw_loss`.
        :return: derivative for every sample, same shape as ``y_pred``.
        """
        mu = y_pred.clamp(self.eps, 1.0 - self.eps)
        odds = mu / (1.0 - mu)
        pointwise = (1.0 - y) * torch.sqrt(odds) - y * torch.rsqrt(odds)
        return pointwise / (mu * (1.0 - mu) * y_pred.shape[-1])

    def null_mle(self, y: Tensor) -> Tensor:
        r"""Fit :math:`\hat{\eta} = \log \frac{\bar{y}}{1 - \bar{y}}`.

        The fit of :meth:`Loss.null_mle`, with the mean clamped to
        :math:`[\varepsilon, 1 - \varepsilon]`: a sample holding no ``0`` or no
        ``1`` has no finite fit.

        :param y: targets drawn under the null hypothesis.
        :return: logit of the clamped mean of ``y``, as a 0-dim tensor.
        """
        mean = y.mean(dim=-1, keepdim=True)
        return self.link.inverse(mean.clamp(self.eps, 1.0 - self.eps))

    def sample_null(self, size: int, generator: Generator | None = None) -> Tensor:
        r"""Draw :math:`y_i \sim \mathcal{B}(0.5)`, the probability the logistic
        link returns at a null linear predictor.

        :param size: number of samples to draw.
        :param generator: pseudo-random generator, for reproducible draws.
        :return: targets, of shape ``(size,)``.
        """
        return torch.bernoulli(torch.full((size,), 0.5), generator=generator)
