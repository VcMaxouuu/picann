r"""Non-regression on the one-neuron example.

One input :math:`x = 1`, one hidden unit, biases frozen at zero:
:math:`\eta = v \sigma(w x)` and, with :math:`u = v w > 0`,
:math:`J_1 = \frac12 (u - 2)^2 + |u|`, minimised at :math:`u^* = 1`. Without the
multiplier term, a retraction alone leads gradient descent to the root of
:math:`(u - 2)(1 + u^2) = -1`, :math:`u \approx 1.75`.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from deeppic.loss.link import IdentityLink
from deeppic.loss.loss import Loss
from deeppic.loss.transform import IdentityTransform
from deeppic.models.base import SelectionMLP


class HalfSquaredLoss(Loss):
    r""":math:`\frac12 \sum_i (y_i - \eta_i)^2`, a loss of the test only."""

    def __init__(self) -> None:
        super().__init__(link=IdentityLink(), transform=IdentityTransform())

    def raw_loss(self, y: Tensor, y_pred: Tensor) -> Tensor:
        return 0.5 * ((y - y_pred) ** 2).sum(dim=-1)


def _toy_model() -> tuple[SelectionMLP, Tensor, Tensor]:
    model = SelectionMLP(1, (1,), HalfSquaredLoss()).double()
    first, second = model._linears
    with torch.no_grad():
        first.weight.fill_(0.5)
        second.weight.fill_(0.5)
        first.bias.zero_()
        second.bias.zero_()
    first.bias.requires_grad_(False)
    second.bias.requires_grad_(False)
    X = torch.ones(1, 1, dtype=torch.float64)
    y = torch.full((1,), 2.0, dtype=torch.float64)
    return model, X, y


def _product(model: SelectionMLP) -> float:
    first, second = model._linears
    return float(first.weight.detach() * second.weight.detach())


def test_toy_example_reaches_the_minimiser():
    model, X, y = _toy_model()
    model.fit_phase(X, y, lam=1.0, tol=1e-10, n_epochs=20000)
    assert abs(_product(model) - 1.0) < 1e-3
    assert abs(float(model.sensitivity(X)) - 1.0) < 1e-12


def test_toy_example_without_the_multiplier_term_misses_it(monkeypatch):
    r"""The former scheme, retraction without :math:`\mu \nabla s`, does not
    reach :math:`u^* = 1`."""
    model, X, y = _toy_model()
    monkeypatch.setattr(
        model,
        "_linearised_output",
        lambda masks, U: torch.zeros((), dtype=U.dtype, requires_grad=True),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit_phase(X, y, lam=1.0, tol=1e-10, n_epochs=3000)
    assert _product(model) > 1.2
