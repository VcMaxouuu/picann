r"""First-order conditions of :math:`(P)` at the end of a phase.

On :math:`W^{(1)}`, on the section :math:`s = \mathbf 1`:
:math:`g_{kj} = -\lambda \operatorname{sign} W^{(1)}_{kj}` on the support and
:math:`|g_{kj}| \leq \lambda` off it; on the other parameters,
:math:`\nabla_\varphi J_\lambda = 0`. Adam stops near the fixed point, not on
it, hence the tolerances.

Where the minimiser lies on an activation boundary, :math:`\ell` is not
differentiable there and the conditions only hold in the sense of Clarke:
the problem below is one whose minimiser does not.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch.nn import LeakyReLU

from _helpers import has_hidden, make_problem

CASES = [
    ("gaussian", None, 0.01),
    ("gaussian", (3,), 0.01),
    ("gaussian", (3, 2), 0.01),
    ("gaussian", (3, 2), 1.0),
    ("binary", (3,), 0.01),
]


def _kkt_residuals(model, X, y, lam):
    optimizer = model._optimizer(lam)
    optimizer.zero_grad()
    _, s = model._gradients(X, y, lam)
    if has_hidden(model):
        s = model._normalize(s, optimizer)
    w1 = model.selector.weight
    g = w1.grad
    support = w1.detach() != 0.0
    assert support.any() and (~support).any()
    on = (g + lam * s.unsqueeze(-1) * w1.detach().sign()).abs()[support].max() / lam
    off = (g.abs() / (lam * s.unsqueeze(-1)))[~support].max()
    phi = max(float(q.grad.abs().max()) for q in model.parameters() if q is not w1)
    return float(on), float(off), phi


@pytest.mark.parametrize(("kind", "hidden", "slope"), CASES)
def test_kkt_conditions_at_convergence(kind, hidden, slope):
    model, X, y = make_problem(kind, hidden, n=60, p=5, seed=1)
    for module in model.layers:
        if isinstance(module, LeakyReLU):
            module.negative_slope = slope
    lam = 0.4 * float(model.loss.null_gradient(X, y).abs().max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit_phase(X, y, lam=lam, tol=1e-10, n_epochs=6000)

    on, off, phi = _kkt_residuals(model, X, y, lam)
    assert on <= 2e-2
    assert off <= 1.0 + 1e-2
    assert phi <= 1e-3
