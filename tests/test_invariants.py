r"""Invariants the calibration under :math:`H_0` relies on (full batch, same design
matrix for the calibration and the fit, gradient handed to the proximal step)."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from _helpers import DEPTHS, LOSSES, has_hidden, loss_gradients, make_problem, rescale

LAM = 0.4


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", (None, *DEPTHS))
def test_prox_receives_the_loss_gradient_at_the_retracted_point(kind, hidden):
    r"""The gradient :math:`W^{(1)}` carries into its proximal step is
    :math:`\nabla_{W^{(1)}} \ell` at the current, retracted, iterate: no penalty
    and no subgradient mixed in. The loss is that of the whole sample."""
    model, X, y = make_problem(kind, hidden)
    if has_hidden(model):
        generator = torch.Generator().manual_seed(5)
        rescale(model, torch.empty(hidden[0], dtype=X.dtype).uniform_(0.2, 5.0, generator=generator))
    optimizer = model._optimizer(LAM)
    optimizer.zero_grad()
    loss, s = model._backward(X, y, LAM)
    with torch.no_grad():
        assert_close(loss, model.loss(model(X), y), rtol=0.0, atol=0.0)
    if has_hidden(model):
        model._retract(s, optimizer)
    expected = loss_gradients(model, X, y)["layers.0.weight"]
    assert_close(model.selector.weight.grad, expected, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fit_calibrates_and_fits_on_the_same_design(monkeypatch, dtype):
    r"""The calibration and every phase see the same standardised design
    matrix, in the dtype of the data."""
    model, X, y = make_problem("gaussian", (4,), n=30, p=5)
    model = model.to(dtype)
    X, y = (3.0 * X + 1.0).to(dtype), y.to(dtype)
    seen = []

    calibrate = model.calibrate

    def spy_calibrate(Z, *args, **kwargs):
        seen.append(("calibrate", Z))
        return calibrate(Z, *args, **kwargs)

    def spy_phase(Z, target, lam, tol, n_epochs):
        seen.append(("phase", Z))
        return torch.empty(0)

    monkeypatch.setattr(model, "calibrate", spy_calibrate)
    monkeypatch.setattr(model, "fit_phase", spy_phase)
    model.fit(X, y, generator=torch.Generator().manual_seed(0))

    assert [name for name, _ in seen] == ["calibrate"] + ["phase"] * model.n_phases
    design = seen[0][1]
    assert design.dtype == dtype
    assert_close(design.mean(dim=0), torch.zeros(5, dtype=dtype), rtol=0.0, atol=1e-5)
    assert_close(design.pow(2).mean(dim=0), torch.ones(5, dtype=dtype), rtol=1e-5, atol=0.0)
    for _, Z in seen[1:]:
        assert Z is design
    assert model.lambda_.dtype == dtype
