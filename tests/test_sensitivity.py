r"""The sensitivity :math:`a_{ik} = \partial \eta_i / \partial h^{(1)}_{ik}` and
:math:`s_k = \max_i |a_{ik}|`, as the fit computes them."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from _helpers import DEPTHS, LOSSES, make_problem


def _reference(model, X):
    with torch.no_grad():
        eta, preactivations = model._forward_with_preactivations(X)
        return eta, model._jacobian(model._masks(preactivations))


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_jacobian_matches_autograd(kind, hidden):
    model, X, _ = make_problem(kind, hidden)
    _, a = _reference(model, X)
    _, preactivations = model._forward_with_preactivations(X)
    h1 = preactivations[0].detach().requires_grad_()
    downstream = model.layers[1:]
    expected = torch.autograd.grad(downstream(h1).sum(), h1)[0]
    assert_close(a, expected, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_loss_backward_gives_the_sensitivity(kind, hidden):
    r"""Every observation goes through the network on its own, so
    :math:`G = \partial \ell / \partial h^{(1)} = r \odot a`: :math:`a = G / r`."""
    model, X, y = make_problem(kind, hidden)
    _, a = _reference(model, X)
    eta, preactivations = model._forward_with_preactivations(X)
    eta.retain_grad()
    preactivations[0].retain_grad()
    model.loss(eta, y).backward()
    ratio = preactivations[0].grad / eta.grad.unsqueeze(-1)
    assert_close(ratio, a, rtol=1e-12, atol=0.0)

    model.zero_grad()
    _, s = model._backward(X, y, 0.0)
    assert_close(s, a.abs().amax(dim=0), rtol=1e-12, atol=0.0)
    assert_close(model.sensitivity(X), a.abs().amax(dim=0), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("hidden", DEPTHS)
def test_rows_with_zero_residual_regression(hidden):
    r"""Where :math:`r_i = 0`, :math:`G_i = 0` says nothing about :math:`a_i`:
    the rows attaining every maximum are made degenerate, and :math:`s` must
    not move."""
    model, X, y = make_problem("gaussian", hidden)
    eta, a = _reference(model, X)
    rows = a.abs().argmax(dim=0).unique()
    y = y.clone()
    y[rows] = eta[rows]

    eta, _ = model._forward_with_preactivations(X)
    eta.retain_grad()
    model.loss(eta, y).backward()
    assert (eta.grad[rows] == 0.0).all()

    for lam in (0.0, 0.5):
        model.zero_grad()
        _, s = model._backward(X, y, lam)
        assert_close(s, a.abs().amax(dim=0), rtol=1e-12, atol=0.0)
        for parameter in model.parameters():
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("hidden", DEPTHS)
def test_rows_with_zero_residual_binary(hidden):
    r"""A saturated probability is clamped, and its residual is exactly zero: with
    every probability saturated, no row carries :math:`a`."""
    model, X, y = make_problem("binary", hidden)
    with torch.no_grad():
        model._linears[-1].bias.add_(60.0)
    _, a = _reference(model, X)

    eta, _ = model._forward_with_preactivations(X)
    eta.retain_grad()
    model.loss(eta, y).backward()
    assert (eta.grad == 0.0).all()

    for lam in (0.0, 0.5):
        model.zero_grad()
        _, s = model._backward(X, y, lam)
        assert_close(s, a.abs().amax(dim=0), rtol=1e-12, atol=0.0)
        for parameter in model.parameters():
            assert torch.isfinite(parameter.grad).all()


def test_linear_model_has_unit_sensitivity():
    model, X, y = make_problem("gaussian", None)
    assert_close(model.sensitivity(X), torch.ones(1, dtype=X.dtype))
    _, s = model._backward(X, y, 0.5)
    assert_close(s, torch.ones(1, dtype=X.dtype))
