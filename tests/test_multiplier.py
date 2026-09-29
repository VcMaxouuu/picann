r"""The multiplier term :math:`\sum_k \mu_k \nabla_\varphi s_k`, :math:`\mu_k =
\lambda \|W^{(1)}_{k \cdot}\|_1`, that the deeper layers receive on top of
:math:`\nabla_\varphi \ell`."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from _helpers import (
    DEPTHS,
    LOSSES,
    backward_gradients,
    is_generic,
    make_problem,
    multiplier_gradients,
)

LAM = 0.7


def _multiplier_from_gradients(model, X, y, lam):
    with_term = backward_gradients(model, X, y, lam)
    without = backward_gradients(model, X, y, 0.0)
    return {name: with_term[name] - without[name] for name in with_term}


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
@pytest.mark.parametrize("degenerate", [False, True])
def test_multiplier_gradient_matches_autograd(kind, hidden, degenerate):
    model, X, y = make_problem(kind, hidden)
    if degenerate and kind == "gaussian":
        with torch.no_grad():
            eta, masks = model._forward(X)
            a = model._jacobian(masks)
            rows = a.abs().argmax(dim=0)
            y = y.clone()
            y[rows] = eta[rows]
    elif degenerate:
        with torch.no_grad():
            model._linears[-1].bias.add_(60.0)

    got = _multiplier_from_gradients(model, X, y, LAM)
    expected = multiplier_gradients(model, X, LAM)
    first = model.selector
    for name, parameter in model.named_parameters():
        assert_close(got[name], expected[name], rtol=1e-10, atol=1e-13)
        if parameter is first.weight or name.endswith("bias"):
            assert (expected[name] == 0.0).all()


@pytest.mark.parametrize("hidden", DEPTHS)
def test_multiplier_output_is_the_multiplier_term(hidden):
    r"""Fed :math:`u^{(k)} = \mu_k \operatorname{sign}(a_{i^*_k k}) e_k` at the
    observation :math:`i^*_k`, the linearised network outputs :math:`\sum_k
    \mu_k s_k`, and its gradient is the multiplier term."""
    model, X, _ = make_problem("gaussian", hidden)
    with torch.no_grad():
        _, masks = model._forward(X)
        a = model._jacobian(masks)
        s = a.abs().amax(dim=0)
        mu = LAM * model.selector.weight.abs().sum(dim=1)
    model.zero_grad()
    output = model._multiplier(masks, a, LAM)
    assert_close(output, torch.dot(mu, s), rtol=1e-13, atol=0.0)
    output.backward()
    expected = multiplier_gradients(model, X, LAM)
    for name, parameter in model.named_parameters():
        grad = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        assert_close(grad, expected[name], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("hidden", DEPTHS)
def test_multiplier_gradient_matches_finite_differences(hidden):
    r"""Central differences of :math:`\sum_k \mu_k s_k(\theta)`, with the
    sensitivity recomputed, masks included, away from every activation
    boundary and every tie of the maxima."""
    for seed in range(200):
        model, X, y = make_problem("gaussian", hidden, seed=seed)
        if is_generic(model, X):
            break
    else:
        pytest.fail("no generic configuration found")

    mu = LAM * model.selector.weight.detach().abs().sum(dim=1)
    got = _multiplier_from_gradients(model, X, y, LAM)
    generator = torch.Generator().manual_seed(0)
    step = 1e-6
    for index, layer in enumerate(model._linears[1:], start=1):
        weight = layer.weight
        name = f"layers.{2 * index}.weight"
        for flat in torch.randperm(weight.numel(), generator=generator)[:6].tolist():
            entry = divmod(flat, weight.shape[1])
            with torch.no_grad():
                weight[entry] += step
                plus = torch.dot(mu, model.sensitivity(X))
                weight[entry] -= 2.0 * step
                minus = torch.dot(mu, model.sensitivity(X))
                weight[entry] += step
            difference = float((plus - minus) / (2.0 * step))
            assert abs(difference - float(got[name][entry])) <= 1e-6 * max(1.0, abs(difference))
