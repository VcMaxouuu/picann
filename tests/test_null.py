r"""Exact behaviour at the null parameter :math:`\theta^0 = (W^{(1)} = 0,
\hat\tau)`: the calibration keeps its meaning along the fit."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from _helpers import DEPTHS, LOSSES, has_hidden, loss_gradients, make_problem, null_point

ARCHITECTURES = (None, *DEPTHS)


def _null_problem(kind, hidden):
    model, X, y = make_problem(kind, hidden)
    null_point(model, X, y)
    with torch.no_grad():
        eta = model(X)
    assert_close(eta, model.loss.null_mle(y).expand_as(eta), rtol=0.0, atol=1e-13)
    return model, X, y


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", ARCHITECTURES)
def test_null_gradient_matches_autograd(kind, hidden):
    r""":math:`\nabla_{W^{(1)}_{k\cdot}} \ell(\theta^0) = a_{0k} X^\top
    \hat{\mathbf r}_0`: divided by :math:`s_k(\theta^0) = |a_{0k}|`, it is
    :meth:`Loss.null_gradient` up to the sign of :math:`a_{0k}`, and the largest
    ratio is :math:`\lambda_0(X, \mathbf y)`."""
    model, X, y = _null_problem(kind, hidden)
    gradient = loss_gradients(model, X, y)["layers.0.weight"]
    null_gradient = model.loss.null_gradient(X, y)

    if has_hidden(model):
        with torch.no_grad():
            _, masks = model._forward(X)
            a = model._jacobian(masks)
        assert_close(a, a[:1].expand_as(a), rtol=0.0, atol=0.0)
        a0 = a[0]
    else:
        a0 = torch.ones(1, dtype=X.dtype)
    s = model.sensitivity(X)
    assert_close(s, a0.abs())

    assert_close(gradient, a0.unsqueeze(-1) * null_gradient, rtol=1e-10, atol=1e-15)
    assert_close(gradient.abs() / s.unsqueeze(-1), null_gradient.abs().expand_as(gradient),
                 rtol=1e-10, atol=1e-15)
    lambda0 = null_gradient.abs().max()
    assert_close((gradient.abs() / s.unsqueeze(-1)).max(), lambda0, rtol=1e-10, atol=0.0)


def _one_step(model, X, y, lam):
    optimizer = model._optimizer(lam)
    optimizer.zero_grad()
    loss, s = model._gradients(X, y, lam)
    grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    if has_hidden(model):
        s = model._normalize(s, optimizer)
    optimizer.param_groups[0]["penalty_weights"] = s.unsqueeze(-1)
    optimizer.step()
    return grads


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", ARCHITECTURES)
def test_null_point_is_fixed_above_lambda0(kind, hidden):
    r"""At :math:`\lambda = 1.01 \lambda_0`, a whole step leaves
    :math:`W^{(1)} = 0` and does not move :math:`\varphi`:
    :math:`\nabla_\varphi J_\lambda(\theta^0) = 0`."""
    model, X, y = _null_problem(kind, hidden)
    lam = 1.01 * float(model.loss.null_gradient(X, y).abs().max())
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    grads = _one_step(model, X, y, lam)

    assert (model.selector.weight == 0.0).all()
    for name, parameter in model.named_parameters():
        if name == "layers.0.weight":
            continue
        assert grads[name].abs().max() <= 1e-14
        assert_close(parameter.detach(), before[name], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", ARCHITECTURES)
def test_null_point_is_left_below_lambda0(kind, hidden):
    r"""At :math:`\lambda = 0.99 \lambda_0`, the same step switches on at least
    one weight of :math:`W^{(1)}`."""
    model, X, y = _null_problem(kind, hidden)
    lam = 0.99 * float(model.loss.null_gradient(X, y).abs().max())
    _one_step(model, X, y, lam)
    assert (model.selector.weight != 0.0).any()
