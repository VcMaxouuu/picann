r"""Euler identities: :math:`\ell` is invariant under the rescaling of every unit
of the first layer, and :math:`s_k` is positively homogeneous of degree one in
every deeper weight matrix once the masks are frozen."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from _helpers import DEPTHS, LOSSES, backward_gradients, loss_gradients, make_problem

LAM = 0.7


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_euler_identity_of_the_loss(kind, hidden):
    r""":math:`\langle W^{(1)}_{k\cdot}, \mathbf g_k\rangle + b^{(1)}_k
    \partial_{b^{(1)}_k}\ell - \langle W^{(2)}_{\cdot k},
    \nabla_{W^{(2)}_{\cdot k}}\ell\rangle = 0` for every unit :math:`k`."""
    model, X, y = make_problem(kind, hidden)
    grads = loss_gradients(model, X, y)
    first, second = model._linears[:2]
    terms = torch.stack(
        [
            (first.weight * grads["layers.0.weight"]).sum(dim=1),
            first.bias * grads["layers.0.bias"],
            -(second.weight * grads["layers.2.weight"]).sum(dim=0),
        ]
    ).detach()
    residual = terms.sum(dim=0)
    assert (residual.abs() <= 1e-12 * terms.abs().sum(dim=0) + 1e-15).all()


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_euler_identity_of_the_multiplier(kind, hidden):
    r""":math:`\langle W^{(l)}, \nabla_{W^{(l)}} P\rangle = \lambda \sum_k s_k
    \|W^{(1)}_{k\cdot}\|_1` for every :math:`l \geq 2`, and unit by unit
    :math:`\langle W^{(2)}_{\cdot k}, \nabla_{W^{(2)}_{\cdot k}} P\rangle =
    \mu_k s_k`, where :math:`P` is the multiplier term."""
    model, X, y = make_problem(kind, hidden)
    with_term = backward_gradients(model, X, y, LAM)
    without = backward_gradients(model, X, y, 0.0)
    s = model.sensitivity(X)
    mu = LAM * model.selector.weight.detach().abs().sum(dim=1)
    expected = torch.dot(mu, s)

    linears = model._linears
    for index, layer in enumerate(linears[1:], start=1):
        name = f"layers.{2 * index}.weight"
        multiplier = with_term[name] - without[name]
        assert_close((layer.weight.detach() * multiplier).sum(), expected, rtol=1e-10, atol=0.0)
        if index == 1:
            per_unit = (layer.weight.detach() * multiplier).sum(dim=0)
            assert_close(per_unit, mu * s, rtol=1e-10, atol=1e-15)
