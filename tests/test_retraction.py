r"""The retraction onto the section :math:`s = \mathbf 1`: exact, covariant, and
the reason why the iterates depend on the orbit only."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.testing import assert_close

from _helpers import (
    DEPTHS,
    LOSSES,
    backward_gradients,
    full_step,
    make_problem,
    objective,
    rescale,
)

LAM = 0.5


def _off_section(kind, hidden, seed=0):
    model, X, y = make_problem(kind, hidden, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    c = torch.empty(hidden[0], dtype=X.dtype).uniform_(0.2, 5.0, generator=generator)
    rescale(model, c)
    return model, X, y


def _gradients(model):
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_retraction_is_exact(kind, hidden):
    model, X, y = _off_section(kind, hidden)
    with torch.no_grad():
        eta = model(X)
    value = objective(model, X, y, LAM)
    optimizer = model._optimizer(LAM)
    optimizer.zero_grad()
    _, s = model._backward(X, y, LAM)
    assert (s - 1.0).abs().max() > 1e-2

    on_section = model._retract(s, optimizer)
    with torch.no_grad():
        assert_close(model(X), eta, rtol=1e-12, atol=1e-14)
    assert_close(objective(model, X, y, LAM), value, rtol=1e-12, atol=0.0)
    ones = torch.ones_like(s)
    assert_close(on_section, ones, rtol=0.0, atol=0.0)
    assert_close(model.sensitivity(X), ones, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_transported_gradients_match_recomputed(kind, hidden):
    r"""The gradients the retraction transports are the gradients at the
    retracted point, for the loss and for the multiplier term alike."""
    transported, recomputed = {}, {}
    for lam in (0.0, LAM):
        model, X, y = _off_section(kind, hidden)
        optimizer = model._optimizer(LAM)
        optimizer.zero_grad()
        _, s = model._backward(X, y, lam)
        model._retract(s, optimizer)
        transported[lam] = _gradients(model)
        recomputed[lam] = backward_gradients(model, X, y, lam)

    for name in transported[0.0]:
        assert_close(transported[0.0][name], recomputed[0.0][name], rtol=1e-10, atol=1e-14)
        assert_close(
            transported[LAM][name] - transported[0.0][name],
            recomputed[LAM][name] - recomputed[0.0][name],
            rtol=1e-9,
            atol=1e-13,
        )


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_moments_are_transported(kind, hidden):
    r"""A weight multiplied by :math:`c` has its first moment divided by
    :math:`c` and its second moment by :math:`c^2`; a weight divided by
    :math:`c` the converse; any other moment is left alone."""
    model, X, y = make_problem(kind, hidden)
    optimizer = model._optimizer(LAM)
    for _ in range(3):
        full_step(model, X, y, LAM, optimizer)
    generator = torch.Generator().manual_seed(3)
    rescale(model, torch.empty(hidden[0], dtype=X.dtype).uniform_(0.2, 5.0, generator=generator))

    optimizer.zero_grad()
    _, s = model._backward(X, y, LAM)
    assert (s - 1.0).abs().max() > 1e-2
    before = {
        parameter: {key: value.clone() for key, value in optimizer.state[parameter].items()
                    if key in ("exp_avg", "exp_avg_sq")}
        for parameter in model.parameters()
    }
    model._retract(s, optimizer)

    first, second = model._linears[:2]
    factors = {first.weight: s.unsqueeze(-1), first.bias: s, second.weight: s.reciprocal()}
    for parameter in model.parameters():
        factor = factors.get(parameter, torch.ones((), dtype=s.dtype))
        for key, power in (("exp_avg", 1), ("exp_avg_sq", 2)):
            if key in before[parameter]:
                assert_close(
                    optimizer.state[parameter][key],
                    before[parameter][key] / factor**power,
                    rtol=1e-13,
                    atol=0.0,
                )


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_iterates_depend_only_on_the_orbit(kind, hidden):
    r"""Started from :math:`\theta` and from :math:`G_c \theta`, the algorithm
    takes the same steps."""
    model, X, y = make_problem(kind, hidden)
    other = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(7)
    rescale(other, torch.empty(hidden[0], dtype=X.dtype).uniform_(0.2, 5.0, generator=generator))

    optimizers = model._optimizer(LAM), other._optimizer(LAM)
    for _ in range(20):
        full_step(model, X, y, LAM, optimizers[0])
        full_step(other, X, y, LAM, optimizers[1])
    for mine, theirs in zip(model.parameters(), other.parameters()):
        assert_close(mine, theirs, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", DEPTHS)
def test_iterates_with_moments_depend_only_on_the_orbit(kind, hidden):
    r"""Rescaling the iterate together with its moments, as the retraction does,
    changes none of the later steps."""
    model, X, y = make_problem(kind, hidden)
    optimizer = model._optimizer(LAM)
    for _ in range(10):
        full_step(model, X, y, LAM, optimizer)

    other, other_optimizer = copy.deepcopy((model, optimizer))
    generator = torch.Generator().manual_seed(11)
    c = torch.empty(hidden[0], dtype=X.dtype).uniform_(0.2, 5.0, generator=generator)
    rescale(other, c, other_optimizer)

    for _ in range(10):
        full_step(model, X, y, LAM, optimizer)
        full_step(other, X, y, LAM, other_optimizer)
    for mine, theirs in zip(model.parameters(), other.parameters()):
        assert_close(mine, theirs, rtol=1e-9, atol=1e-12)
