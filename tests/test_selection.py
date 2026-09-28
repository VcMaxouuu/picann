r"""What :attr:`SelectionMLP.selected` reports: the variables whose invariant
coefficients :math:`B_{k\cdot} = s_k W^{(1)}_{k\cdot}` do not all vanish."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch.testing import assert_close

from _helpers import make_problem


def _only_through(model, unit, variable):
    with torch.no_grad():
        model.selector.weight.zero_()
        model.selector.weight[unit, variable] = 1.3


def test_variable_through_a_dead_unit_is_not_selected():
    r"""A unit whose column of :math:`W^{(2)}` is zero has :math:`s_k = 0`: it
    reaches neither the output nor the objective."""
    model, X, _ = make_problem("gaussian", (4,))
    _only_through(model, unit=1, variable=3)
    assert model.selected_indices == [3]
    with torch.no_grad():
        model._linears[1].weight[:, 1] = 0.0
    assert float(model.sensitivity(X)[1]) == 0.0
    assert model.selected_indices == []
    assert not model.selected.any()


def test_variable_through_a_unit_cut_further_down_is_not_selected():
    r"""Unit 0 of the first layer feeds only unit 2 of the second, whose own
    outgoing weights are zero: every path to the output is cut."""
    model, X, _ = make_problem("gaussian", (4, 3))
    _only_through(model, unit=0, variable=5)
    with torch.no_grad():
        second, third = model._linears[1:]
        second.weight[:, 0] = 0.0
        second.weight[2, 0] = 0.8
        third.weight[:, 2] = 0.0
    assert float(model.sensitivity(X)[0]) == 0.0
    assert model.selected_indices == []

    with torch.no_grad():
        third.weight[0, 2] = 0.5
    assert float(model.sensitivity(X)[0]) > 0.0
    assert model.selected_indices == [5]


def test_linear_model_selects_its_non_zero_coefficients():
    model, _, _ = make_problem("gaussian", None)
    with torch.no_grad():
        model.selector.weight.zero_()
        model.selector.weight[0, [1, 4]] = torch.tensor([0.2, -1.0], dtype=torch.float64)
    assert model.selected_indices == [1, 4]


@pytest.mark.parametrize("hidden", [(5,), (5, 4)])
def test_phase_ends_on_the_section(hidden):
    r"""A phase stopped by its budget of epochs still hands back the canonical
    representative, :math:`s = \mathbf 1`."""
    model, X, y = make_problem("gaussian", hidden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit_phase(X, y, lam=0.05, tol=1e-12, n_epochs=37)
    s = model.sensitivity(X)
    assert_close(s, torch.ones_like(s), rtol=1e-12, atol=0.0)
