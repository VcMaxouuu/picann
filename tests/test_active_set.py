r"""Columns of :math:`W^{(1)}` dropped between phases, and the first-order test
that takes them back."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.testing import assert_close

from _helpers import DEPTHS, LOSSES, full_step, has_hidden, make_problem

ARCHITECTURES = (None, *DEPTHS)


def _with_zero_columns(kind, hidden, columns=(1, 3, 4), seed=0):
    model, X, y = make_problem(kind, hidden, n=60, p=8, seed=seed)
    with torch.no_grad():
        model.selector.weight[:, list(columns)] = 0.0
    return model, X, y


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", ARCHITECTURES)
def test_violations_match_a_proximal_step_on_every_column(kind, hidden):
    r"""A dropped column fails the test exactly when one step of the algorithm
    on the whole of :math:`W^{(1)}` would switch it on."""
    for lam in (0.02, 0.1, 0.3):
        model, X, y = _with_zero_columns(kind, hidden)
        active = (model.selector.weight != 0.0).any(dim=0)
        violations = model._violations(X, y, lam, active)

        full = copy.deepcopy(model)
        full_step(full, X, y, lam, full._optimizer(lam))
        switched_on = (full.selector.weight != 0.0).any(dim=0) & ~active
        assert torch.equal(violations, switched_on)


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("hidden", ARCHITECTURES)
def test_dropped_columns_pass_the_test_when_the_phase_ends(kind, hidden, monkeypatch):
    model, X, y = _with_zero_columns(kind, hidden)
    runs = []
    run_phase = model._run_phase

    def spy(X, y, lam, tol, n_epochs, active):
        runs.append(active.clone())
        return run_phase(X, y, lam, tol, n_epochs, active)

    monkeypatch.setattr(model, "_run_phase", spy)
    lam = 0.3 * float(model.loss.null_gradient(X, y).abs().max())
    model.fit_phase(X, y, lam=lam, tol=1e-4, n_epochs=2000)

    last = runs[-1]
    assert not last.all()
    assert not model._violations(X, y, lam, last).any()
    assert (model.selector.weight[:, ~last] == 0.0).all()


@pytest.mark.parametrize("kind", LOSSES)
def test_fit_ends_with_the_test_at_the_calibrated_level(kind, monkeypatch):
    r"""After :meth:`fit`, every column the last run dropped passes the
    zero-thresholding test at :math:`\lambda^{\mathrm{DB}}_\alpha`: the
    selection is the one of a fixed point of the full problem."""
    model, X, y = make_problem(kind, (6, 4), n=80, p=30, seed=2)
    runs = []
    run_phase = model._run_phase

    def spy(X, y, lam, tol, n_epochs, active):
        runs.append((lam, active.clone()))
        return run_phase(X, y, lam, tol, n_epochs, active)

    monkeypatch.setattr(model, "_run_phase", spy)
    model.fit(X, y, n_epochs=1500, generator=torch.Generator().manual_seed(0))

    lam, last = runs[-1]
    assert lam == pytest.approx(float(model.lambda_))
    assert not last.all()
    Z = model.standardize_inputs(X)
    assert not model._violations(Z, y, lam, last).any()
    assert (model.selector.weight[:, ~last] == 0.0).all()


def test_failing_column_is_taken_back(monkeypatch):
    r"""A column whose zero fails the test at the end of a run joins the active
    set, and the phase runs again."""
    model, X, y = _with_zero_columns("gaussian", (5, 4), columns=(0, 3, 4))
    lam = 0.05 * float(model.loss.null_gradient(X, y).abs().max())
    violations = model._violations
    calls = []

    def blind_first(X, y, lam, active):
        calls.append(active.clone())
        found = violations(X, y, lam, active)
        return torch.zeros_like(found) if len(calls) == 1 else found

    monkeypatch.setattr(model, "_violations", blind_first)
    history = model.fit_phase(X, y, lam=lam, tol=1e-4, n_epochs=2000)

    assert len(calls) >= 2
    first_run, after = calls[0], calls[1]
    assert not first_run[0]
    assert (model.selector.weight[:, 0] != 0.0).any()
    assert history.numel() > 0
    assert torch.isfinite(history).all()


def test_selector_keeps_its_parameter_and_shape():
    model, X, y = _with_zero_columns("gaussian", (5,))
    weight = model.selector.weight
    keys = list(model.state_dict())
    model.fit_phase(X, y, lam=0.1, tol=1e-4, n_epochs=500)
    assert model.selector.weight is weight
    assert weight.shape == (5, 8)
    assert list(model.state_dict()) == keys
    assert (weight[:, ~model.selected] == 0.0).all()


@pytest.mark.parametrize("hidden", [None, (5,), (5, 4)])
def test_pruned_phase_matches_the_full_phase(hidden, monkeypatch):
    r"""Where no dropped column would come back, dropping them changes nothing
    but the rounding: same selection, same objective, same weights."""
    model, X, y = _with_zero_columns("gaussian", hidden, columns=(3, 4, 5))
    lam = 0.5 * float(model.loss.null_gradient(X, y).abs().max())
    full = copy.deepcopy(model)

    pruned_history = model.fit_phase(X, y, lam=lam, tol=1e-6, n_epochs=3000)

    every = torch.ones(X.shape[1], dtype=torch.bool)
    monkeypatch.setattr(full, "_violations", lambda X, y, lam, active: torch.zeros_like(active))
    full_history = full._run_phase(X, y, lam, 1e-6, 3000, every)

    assert torch.equal(model.selected, full.selected)
    assert pruned_history.numel() == full_history.numel()
    assert_close(pruned_history, full_history, rtol=1e-8, atol=0.0)
    for mine, theirs in zip(model.parameters(), full.parameters()):
        assert_close(mine, theirs, rtol=1e-6, atol=1e-9)
