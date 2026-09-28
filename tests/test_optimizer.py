r""":class:`ProxGenAdam`: proximal RMSProp on the penalised group, plain Adam on
the others."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.testing import assert_close

from deeppic.optim.optimizer import ProxGenAdam

LR, EPS, BETA2 = 1e-2, 1e-8, 0.999


def _soft(x, threshold):
    return x.sign() * (x.abs() - threshold).clamp_min(0.0)


def test_penalised_group_is_proximal_rmsprop():
    r"""With :math:`\beta_1 = 0`: :math:`W \leftarrow
    \operatorname{Soft}_{D \lambda \omega}(W - D g)`, :math:`D =
    \mathrm{lr} / (\sqrt{\hat v} + \varepsilon)`, with the gradient of the
    current step and no memory of the earlier ones."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    weight[:, :2] = 0.0
    parameter = torch.nn.Parameter(weight.clone())
    omega = torch.rand(4, 1, generator=generator, dtype=torch.float64) + 0.5
    lam = 0.3
    optimizer = ProxGenAdam(
        [{"params": [parameter], "lam": lam, "betas": (0.0, BETA2), "penalty_weights": omega}],
        LR,
    )

    reference, v = weight.clone(), torch.zeros_like(weight)
    for t in range(1, 6):
        g = torch.randn(4, 7, generator=generator, dtype=torch.float64)
        parameter.grad = g.clone()
        optimizer.step()

        v = BETA2 * v + (1.0 - BETA2) * g * g
        D = LR / ((v / (1.0 - BETA2**t)).sqrt() + EPS)
        reference = _soft(reference - D * g, D * lam * omega)
        assert_close(parameter.detach(), reference, rtol=1e-13, atol=1e-16)


def test_threshold_test_does_not_depend_on_the_metric():
    r"""A zero weight stays at zero if and only if :math:`|g| \leq \lambda
    \omega`, whatever :math:`D`."""
    generator = torch.Generator().manual_seed(1)
    lam = 0.5
    omega = torch.rand(6, 1, generator=generator, dtype=torch.float64) + 0.5
    history = torch.randn(3, 6, 5, generator=generator, dtype=torch.float64).abs() * 10.0
    for factor in (0.9, 1.1):
        sign = torch.randn(6, 5, generator=generator, dtype=torch.float64).sign()
        g = factor * lam * omega * sign
        parameter = torch.nn.Parameter(torch.zeros(6, 5, dtype=torch.float64))
        optimizer = ProxGenAdam(
            [{"params": [parameter], "lam": lam, "betas": (0.0, BETA2),
              "penalty_weights": omega}],
            LR,
        )
        # earlier steps only shape the metric D; they are undone afterwards
        for past in history:
            parameter.grad = past.clone()
            optimizer.step()
            with torch.no_grad():
                parameter.zero_()
        parameter.grad = g
        optimizer.step()
        if factor < 1.0:
            assert (parameter == 0.0).all()
        else:
            assert (parameter != 0.0).all()
            assert (parameter.sign() == -sign).all()


@pytest.mark.parametrize("bias_correction", [True, False])
def test_unpenalised_group_is_adam(bias_correction):
    generator = torch.Generator().manual_seed(2)
    shapes = [(5, 3), (5,), (1, 5), (1,)]
    parameters = [
        torch.nn.Parameter(torch.randn(*shape, generator=generator, dtype=torch.float64))
        for shape in shapes
    ]
    twins = copy.deepcopy(parameters)
    ours = ProxGenAdam(parameters, LR, bias_correction=bias_correction)
    theirs = torch.optim.Adam(twins, LR, foreach=False)
    for _ in range(8):
        for mine, twin in zip(parameters, twins):
            g = torch.randn(mine.shape, generator=generator, dtype=torch.float64)
            mine.grad, twin.grad = g.clone(), g.clone()
        ours.step()
        if bias_correction:
            theirs.step()
        else:
            # Adam without bias correction, written out
            for twin in twins:
                state = theirs.state.setdefault(twin, {})
                m = state.setdefault("m", torch.zeros_like(twin))
                v = state.setdefault("v", torch.zeros_like(twin))
                m.mul_(0.9).add_(twin.grad, alpha=0.1)
                v.mul_(BETA2).addcmul_(twin.grad, twin.grad, value=1.0 - BETA2)
                with torch.no_grad():
                    twin.addcdiv_(m, v.sqrt().add(EPS), value=-LR)
        for mine, twin in zip(parameters, twins):
            assert_close(mine, twin, rtol=1e-12, atol=1e-15)


def test_phase_optimizer_groups():
    r"""The penalised group holds :math:`W^{(1)}` alone, without first moment;
    no group carries a weight decay, a Nesterov or any other modification of
    the gradient."""
    from _helpers import make_problem

    model, _, _ = make_problem("gaussian", (5, 4))
    optimizer = model._optimizer(0.3)
    penalised, others = optimizer.param_groups
    assert penalised["params"] == [model.selector.weight]
    assert penalised["betas"][0] == 0.0
    assert penalised["lam"] == 0.3
    assert others["lam"] == 0.0
    expected = [q for q in model.parameters() if q is not model.selector.weight]
    assert len(others["params"]) == len(expected)
    assert all(a is b for a, b in zip(others["params"], expected))
    allowed = {"params", "lr", "betas", "eps", "lam", "penalty_weights", "bias_correction"}
    for group in optimizer.param_groups:
        assert set(group) - {"initial_lr"} <= allowed
