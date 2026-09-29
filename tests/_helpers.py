r"""Small random problems in double precision, and the reference computations the
tests compare the implementation against.

Every reference is computed independently of the fitting code: gradients come
from autograd, and the multiplier term from autograd through the sensitivity
with its activation masks frozen.
"""

from __future__ import annotations

import torch
from torch import Tensor

from deeppic import BinaryClassifier, Regressor
from deeppic.models.base import SelectionMLP
from deeppic.optim.optimizer import ProxGenAdam

LOSSES = ("gaussian", "binary")
DEPTHS = ((5,), (5, 4), (5, 4, 3))


def standardized(n: int, p: int, generator: torch.Generator) -> Tensor:
    r"""Draw a design matrix with centred columns of unit second moment."""
    X = torch.randn(n, p, generator=generator, dtype=torch.float64)
    X = X - X.mean(dim=0)
    return X / X.pow(2).mean(dim=0).sqrt()


def make_problem(
    kind: str, hidden, n: int = 40, p: int = 6, seed: int = 0, **kwargs
) -> tuple[SelectionMLP, Tensor, Tensor]:
    r"""Build a random double-precision model and data for the loss ``kind``."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X = standardized(n, p, generator)
    if kind == "gaussian":
        model = Regressor(p, hidden, **kwargs)
        noise = torch.randn(n, generator=generator, dtype=torch.float64)
        y = X[:, 0] - X[:, 1].abs() + noise
    else:
        model = BinaryClassifier(p, hidden, **kwargs)
        y = torch.bernoulli(torch.sigmoid(2.0 * X[:, 0]), generator=generator)
    return model.double(), X, y


def has_hidden(model: SelectionMLP) -> bool:
    return len(model._linears) > 1


def full_step(
    model: SelectionMLP, X: Tensor, y: Tensor, lam: float, optimizer: ProxGenAdam
) -> tuple[Tensor, Tensor]:
    r"""Take one step of the algorithm, as :meth:`SelectionMLP.fit_phase` does:
    gradients, retraction onto :math:`s = 1`, proximal Adam step."""
    optimizer.zero_grad()
    loss, s = model._gradients(X, y, lam)
    if has_hidden(model):
        s = model._normalize(s, optimizer)
    optimizer.param_groups[0]["penalty_weights"] = s.unsqueeze(-1)
    optimizer.step()
    return loss, s


def loss_gradients(model: SelectionMLP, X: Tensor, y: Tensor) -> dict[str, Tensor]:
    r"""Return :math:`\nabla \ell` for every parameter, by autograd."""
    names, parameters = zip(*model.named_parameters())
    loss = model.loss(model(X), y)
    return dict(zip(names, torch.autograd.grad(loss, parameters)))


def backward_gradients(
    model: SelectionMLP, X: Tensor, y: Tensor, lam: float
) -> dict[str, Tensor]:
    r"""Return the gradients :meth:`SelectionMLP._gradients` leaves in the
    parameters."""
    model.zero_grad()
    model._gradients(X, y, lam)
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters()}


def multiplier_gradients(model: SelectionMLP, X: Tensor, lam: float) -> dict[str, Tensor]:
    r"""Return :math:`\sum_k \mu_k \nabla s_k`, :math:`\mu_k = \lambda
    \|W^{(1)}_{k \cdot}\|_1` held fixed, by autograd through
    :math:`s_k = \max_i |a_{ik}|` with the activation masks frozen."""
    names, parameters = zip(*model.named_parameters())
    _, masks = model._forward(X)
    a = model._jacobian(masks)
    mu = lam * model.selector.weight.detach().abs().sum(dim=1)
    penalty = (mu * a.abs().amax(dim=0)).sum()
    grads = torch.autograd.grad(penalty, parameters, allow_unused=True)
    return {
        name: torch.zeros_like(parameter) if grad is None else grad
        for name, parameter, grad in zip(names, parameters, grads)
    }


@torch.no_grad()
def objective(model: SelectionMLP, X: Tensor, y: Tensor, lam: float) -> Tensor:
    r"""Evaluate :math:`J_\lambda = \ell + \lambda \sum_k s_k \|W^{(1)}_{k
    \cdot}\|_1`."""
    s = model.sensitivity(X)
    return model.loss(model(X), y) + lam * torch.dot(
        s, model.selector.weight.abs().sum(dim=1)
    )


@torch.no_grad()
def rescale(model: SelectionMLP, c: Tensor, optimizer: ProxGenAdam | None = None) -> None:
    r"""Apply :math:`G_c`: row :math:`k` of :math:`W^{(1)}` and :math:`b^{(1)}_k`
    times :math:`c_k`, column :math:`k` of :math:`W^{(2)}` divided by
    :math:`c_k`. The moments of ``optimizer`` follow, like gradients: first
    moment divided by the factor, second moment by its square."""
    first, second = model._linears[:2]
    targets = [
        (first.weight, c.unsqueeze(-1)),
        (first.bias, c),
        (second.weight, c.reciprocal()),
    ]
    for parameter, factor in targets:
        parameter.mul_(factor)
        if optimizer is not None:
            state = optimizer.state.get(parameter, {})
            if "exp_avg" in state:
                state["exp_avg"].div_(factor)
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].div_(factor**2)


@torch.no_grad()
def null_point(model: SelectionMLP, X: Tensor, y: Tensor) -> None:
    r"""Move the model to :math:`\theta^0`: :math:`W^{(1)} = 0`, on the section
    :math:`s = 1`, with the constant output :math:`\hat\eta = \hat\tau` of the
    null fit."""
    model.selector.weight.zero_()
    if has_hidden(model):
        rescale(model, model.sensitivity(X))
    eta = model(X)
    model._linears[-1].bias.add_(model.loss.null_mle(y) - eta[0])


def preactivations(model: SelectionMLP, X: Tensor) -> tuple[Tensor, list[Tensor]]:
    r"""Return the linear predictor and the pre-activations of the hidden
    layers, attached to the graph."""
    hs, h = [], X
    for module in model.layers:
        if isinstance(module, torch.nn.LeakyReLU):
            hs.append(h)
        h = module(h)
    return h.squeeze(-1), hs


@torch.no_grad()
def is_generic(model: SelectionMLP, X: Tensor, margin: float = 1e-3) -> bool:
    r"""Tell whether the model is away from every activation boundary and from
    every tie of the maxima :math:`s_k` between observations whose masks
    differ, so that finite differences see a single smooth piece."""
    _, hs = preactivations(model, X)
    if any(h.abs().min() < margin for h in hs):
        return False
    _, masks = model._forward(X)
    a = model._jacobian(masks).abs()
    top = a.amax(dim=0)
    for k in range(a.shape[1]):
        tied = a[:, k] >= top[k] * (1.0 - 1e-12)
        rest = a[~tied, k]
        if rest.numel() and rest.max() > top[k] * (1.0 - margin):
            return False
        pattern = torch.cat([masks[0][tied, k : k + 1], *(m[tied] for m in masks[1:])], dim=1)
        if (pattern != pattern[0]).any():
            return False
    return True
