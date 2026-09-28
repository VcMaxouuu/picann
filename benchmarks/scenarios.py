r"""Simulated data sets shared by the benchmarks and the selection harnesses.

Every generator is seeded and returns the raw design matrix, the targets and
the sorted list of relevant variables. The models standardise the inputs
themselves.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def linear(n: int, p: int, s: int, seed: int, amplitude: float = 1.0) -> tuple[Tensor, Tensor, list[int]]:
    r"""Linear signal on ``s`` random variables, unit Gaussian noise.

    :math:`y = \sum_{j \in S} \beta_j x_j + \varepsilon`, with
    :math:`\beta_j = \pm` ``amplitude``, signs drawn at random.
    """
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=generator)
    support = torch.randperm(p, generator=generator)[:s]
    signs = torch.randint(0, 2, (s,), generator=generator) * 2.0 - 1.0
    y = X[:, support] @ (amplitude * signs) + torch.randn(n, generator=generator)
    return X, y, sorted(support.tolist())


def nonlinear(n: int, p: int, seed: int) -> tuple[Tensor, Tensor, list[int]]:
    r"""Additive nonlinear signal on three random variables, unit Gaussian noise.

    .. math::
        y = 2 \tanh(2 x_{j_1}) + 2 \left(|x_{j_2}| - \sqrt{2 / \pi}\right)
            + 2 \sin(2 x_{j_3}) + \varepsilon.

    The second component is even: it has no linear association with the
    response, and is only visible once the network has learnt it.
    """
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=generator)
    support = torch.randperm(p, generator=generator)[:3]
    a, b, c = (X[:, j] for j in support)
    signal = 2.0 * torch.tanh(2.0 * a) + 2.0 * (b.abs() - math.sqrt(2.0 / math.pi))
    signal = signal + 2.0 * torch.sin(2.0 * c)
    y = signal + torch.randn(n, generator=generator)
    return X, y, sorted(support.tolist())


SCENARIOS = {
    # name: (generator, kwargs, hidden_dims)
    "linear_p100": (linear, dict(n=200, p=100, s=3), (32, 32)),
    "linear_p1000": (linear, dict(n=200, p=1000, s=3), (32, 32)),
    "linear_p5000": (linear, dict(n=200, p=5000, s=3), (32, 32)),
    "nonlinear_p100": (nonlinear, dict(n=500, p=100), (32, 32)),
}


def make(name: str, seed: int) -> tuple[Tensor, Tensor, list[int], tuple[int, ...]]:
    r"""Draw the data set of scenario ``name`` for ``seed``.

    :return: design matrix, targets, relevant variables and hidden widths.
    """
    generator, kwargs, hidden = SCENARIOS[name]
    X, y, support = generator(seed=seed, **kwargs)
    return X, y, support, hidden
