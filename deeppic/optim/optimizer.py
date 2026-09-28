r"""Proximal optimisers.

A proximal optimiser splits the objective in two: the gradient handles the loss,
and the proximal operator of the penalty is applied to the parameters right
after the update. The penalty is therefore never differentiated, which is what
lets an :math:`\ell_1` term reach exact zeros instead of hovering around them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["ProxGenAdam"]


class ProxGenAdam(Optimizer):
    r"""Adam followed by the proximal step of a weighted :math:`\ell_1` penalty,
    in the PROXGEN family.

    A group with level :math:`\lambda` and weights :math:`\omega` is penalised by
    :math:`\lambda \sum_{ij} \omega_{ij} |W_{ij}|`. Every step is an Adam update
    followed by the proximal operator of that penalty, taken in the metric Adam
    induces:

    .. math::
        W \leftarrow \operatorname{Soft}_{D \lambda \omega}
        \left( W - D \hat{m} \right),
        \qquad D = \frac{\mathrm{lr}}{\sqrt{\hat{v}} + \varepsilon},

    elementwise, with :math:`\operatorname{Soft}_\tau(w) = \operatorname{sign}(w)
    \max(|w| - \tau, 0)`. A coordinate Adam moves cautiously is thresholded as
    cautiously.

    The penalty stands in for any weight decay, and the loss the closure returns
    must hold no penalty term of its own, or the level is applied twice.

    :param params: parameters to optimise, or parameter groups; a group may
        carry its own ``lam`` and ``penalty_weights``.
    :param lr: learning rate.
    :param betas: decay rates of the first and second moments.
    :param eps: term added to the denominator for numerical stability.
    :param lam: default regularisation level :math:`\lambda`; ``0`` leaves the
        group unpenalised.
    :param penalty_weights: default weights :math:`\omega`, a float or any tensor
        broadcastable against the parameters of the group; a column of shape
        ``(rows, 1)`` weights every row on its own.
    :param bias_correction: whether the moments are corrected for having started
        at zero, :math:`\hat{m} = m / (1 - \beta_1^t)` and
        :math:`\hat{v} = v / (1 - \beta_2^t)`.
    :raises ValueError: if ``lr``, ``betas``, ``eps`` or ``lam`` is out of range.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lam: float = 0.0,
        penalty_weights: float | Tensor = 1.0,
        bias_correction: bool = True,
    ) -> None:
        lr = float(lr)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"lr must be finite and strictly positive, got {lr!r}.")

        try:
            beta1, beta2 = (float(beta) for beta in betas)
        except (TypeError, ValueError):
            raise ValueError(
                f"betas must be a pair of floats, got {betas!r}."
            ) from None
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Both betas must lie in [0, 1), got {betas!r}.")

        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and strictly positive, got {eps!r}.")

        lam = float(lam)
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError(f"lam must be finite and non-negative, got {lam!r}.")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                lam=lam,
                penalty_weights=penalty_weights,
                bias_correction=bool(bias_correction),
            ),
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        r"""Restore the optimiser, filling in the keys an older state lacks.

        :param state: state to restore, as produced by :meth:`state_dict`.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("lam", 0.0)
            group.setdefault("penalty_weights", 1.0)
            group.setdefault("bias_correction", True)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor] | None = None,
    ) -> Tensor | None:
        r"""Take one Adam step, then apply the proximal operator.

        :param closure: callable re-evaluating the model and returning the loss.
        :return: the loss the closure returned, or ``None`` if none was given.
        :raises RuntimeError: if a parameter carries a sparse gradient.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["lam"]
            penalty_weights = group["penalty_weights"]
            bias_correction = group["bias_correction"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("ProxGenAdam does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if bias_correction:
                    m_used = m / (1.0 - beta1**t)
                    v_used = v / (1.0 - beta2**t)
                else:
                    m_used = m
                    v_used = v

                denom = v_used.sqrt().add(eps)
                parameter.addcdiv_(m_used, denom, value=-lr)

                if lam > 0.0:
                    threshold = (lr / denom) * (lam * penalty_weights)
                    parameter.copy_(
                        parameter.sign() * (parameter.abs() - threshold).clamp_min(0.0)
                    )

        return loss
