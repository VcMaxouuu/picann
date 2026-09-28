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

    A group with :math:`\beta_1 = 0` keeps no first moment: :math:`\hat m` is
    the gradient itself, and the step is a proximal RMSProp step. A zero
    coordinate then stays at zero if and only if :math:`|g| \leq \lambda
    \omega`, whatever :math:`D`, on the gradient of the current step.

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

        The bias corrections are folded into scalars, as in
        :class:`torch.optim.Adam`: :math:`D = \mathrm{lr}_t / (\sqrt{v} /
        \sqrt{1 - \beta_2^t} + \varepsilon)` with :math:`\mathrm{lr}_t =
        \mathrm{lr} / (1 - \beta_1^t)`, the same step as with the corrected
        moments. The moments of all the parameters of a group are updated at
        once.

        :param closure: callable re-evaluating the model and returning the loss.
        :return: the loss the closure returned, or ``None`` if none was given.
        :raises RuntimeError: if a parameter carries a sparse gradient.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("ProxGenAdam does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                if beta1 > 0.0 and "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                state["step"] += 1

                params.append(parameter)
                grads.append(parameter.grad)
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])
                if beta1 > 0.0:
                    exp_avgs.append(state["exp_avg"])
            if params:
                self._update(group, params, grads, exp_avgs, exp_avg_sqs, steps)

        return loss

    @staticmethod
    def _update(
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        steps: list[int],
    ) -> None:
        r"""Update the moments of a group, then its parameters.

        :param group: parameter group, with its hyperparameters.
        :param params: parameters of the group that carry a gradient.
        :param grads: their gradients.
        :param exp_avgs: their first moments, empty if :math:`\beta_1 = 0`.
        :param exp_avg_sqs: their second moments.
        :param steps: number of steps each has taken, this one included.
        """
        lr, eps, lam = group["lr"], group["eps"], group["lam"]
        beta1, beta2 = group["betas"]

        if group["bias_correction"]:
            step_sizes = [lr / (1.0 - beta1**t) for t in steps]
            corrections = [math.sqrt(1.0 - beta2**t) for t in steps]
        else:
            step_sizes = [lr] * len(steps)
            corrections = [1.0] * len(steps)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        if beta1 > 0.0:
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
            moments = exp_avgs
        else:
            moments = grads

        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denoms, corrections)
        torch._foreach_add_(denoms, eps)

        if lam == 0.0:
            torch._foreach_addcdiv_(params, moments, denoms, [-size for size in step_sizes])
            return

        level = lam * group["penalty_weights"]
        for parameter, moment, denom, size in zip(params, moments, denoms, step_sizes):
            # D = lr_t / denom, then W <- Soft_{D lam w}(W - D m), where
            # Soft_tau(x) = x - clamp(x, -tau, tau) is exactly zero for |x| <= tau.
            step = denom.reciprocal_().mul_(size)
            parameter.addcmul_(moment, step, value=-1.0)
            threshold = step.mul_(level)
            parameter.sub_(parameter.clamp(-threshold, threshold))
