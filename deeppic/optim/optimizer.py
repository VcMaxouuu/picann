"""Preconditioned proximal gradient: one Adam step, then one proximal step."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .penalty import Penalty, Zero

__all__ = ["ProxAdam"]


class ProxAdam(Optimizer):
    r"""ProxGen (Yun et al., NeurIPS 2021) with the Adam preconditioner.

    Minimises

    .. math::
        F(\theta) = f(\theta) + \sum_j P_{\lambda_j}\!\left(\theta^{(j)}\right)

    where :math:`f` is smooth and possibly nonconvex and each
    :math:`P_\lambda` is proper, lower semicontinuous and admits a computable
    proximal operator. At each iteration and for each group,

    .. math::
        \theta_{t+1} = \operatorname{prox}^{C_t + \delta I}_{\alpha_t P_\lambda}
                       \bigl(\theta_t - \alpha_t (C_t + \delta I)^{-1} m_t\bigr),

    with :math:`m_t` the bias-corrected first moment and :math:`C_t` the
    diagonal Adam preconditioner :math:`\operatorname{diag}(\sqrt{\hat v_t})`.
    The scaled proximal operator is then evaluated with the step
    :math:`\alpha_t / (C_t + \delta)`.

    Groups with ``lam == 0``, or carrying :class:`~deeppic.optim.Zero`, take a
    plain Adam step: the proximal operator is the identity there and is
    skipped. A single instance therefore drives both the penalized and the
    free tensors of a network, each group with its own weight, and ``lam`` may
    be raised between iterations to walk a continuation path.

    Parameters
    ----------
    params : iterable
        Iterable of tensors, or of parameter-group dictionaries. Recognised
        group keys, beyond the mandatory ``params``, are ``lr``, ``betas``,
        ``eps``, ``weight_decay``, ``lam`` and ``penalty``; each falls back to
        the corresponding constructor argument when absent.
    lr : float, default=1e-3
        Step size :math:`\alpha_t`. Schedulable: a
        :mod:`torch.optim.lr_scheduler` object drives it as it would for Adam.
    betas : tuple of two float, default=(0.9, 0.999)
        Exponential decay rates of the first and second moment estimates.
    eps : float, default=1e-5
        The :math:`\delta` added to the preconditioner. Larger values than
        Adam's usual default are often wanted here: :math:`\delta` bounds the
        effective proximal step by :math:`\alpha_t/\delta`, so it decides how
        far a coordinate with a vanishing second moment may be thresholded.
    weight_decay : float, default=0.0
        Decoupled weight decay, applied multiplicatively before the gradient
        step and hence before the proximal step; ``0`` disables it.
    lam : float, default=0.0
        Default penalty weight, non-negative. ``0`` disables the proximal
        step for the group.
    penalty : Penalty or None, default=None
        Default penalty for groups that do not supply one. ``None`` becomes
        :class:`~deeppic.optim.Zero`, for which the update is plain Adam.

    Attributes
    ----------
    gradient_mapping : dict[Tensor, Tensor] or None
        Per-parameter stationarity measure of the last step; ``None`` before
        the first one. See the note below.
    residual : Tensor or None
        The same quantity aggregated over every parameter, as a 0-dim tensor.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range, or if a group carries a negative
        ``lam``.
    TypeError
        If a group carries something other than a :class:`Penalty`.

    Notes
    -----
    **Exact sparsity.** On return from :meth:`step` the penalized tensors hold
    the output of ``penalty.prox`` and their zeros are exact, not merely
    small. The support can therefore be read off with ``!= 0``.

    **The gradient mapping measures stationarity.** :attr:`gradient_mapping`
    reports, for each parameter,

    .. math::
        G_t = \alpha_t^{-1} (C_t + \delta I)(\theta_t - \theta_{t+1}),

    which rearranging the optimality condition of the proximal subproblem
    identifies as the computable part of an element of
    :math:`\partial \hat F(\theta_{t+1})` -- precisely the quantity Theorem 1
    of the paper bounds. It is the criterion to stop on, in place of the
    variation of :math:`F`: two consecutive objectives can agree to five
    digits in the middle of a monotone climb, whereas a mapping can only be
    small if the iterate has stopped moving.

    Two factors in that expression are what make it a property of the point
    rather than of the run. Dividing by :math:`\alpha_t` removes the schedule:
    without it, annealing the learning rate shrinks the displacement and the
    raw norm falls to zero on its own, so the test would fire because the
    anneal ran its course. Multiplying by :math:`C_t + \delta I` removes the
    preconditioner: the bare displacement over :math:`\alpha_t` is
    :math:`\hat m_t / (\sqrt{\hat v_t} + \delta)`, which a decaying second
    moment inflates by up to :math:`1/\delta`, putting a floor under the
    quantity that has nothing to do with convergence.

    It is returned per parameter rather than aggregated because the two are
    not interchangeable under a gauge constraint. Fixing a normalisation on
    the downstream layers makes the problem constrained, and at a constrained
    stationary point the full-space mapping converges to the Lagrange
    multiplier rather than to zero. The blocks the constraint does not act on
    -- the penalized weights among them -- keep their unconstrained
    stationarity condition, so their mapping does vanish. Restricting to those
    is what gives a criterion that can actually be met.

    **The preconditioner is routed through the penalty.** For a separable
    penalty such as :class:`~deeppic.optim.L1` the proximal step differs from
    coordinate to coordinate and the closed form is unaffected. A group penalty
    is not separable that way: a step varying inside a block would threshold
    every entry against the same block norm with a different cutoff, which is
    the proximal operator of nothing and leaves stray survivors in blocks that
    should die outright. :meth:`~deeppic.optim.Penalty.condition` therefore gets
    the last word on the metric, and it is applied *before* the gradient step so
    that the quadratic subproblem and the step that produced its centre share
    one preconditioner.

    **No line search, no monotonicity.** Adam is not a descent method, and
    neither is this. The objective may rise from one iteration to the next;
    what the paper guarantees is convergence to a stationary point of
    :math:`F` under a decaying step size, not a decreasing sequence. Anneal
    ``lr`` rather than watching for a single upward step.

    **State under external rescaling.** :meth:`transform_state` realigns the
    moment estimates after a gauge retraction. Skipping it leaves the moving
    averages in the previous gauge for the whole horizon of their exponential
    windows, which is several hundred iterations at the default ``betas``.
    The mapping is left in the gauge of the step that produced it, so read it
    before retracting if it is to be compared against the parameters.

    References
    ----------
    .. [1] J. Yun, A. C. Lozano and E. Yang, "Adaptive Proximal Gradient
       Methods for Structured Neural Networks", NeurIPS, 2021.
    .. [2] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic
       Optimization", ICLR, 2015.

    Examples
    --------
    >>> w1, others = model.parameter_groups()
    >>> optimizer = ProxAdam(
    ...     [
    ...         {"params": [w1], "penalty": GroupL1(), "lam": 0.05},
    ...         {"params": others},
    ...     ],
    ...     lr=1e-2,
    ...     eps=1e-4,
    ... )
    >>> for _ in range(n_iterations):
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(inputs), target=targets)
    ...     loss.backward()
    ...     optimizer.step()
    ...     mapping = optimizer.gradient_mapping[w1]
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        lam: float = 0.0,
        penalty: Penalty | None = None,
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

        weight_decay = float(weight_decay)
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be finite and non-negative, got {weight_decay!r}."
            )

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                lam=lam,
                penalty=Zero() if penalty is None else penalty,
            ),
        )

        self._mapping: dict[Tensor, Tensor] | None = None

    # ------------------------------------------------------------------ #
    # Public state
    # ------------------------------------------------------------------ #
    @property
    def gradient_mapping(self) -> dict[Tensor, Tensor] | None:
        r"""Stationarity measure of the last step, ``None`` before the first.

        Maps each stepped parameter to
        :math:`\alpha_t^{-1}(C_t + \delta I)(\theta_t - \theta_{t+1})`, shaped
        like the parameter; see the class notes. Parameters that carried no
        gradient are absent.

        It is computed from the displacement, so in ``float32`` it carries the
        cancellation of :math:`\theta_{t+1} - \theta_t`: about four significant
        digits for a step of relative size :math:`10^{-3}`, less as the
        learning rate anneals. Plenty for a stopping test, not enough to
        compare two mappings that agree to five digits.
        """
        return self._mapping

    @property
    def residual(self) -> Tensor | None:
        """The gradient mapping aggregated over every parameter, as a norm.

        Convenient for a quick read, but under a gauge constraint it floors at
        the Lagrange multiplier and never reaches zero; prefer restricting
        :attr:`gradient_mapping` to the blocks the constraint leaves free.
        """
        if not self._mapping:
            return None
        squared = sum(mapping.pow(2).sum() for mapping in self._mapping.values())
        return squared.sqrt()

    # ------------------------------------------------------------------ #
    # Groups
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_group(group: dict, index: int) -> None:
        """Check the penalty-specific keys, normalising ``lam`` to a float."""
        where = f" in group {index}"

        if not isinstance(group["penalty"], Penalty):
            raise TypeError(
                f"penalty must be a Penalty instance{where}, got "
                f"{type(group['penalty']).__name__}."
            )

        lam = float(group["lam"])
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError(
                f"lam must be finite and non-negative{where}, got {lam!r}."
            )
        group["lam"] = lam

    def add_param_group(self, param_group: dict) -> None:
        """Add a group, then validate its penalty and weight.

        Also reached during construction, so every group is validated.
        """
        super().add_param_group(param_group)
        self._validate_group(self.param_groups[-1], len(self.param_groups) - 1)

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], Tensor] | None = None
    ) -> Tensor | None:
        """Perform one preconditioned proximal-gradient step.

        Parameters
        ----------
        closure : callable, optional
            ``closure() -> Tensor`` evaluating the smooth part :math:`f` --
            the penalties are the optimizer's business, not the closure's.
            It must clear the gradients and call ``backward()``. Nothing
            here re-evaluates the objective, so an ordinary
            ``zero_grad / backward / step`` loop is the expected usage and
            the closure exists only for callers that want the
            :mod:`torch.optim` convention.

        Returns
        -------
        Tensor or None
            Whatever the closure returned, ``None`` when there is none. Note
            that this is the smooth part alone, evaluated *before* the step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mapping: dict[Tensor, Tensor] = {}

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            lam, weight_decay = float(group["lam"]), group["weight_decay"]
            penalty = group["penalty"]

            active = lam > 0.0 and not isinstance(penalty, Zero)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**t)
                denom = (v / (1 - beta2**t)).sqrt_().add_(eps)  # C_t + delta

                # The penalty has the last word on the metric: a group penalty
                # only admits a closed form when the preconditioner is constant
                # within each block. Applied here, before the gradient step, so
                # that both halves of the update share one preconditioner.
                if active:
                    denom = penalty.condition(denom)

                previous = parameter.clone()

                if weight_decay != 0.0:
                    parameter.mul_(1 - lr * weight_decay)  # decoupled
                parameter.addcdiv_(m_hat, denom, value=-lr)  # theta_hat

                if active:
                    parameter.copy_(penalty.prox(parameter, lam, lr / denom))

                # (1 / alpha) (C_t + delta I) (theta_t - theta_{t+1}), read off
                # the optimality condition of the proximal subproblem. Both
                # factors matter: 1 / alpha strips the schedule, C_t + delta I
                # strips the preconditioner, which would otherwise inflate the
                # quantity by up to 1 / delta once the second moment decays.
                mapping[parameter] = denom * (previous - parameter) / lr

        self._mapping = mapping or None
        return loss

    # ------------------------------------------------------------------ #
    # Change internal state after external parameter rescaling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def transform_state(self, scales: dict[Tensor, Tensor | float]) -> None:
        r"""Realign the optimizer state after a gauge retraction.

        Under :math:`\theta \mapsto s\,\theta` the loss is unchanged, so the
        gradient scales as :math:`1/s` and the moments must follow:
        :math:`m \mapsto m/s` and :math:`v \mapsto v/s^2`. Without this the
        moving averages stay in the previous gauge for the whole horizon of
        their exponential windows.

        Parameters
        ----------
        scales : dict[Parameter, Tensor | float]
            Maps each retracted parameter to the factor **already applied** to
            it, e.g. ``{W1: A, W2: 1 / A}``. Values may be scalars or any shape
            broadcastable to the parameter, which allows per-row gauges.
            Parameters the optimizer has never stepped are skipped.

        Raises
        ------
        KeyError
            If a parameter is not held by this optimizer.
        ValueError
            If a scale is not strictly positive.
        """
        known = {p for group in self.param_groups for p in group["params"]}

        for parameter, scale in scales.items():
            if parameter not in known:
                raise KeyError(
                    "transform_state received a parameter that is not held by "
                    f"{type(self).__name__}."
                )
            state = self.state.get(parameter)
            if not state:
                continue  # never stepped, nothing to realign
            scale = torch.as_tensor(
                scale, dtype=parameter.dtype, device=parameter.device
            )
            if not bool((scale > 0).all()):
                raise ValueError(
                    "Retraction scales must be strictly positive; a null scale "
                    "would collapse the gauge and leave the state undefined."
                )
            state["exp_avg"].div_(scale)
            state["exp_avg_sq"].div_(scale * scale)
