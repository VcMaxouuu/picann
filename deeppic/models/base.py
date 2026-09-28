r"""Base class for variable-selection multilayer perceptrons.

The network maps the inputs to a scalar linear predictor, which the loss reads
through its link. Selection comes from the penalty

.. math::
    \lambda \sum_{k=1}^{p_1} s_k(\theta) \, \|W^{(1)}_{k \cdot}\|_1,
    \qquad
    s_k(\theta) = \|\mathbf{a}_{\cdot k}(\theta)\|_\infty
    = \max_i \left| \frac{\partial f_\theta(x_i)}{\partial h^{(1)}_{ik}} \right|,

on the weights of the first layer. The level :math:`\lambda` is common to every
unit and calibrated under :math:`H_0`: a variable survives only if it moves the
loss more than pure noise does.

Rescaling a unit, its row of :math:`W^{(1)}` and its bias by :math:`c` and its
column of :math:`W^{(2)}` by :math:`1 / c`, leaves the fitted function and the
objective alone and divides :math:`s_k` by :math:`c`. The fit therefore works
on the representatives where :math:`s_k = 1`, that is on the equivalent
problem

.. math::
    \min_\theta \; \ell(\theta) + \lambda \|W^{(1)}\|_1
    \quad \text{subject to} \quad s_k(\theta) = 1, \quad k = 1, \dots, p_1,

a lasso on the first layer. The deeper layers see the constraint through its
normal :math:`\nabla s_k`, weighted by its Lagrange multiplier
:math:`\mu_k = \lambda \|W^{(1)}_{k \cdot}\|_1`, and every step ends on the
constraint again. A variable leaves the network exactly when its column of
:math:`W^{(1)}` is zero.

A column a phase leaves at zero is dropped from the next one, which computes
with the other columns only, and is taken back if its zero fails the
first-order condition of the lasso, the very test the calibration of
:math:`\lambda` rests on. Every column dropped at the end of the fit passes
that test at the calibrated level.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Self, cast

import torch
from torch import Generator, Tensor
from torch.nn import LeakyReLU, Linear, Module, Parameter, Sequential, functional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deeppic.loss.loss import Loss
from deeppic.optim.optimizer import ProxGenAdam
from deeppic.optim.regularisation import calibrate_lambda
from deeppic.utils.path import geometric_path

__all__ = ["SelectionMLP"]


# On an accelerator, convergence is checked every _CHECK_EVERY epochs only:
# every check reads values back to the host, which synchronises the device.
_CHECK_EVERY = 10
# A phase stops once the objective has moved, relatively, by at most the
# tolerance over the last _WINDOW epochs.
_WINDOW = 50
# The learning rate is halved once the checks have seen the objective go more
# than _PATIENCE epochs without improving, never below _MIN_LR_RATIO times its
# start.
_PATIENCE = 30
_MIN_LR_RATIO = 1e-2
# Moments of the optimiser, with the power of the gradient they scale like.
_MOMENTS = (("exp_avg", 1), ("exp_avg_sq", 2))
# A phase that ends with dropped columns failing the first-order condition is
# run again with them; the last of these runs takes every column.
_MAX_KKT_ROUNDS = 5


def _check_every(device: torch.device) -> int:
    """Return the number of epochs between two convergence checks on ``device``."""
    return _CHECK_EVERY

_TRACE_COLUMNS = (
    ("iter", 5),
    ("loss", 10),
    ("penalty", 10),
    ("objective", 11),
    ("lr", 9),
    ("change", 9),
    ("sel", 4),
)


def _trace_header() -> str:
    """Build the header and rule of the per-epoch trace."""
    cells = "  ".join(f"{name:>{width}}" for name, width in _TRACE_COLUMNS)
    rule = "  ".join("-" * width for _, width in _TRACE_COLUMNS)
    return f"    {cells}\n    {rule}"


class SelectionMLP(Module):
    r"""Base class for a multilayer perceptron performing variable selection.

    The network is the :class:`~torch.nn.Sequential` :attr:`layers`, where every
    :class:`~torch.nn.Linear` layer but the last is followed by a
    :class:`~torch.nn.LeakyReLU`, and the last maps to the linear predictor
    :math:`\eta`. The first layer, :attr:`selector`, holds the weights
    :math:`W^{(1)}` the penalty applies to: its column :math:`j` gathers every
    weight through which variable :math:`j` enters the network. Without hidden
    layers the network is the linear model :math:`\eta = X \beta + b`.

    The LeakyReLU is positively homogeneous, so rescaling a unit against the
    layer above leaves the fitted function alone, and its derivative never
    vanishes, so no activation blocks the signal the sensitivity is read from.

    :param input_dim: number of input variables :math:`p`.
    :param hidden_dims: width of each hidden layer, from the first to the last.
        ``None`` or empty reduces the network to a linear model.
    :param loss: loss driving the fit, and whose null distribution calibrates
        :math:`\lambda`.
    :param alpha: level at which :math:`\lambda` is calibrated.
    :param lambda_: common regularisation level :math:`\lambda`. ``None``
        calibrates it from the data, any other value is used as is and
        overrides the calibration, leaving ``alpha`` unused. Recorded in
        :attr:`fixed_lambda`.
    :param standardize: whether :meth:`fit` centres and scales the inputs on the
        training statistics, which :meth:`predict` then replays on every call.
    :param lr: learning rate every phase starts from.
    :param tol: relative change of the objective over :data:`_WINDOW` epochs
        below which the last phase stops; earlier phases stop at ``100 * tol``.
    :param n_phases: number of phases the fit walks through.
    :param lambda_ratio: fraction of :math:`\lambda` the first phase runs at.
    :param verbose: whether the fit reports the level it uses, then every phase
        and every epoch.
    :param log_every: number of epochs between two log lines. The first and
        the last epoch of a phase are always reported, as is the one a phase
        converges on.
    :raises ValueError: if ``lambda_`` is negative, or if ``log_every`` is not
        positive.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None,
        loss: Loss,
        alpha: float = 0.05,
        lambda_: float | None = None,
        standardize: bool = True,
        lr: float = 1e-2,
        tol: float = 1e-4,
        n_phases: int = 5,
        lambda_ratio: float = 1e-2,
        verbose: bool = False,
        log_every: int = 50,
    ) -> None:
        super().__init__()
        if lambda_ is not None and lambda_ < 0.0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        if log_every < 1:
            raise ValueError(f"log_every must be positive, got {log_every}")

        self.loss = loss
        self.alpha = alpha

        dims = [input_dim, *(hidden_dims or ()), 1]
        modules: list[Module] = []
        for width, next_width in zip(dims, dims[1:]):
            modules += [Linear(width, next_width), LeakyReLU()]
        self.layers = Sequential(*modules[:-1])

        self.fixed_lambda = lambda_ is not None
        self.register_buffer(
            "lambda_", torch.tensor(0.0 if lambda_ is None else float(lambda_))
        )

        self.standardize = standardize
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))

        self.lr = lr
        self.tol = tol
        self.n_phases = n_phases
        self.lambda_ratio = lambda_ratio
        self.verbose = verbose
        self.log_every = log_every

    @property
    def selector(self) -> Linear:
        r"""Return the first layer, whose weights :math:`W^{(1)}` are penalised.

        :return: first :class:`~torch.nn.Linear` layer of :attr:`layers`.
        """
        return cast(Linear, self.layers[0])

    @property
    def selected(self) -> Tensor:
        r"""Report the variables the network still uses.

        Variable :math:`j` is selected when a coefficient of the matrix
        :math:`B_{k \cdot} = s_k W^{(1)}_{k \cdot}` is not zero in its column.
        :math:`B` is the same all along an orbit of the rescalings, and
        :math:`B = W^{(1)}` on the representative :meth:`fit` hands back, where
        :math:`s_k = 1`. A unit with :math:`s_k = 0` reaches neither the output
        nor the objective: a variable entering through such units only is not
        selected, whatever its weights.

        The mask is read off the weights, without the data: a unit counts when
        a path of non-zero weights links it to the output. Every slope of the
        LeakyReLU is positive, so this is :math:`s_k > 0`, exactly with one
        hidden layer and up to exact cancellations between paths otherwise.

        :return: mask of the selected variables, of shape ``(p,)``.
        """
        live = self._reaching_output()[0]
        return ((self.selector.weight != 0.0) & live.unsqueeze(-1)).any(dim=0)

    @property
    def selected_indices(self) -> list[int]:
        r"""Report the indices of the variables the network still uses.

        :return: indices of the selected variables, in ascending order.
        """
        return self.selected.nonzero(as_tuple=True)[0].tolist()

    @torch.no_grad()
    def count_effective_weights(self) -> tuple[list[tuple[int, int]], float]:
        r"""Count, layer by layer, the weights the fitted function still uses.

        A non-zero weight counts only if the unit it leaves still depends on the
        inputs and the unit it enters still reaches the output. A unit whose
        incoming weights are all discarded outputs a constant, which its
        outgoing weights only add to the bias of the next layer; the loss of a
        unit cascades through the layers, so a network whose :math:`W^{(1)}` is
        zero has no effective weight at all. Only :math:`W^{(1)}` is penalised,
        so the deeper layers seldom hold an exact zero, but a unit whose
        outgoing weights all vanish, :math:`s_k = 0`, uses none of its incoming
        ones. Biases are not counted.

        :return: pair ``(effective, total)`` for every linear layer of
            :attr:`layers`, from the first to the last, and the fraction of all
            weights that are effective.
        """
        counts = []
        w1 = self.selector.weight
        reached = torch.ones(w1.shape[1], dtype=torch.bool, device=w1.device)
        for layer, reaching in zip(self._linears, self._reaching_output()):
            mask = (layer.weight != 0.0) & reached & reaching.unsqueeze(-1)
            counts.append((int(mask.sum()), mask.numel()))
            reached = mask.any(dim=1)

        effective, total = map(sum, zip(*counts))
        return counts, effective / total

    def forward(self, X: Tensor) -> Tensor:
        r"""Map the inputs to the linear predictor.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :return: linear predictor :math:`\eta`, of shape ``(n,)``.
        """
        return self.layers(X).squeeze(-1)

    def standardize_inputs(self, X: Tensor) -> Tensor:
        r"""Centre and scale the inputs with the statistics of the fit.

        The statistics are the identity, zero mean and unit scale, until
        :meth:`fit` estimates them, and stay so when :attr:`standardize` is off.

        :param X: raw design matrix, of shape ``(n, p)``.
        :return: standardised design matrix, of shape ``(n, p)``.
        """
        return (X - self.input_mean) / self.input_scale

    # ------------------------------------------------------------------
    # Sensitivity, its gradient and the constraint
    # ------------------------------------------------------------------

    @property
    def _linears(self) -> list[Linear]:
        """Return the linear layers :math:`W^{(1)}, \\dots, W^{(L)}`, in order."""
        return [m for m in self.layers if isinstance(m, Linear)]

    @property
    def _slopes(self) -> list[float]:
        """Return the negative slope of every hidden activation, in order."""
        return [m.negative_slope for m in self.layers if isinstance(m, LeakyReLU)]

    @torch.no_grad()
    def _reaching_output(self) -> list[Tensor]:
        r"""Mark, layer by layer, the units a path of non-zero weights links to
        the output.

        :return: one mask per linear layer, over the units it outputs, from the
            first layer to the last, whose single unit is the output itself.
        """
        linears = self._linears
        reaching = [torch.ones(1, dtype=torch.bool, device=linears[-1].weight.device)]
        for layer in reversed(linears[1:]):
            reaching.append(((layer.weight != 0.0) & reaching[-1].unsqueeze(-1)).any(dim=0))
        return reaching[::-1]

    def _layers_above(self, h1: Tensor) -> tuple[Tensor, list[Tensor]]:
        r"""Run the network from the pre-activation of its first layer.

        :param h1: pre-activation :math:`h^{(1)}`, of shape ``(n, p_1)``.
        :return: linear predictor :math:`\eta`, of shape ``(n,)``, and the
            pre-activations :math:`h^{(1)}, \dots, h^{(L-1)}` of the hidden
            layers, empty for a linear model.
        """
        preactivations = [h1]
        h = h1
        for module in self.layers[1:]:
            h = module(h)
            if isinstance(module, Linear):
                preactivations.append(h)
        return h.squeeze(-1), preactivations[:-1]

    def _forward_with_preactivations(
        self, X: Tensor, weight: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        r"""Compute the linear predictor and keep the pre-activations of the
        hidden layers.

        :param X: standardised design matrix, of shape ``(n, p)``, or the
            columns of it that ``weight`` reads.
        :param weight: weights standing for :math:`W^{(1)}`, one column per
            column of ``X``; ``None`` uses :math:`W^{(1)}` itself.
        :return: linear predictor :math:`\eta`, of shape ``(n,)``, attached to
            the graph, and the pre-activations :math:`h^{(1)}, \dots,
            h^{(L-1)}`, empty for a linear model.
        """
        first = self.selector
        h1 = functional.linear(X, first.weight if weight is None else weight, first.bias)
        return self._layers_above(h1)

    def _masks(
        self, preactivations: list[Tensor], rows: Tensor | None = None
    ) -> list[Tensor]:
        r"""Read the derivatives :math:`D^{(l)} = \sigma_l'(h^{(l)})` of the
        activations, detached.

        The convention, slope ``1`` if and only if :math:`h > 0`, is the one of
        the backward pass of :class:`~torch.nn.LeakyReLU`.

        :param preactivations: pre-activations of the hidden layers.
        :param rows: observations to keep; ``None`` keeps them all.
        :return: one mask per hidden layer, of shape ``(n, p_l)`` or
            ``(len(rows), p_l)``.
        """
        masks = []
        for h, slope in zip(preactivations, self._slopes):
            h = h.detach() if rows is None else h.detach()[rows]
            masks.append((h > 0.0).to(h.dtype).mul_(1.0 - slope).add_(slope))
        return masks

    def _jacobian(self, masks: list[Tensor]) -> Tensor:
        r"""Compute the rows :math:`\mathbf{a}_i^\top = W^{(L)} D^{(L-1)}_i
        W^{(L-1)} \cdots W^{(2)} D^{(1)}_i` for the observations the masks
        were read at.

        :param masks: masks of the hidden layers, as returned by :meth:`_masks`.
        :return: :math:`a`, of shape ``(rows, p_1)``.
        """
        linears = self._linears
        v = linears[-1].weight.expand(masks[0].shape[0], -1)
        for layer in range(len(masks) - 1, -1, -1):
            v = v * masks[layer]
            if layer > 0:
                v = v @ linears[layer].weight
        return v

    def _linearised_output(self, masks: list[Tensor], U: Tensor) -> Tensor:
        r"""Feed virtual inputs through the network linearised at a few
        observations.

        With the masks frozen and the biases dropped, the network at
        observation :math:`i` is the linear map
        :math:`u \mapsto \mathbf{a}_i^\top u`. One virtual row per unit
        :math:`k`, read at the observation :math:`i^*_k` attaining
        :math:`s_k` and fed :math:`u^{(k)} = \mu_k \operatorname{sign}(a_{i^*_k
        k}) \, e_k`, gives

        .. math::
            \sum_k \mathbf{a}_{i^*_k}^\top u^{(k)} = \sum_k \mu_k s_k,
            \qquad \mu_k = \lambda \|W^{(1)}_{k \cdot}\|_1,

        whose gradient is :math:`\sum_k \mu_k \nabla s_k` and reaches
        :math:`W^{(2)}, \dots, W^{(L)}` only. Two units may share an
        observation: the rows are summed, so the repetition is harmless.

        :param masks: masks of the hidden layers at the observations the
            virtual rows are read at, one row each.
        :param U: virtual inputs, one row per virtual row, of shape
            ``(rows, p_1)``.
        :return: the scalar :math:`\sum_r \mathbf{a}_{i_r}^\top U_{r \cdot}`.
        """
        linears = self._linears
        v = U * masks[0]
        for layer in range(1, len(masks)):
            v = (v @ linears[layer].weight.T) * masks[layer]
        return (v @ linears[-1].weight.T).sum()

    @torch.no_grad()
    def sensitivity(self, X: Tensor) -> Tensor:
        r"""Compute the sensitivity :math:`s_k(\theta)` of every unit of the first
        layer.

        :math:`a_{ik} = \partial f_\theta(x_i) / \partial h^{(1)}_{ik}` measures
        what the rest of the network does with unit :math:`k` at observation
        :math:`i`, and :math:`s_k = \max_i |a_{ik}|` is the sup-norm of that
        column. A linear model has a single unit, of sensitivity ``1``.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :return: :math:`s(\theta)`, of shape ``(p_1,)``.
        """
        return self._sensitivity(X)

    @torch.no_grad()
    def _sensitivity(self, X: Tensor, weight: Tensor | None = None) -> Tensor:
        r"""Compute :math:`s(\theta)`, possibly on the active columns only.

        :param X: standardised design matrix, or the columns ``weight`` reads.
        :param weight: weights standing for :math:`W^{(1)}`; ``None`` uses
            :math:`W^{(1)}` itself.
        :return: :math:`s(\theta)`, of shape ``(p_1,)``.
        """
        _, preactivations = self._forward_with_preactivations(X, weight)
        if not preactivations:
            return self.selector.weight.new_ones(1)
        return self._jacobian(self._masks(preactivations)).abs().amax(dim=0)

    def _backward(
        self, X: Tensor, y: Tensor, lam: float, weight: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        r"""Fill the gradients of the objective and return the loss and the
        sensitivity at the current iterate.

        1. The forward pass of the loss also yields the masks
           :math:`D^{(l)}_i`, hence :math:`a_{ik} = \partial \eta_i / \partial
           h^{(1)}_{ik}` on every row, at the cost of a forward pass of the
           layers above the first one, and with it :math:`s_k` and the
           observation :math:`i^*_k` attaining it. Every observation goes
           through the network on its own, so :math:`a` is also
           :math:`G / r`, with :math:`G = \partial \ell / \partial h^{(1)}` and
           :math:`r = \partial \ell / \partial \eta`, but that ratio is
           undefined where :math:`r_i = 0`, a saturated probability for one.
        2. The linearised network, read on the :math:`p_1` rows
           :math:`i^*_1, \dots, i^*_{p_1}`, outputs :math:`\sum_k \mu_k s_k`
           with :math:`\mu_k = \lambda \|W^{(1)}_{k \cdot}\|_1`.
        3. One backward pass of the loss plus that output leaves
           :math:`\nabla \ell` in every parameter and adds
           :math:`\sum_k \mu_k \nabla s_k` to :math:`W^{(2)}, \dots, W^{(L)}`:
           the normal of the constraint :math:`s_k = 1`, weighted by its
           multiplier.

        :math:`W^{(1)}` receives :math:`\nabla \ell` only: its penalty is left to
        the proximal step. Every shape is fixed and no branch depends on a
        value, so the device never waits for the host.

        :param X: standardised design matrix, of shape ``(n, p)``, or the
            columns ``weight`` reads.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda^{(m)}` of the phase.
        :param weight: weights standing for :math:`W^{(1)}`; ``None`` uses
            :math:`W^{(1)}` itself.
        :return: detached loss, and detached sensitivity :math:`s(\theta^t)`, of
            shape ``(p_1,)``.
        """
        w1 = self.selector.weight if weight is None else weight
        eta, preactivations = self._forward_with_preactivations(X, w1)
        loss = self.loss(eta, y)
        if not preactivations:
            loss.backward()
            return loss.detach(), w1.new_ones(1)

        with torch.no_grad():
            masks = self._masks(preactivations)
            a = self._jacobian(masks)
            s, rows = a.abs().max(dim=0)

        objective = loss
        if lam > 0.0:
            with torch.no_grad():
                signs = a.gather(0, rows.unsqueeze(0)).squeeze(0).sign()
                U = torch.diag(lam * w1.abs().sum(dim=1) * signs)
            multiplier = self._linearised_output([mask[rows] for mask in masks], U)
            objective = loss + multiplier
        objective.backward()

        return loss.detach(), s

    @torch.no_grad()
    def _retract(
        self, s: Tensor, optimizer: ProxGenAdam, weight: Tensor | None = None
    ) -> Tensor:
        r"""Bring the iterate back onto the constraint :math:`s_k = 1`.

        Unit :math:`k` is rescaled by :math:`c_k = s_k`:

        .. math::
            W^{(1)}_{k \cdot} \mapsto c_k W^{(1)}_{k \cdot}, \qquad
            b^{(1)}_k \mapsto c_k b^{(1)}_k, \qquad
            W^{(2)}_{\cdot k} \mapsto W^{(2)}_{\cdot k} / c_k,

        which changes neither the fitted function nor the objective. What is
        attached to a rescaled weight follows it: its gradient and the first
        moment of the optimiser scale like :math:`1 / c`, the second moment
        like :math:`1 / c^2`. Most steps move :math:`s_k` only slightly and
        :math:`c_k` is then close to one, but :math:`s_k` is not continuous: it
        jumps, by up to the ratio of the two slopes of the LeakyReLU, when an
        activation flips at the observation attaining its maximum. Moments left
        at the old scale would then meet gradients at the new one, and the
        oversized step would make the loss spike.

        A unit whose sensitivity vanishes is left alone: its rescaling is
        undefined. Only the moments the optimiser keeps are moved: the group of
        :math:`W^{(1)}` has no first moment.

        :param s: sensitivity at the current iterate, of shape ``(p_1,)``.
        :param optimizer: optimiser of the phase.
        :param weight: weights standing for :math:`W^{(1)}`; ``None`` uses
            :math:`W^{(1)}` itself.
        :return: the sensitivity on the constraint, one on every unit whose
            sensitivity does not vanish.
        """
        first, second = self._linears[:2]
        w1 = first.weight if weight is None else weight
        c = torch.where(s > 0.0, s, torch.ones_like(s))

        targets = [(w1, c.unsqueeze(-1)), (second.weight, c.reciprocal())]
        if first.bias is not None:
            targets.append((first.bias, c))

        for parameter, factor in targets:
            parameter.mul_(factor)
            if parameter.grad is not None:
                parameter.grad.div_(factor)
            state = optimizer.state.get(parameter, {})
            for key, power in _MOMENTS:
                if key in state:
                    state[key].div_(factor if power == 1 else factor**power)
        return s / c

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _optimizer(self, lam: float, weight: Tensor | None = None) -> ProxGenAdam:
        r"""Build the optimiser of a phase at the level :math:`\lambda^{(m)}`.

        :math:`W^{(1)}` has a group of its own, penalised at ``lam`` and without
        first moment, :math:`\beta_1 = 0`: its proximal step then tests the
        gradient at the current iterate against the threshold, and nothing
        carries it across zero once it is cut. Every other parameter takes a
        plain Adam step, with no penalty and no weight decay.

        :param lam: level :math:`\lambda^{(m)}` of the phase.
        :param weight: weights standing for :math:`W^{(1)}`; ``None`` uses
            :math:`W^{(1)}` itself.
        :return: optimiser whose first group holds :math:`W^{(1)}` alone.
        """
        w1 = self.selector.weight
        others = [parameter for parameter in self.parameters() if parameter is not w1]
        penalised = w1 if weight is None else weight
        return ProxGenAdam(
            [{"params": [penalised], "lam": lam, "betas": (0.0, 0.999)}, {"params": others}],
            self.lr,
        )

    @torch.no_grad()
    def calibrate(
        self,
        X: Tensor,
        n_simulations: int = 1000,
        batch_size: int | None = None,
        generator: Generator | None = None,
    ) -> Tensor:
        r"""Calibrate :math:`\lambda` on the design matrix and store it.

        :math:`\lambda` is a constant of the problem, not a parameter: it depends
        on ``X`` and on the null distribution of the loss, never on the weights,
        and is therefore computed once, before the optimisation starts.

        A level supplied at construction wins: nothing is simulated and
        :attr:`lambda_` is left untouched. Clearing :attr:`fixed_lambda` hands
        the choice back to the calibration.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :param n_simulations: number of Monte-Carlo draws.
        :param batch_size: number of draws evaluated at once; ``None`` evaluates
            them all in a single batch.
        :param generator: pseudo-random generator, for reproducible draws.
        :return: the level the fit will use, stored in :attr:`lambda_`.
        """
        if not self.fixed_lambda:
            value = calibrate_lambda(
                self.loss, X, self.alpha, n_simulations, batch_size, generator
            )
            self.lambda_ = value.to(self.lambda_)
        return self.lambda_

    def fit_phase(
        self,
        X: Tensor,
        y: Tensor,
        lam: float,
        tol: float,
        n_epochs: int,
    ) -> Tensor:
        r"""Run one phase at the fixed level :math:`\lambda^{(m)}`.

        The phase solves
        :math:`\min \ell + \lambda^{(m)} \|W^{(1)}\|_1` subject to
        :math:`s_k = 1`, and owns its optimiser and its scheduler. Every epoch:

        1. :meth:`_backward` fills the gradients at the current iterate;
        2. :meth:`_retract` brings the iterate, its gradients and the moments
           of the optimiser back onto the constraint;
        3. the objective is recorded there, where the penalty is
           :math:`\lambda^{(m)} \|W^{(1)}\|_1`;
        4. :math:`W^{(1)}` takes a proximal lasso step at level
           :math:`\lambda^{(m)}`, in its own parameter group and without
           momentum, and the other parameters a plain Adam step.

        The phase stops once the objective has moved, relatively, by at most
        ``tol`` over the last :data:`_WINDOW` epochs, which is checked, and the
        scheduler stepped, every :func:`_check_every` epochs. The window
        averages out the oscillations of the steps, and the floor of the
        learning rate keeps a stalled objective from passing for a converged
        one. The phase always ends on the constraint.

        Only the active columns of :math:`W^{(1)}` are computed with: the ones
        that are not zero when the phase starts, every one from a dense start,
        and the zero ones whose first-order condition at
        :math:`\lambda^{(m)}` fails there (:meth:`_violations`). When the phase
        ends, the dropped columns are tested again, and those that fail are
        taken back for another run of the phase, from where it stopped; after
        :data:`_MAX_KKT_ROUNDS` runs, the last one takes every column. Every
        dropped column therefore passes, at the end, the test a proximal step
        on the whole of :math:`W^{(1)}` would put it to: the fixed points of
        the phase are the ones it would have without dropping anything.
        Within :meth:`fit`, only the last phase, at the calibrated level, ends
        with this test: an earlier phase hands its dropped columns to the next
        one, which tests them at its own level when it starts.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda^{(m)}` the phase runs at.
        :param tol: relative change of the objective over :data:`_WINDOW`
            epochs below which the phase stops.
        :param n_epochs: number of gradient steps each run of the phase may
            take; a run that reaches it without stopping raises a
            :class:`RuntimeWarning`.
        :return: objective on the constraint at every epoch of every run, in
            order, of shape ``(epochs taken,)``.
        """
        return self._fit_phase(X, y, lam, tol, n_epochs, final=True)

    def _fit_phase(
        self,
        X: Tensor,
        y: Tensor,
        lam: float,
        tol: float,
        n_epochs: int,
        final: bool,
    ) -> Tensor:
        r"""Run one phase, as :meth:`fit_phase` documents.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda^{(m)}` the phase runs at.
        :param tol: relative change of the objective over :data:`_WINDOW`
            epochs below which the phase stops.
        :param n_epochs: number of gradient steps each run of the phase may
            take.
        :param final: whether the dropped columns are tested when the phase
            ends, and taken back for another run if they fail.
        :return: objective on the constraint at every epoch of every run.
        """
        with torch.no_grad():
            active = (self.selector.weight != 0.0).any(dim=0)
        if not bool(active.all()):
            active = active | self._violations(X, y, lam, active)
        histories = [self._run_phase(X, y, lam, tol, n_epochs, active)]

        for run in range(1, _MAX_KKT_ROUNDS + 1):
            if not final or bool(active.all()):
                break
            violations = self._violations(X, y, lam, active)
            if not bool(violations.any()):
                break
            if self.verbose:
                print(
                    f"    {int(violations.sum())} dropped column(s) fail the "
                    f"first-order condition: run {run + 1} of the phase"
                )
            if run == _MAX_KKT_ROUNDS:
                active = torch.ones_like(active)
            else:
                active = active | violations
            histories.append(self._run_phase(X, y, lam, tol, n_epochs, active))
        return torch.cat(histories)

    def _violations(self, X: Tensor, y: Tensor, lam: float, active: Tensor) -> Tensor:
        r"""Find the dropped columns of :math:`W^{(1)}` whose zero fails the
        first-order condition at the level ``lam``.

        A dropped column :math:`j` is zero, and a proximal step at level
        :math:`\lambda` on the whole of :math:`W^{(1)}` keeps it there if and
        only if :math:`|g_{kj}| \leq \lambda s_k` for every unit :math:`k`, where
        :math:`\mathbf{g}_{\cdot j} = G^\top X_j` and :math:`G = \partial \ell /
        \partial h^{(1)}`: the zero-thresholding test of the calibration. The
        dropped columns do not enter :math:`h^{(1)}`, so one pass through the
        active network and one product with the dropped columns suffice.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda` of the test.
        :param active: mask of the columns computed with, of shape ``(p,)``;
            every other column of :math:`W^{(1)}` is zero.
        :return: mask of the dropped columns that fail the test, of shape
            ``(p,)``.
        """
        first = self.selector
        with torch.no_grad():
            h1 = functional.linear(X[:, active], first.weight[:, active], first.bias)
        h1.requires_grad_()
        eta, preactivations = self._layers_above(h1)
        (G,) = torch.autograd.grad(self.loss(eta, y), h1)

        with torch.no_grad():
            if preactivations:
                s = self._jacobian(self._masks(preactivations)).abs().amax(dim=0)
            else:
                s = G.new_ones(1)
            dropped = ~active
            scores = G.T @ X[:, dropped]
            violations = torch.zeros_like(active)
            violations[dropped] = (scores.abs() > lam * s.unsqueeze(-1)).any(dim=0)
        return violations

    def _run_phase(
        self,
        X: Tensor,
        y: Tensor,
        lam: float,
        tol: float,
        n_epochs: int,
        active: Tensor,
    ) -> Tensor:
        r"""Run the epochs of a phase on the active columns of :math:`W^{(1)}`.

        With columns dropped, the active columns of ``X`` are gathered once
        and the phase works on a copy of the active columns of
        :math:`W^{(1)}`, written back when it ends; the dropped columns stay at
        zero. :attr:`selector` keeps its full shape throughout.

        :param X: standardised design matrix, of shape ``(n, p)``.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda^{(m)}` the phase runs at.
        :param tol: relative change of the objective over :data:`_WINDOW`
            epochs below which the phase stops.
        :param n_epochs: number of gradient steps the run may take.
        :param active: mask of the columns computed with, of shape ``(p,)``.
        :return: objective on the constraint at every epoch of the run.
        """
        first = self.selector
        pruned = not bool(active.all())
        if pruned:
            X = X[:, active]
            w1 = Parameter(first.weight.detach()[:, active].clone())
        else:
            w1 = first.weight
        try:
            return self._descend(X, y, lam, tol, n_epochs, w1)
        finally:
            if pruned:
                with torch.no_grad():
                    first.weight[:, active] = w1

    def _descend(
        self,
        X: Tensor,
        y: Tensor,
        lam: float,
        tol: float,
        n_epochs: int,
        w1: Tensor,
    ) -> Tensor:
        r"""Take the steps of a run, from the current iterate, until the
        objective settles or the budget runs out.

        :param X: standardised design matrix, or its active columns.
        :param y: targets, of shape ``(n,)``.
        :param lam: level :math:`\lambda^{(m)}` the phase runs at.
        :param tol: relative change of the objective over :data:`_WINDOW`
            epochs below which the run stops.
        :param n_epochs: number of gradient steps the run may take.
        :param w1: weights standing for :math:`W^{(1)}`, one column per
            column of ``X``.
        :return: objective on the constraint at every epoch of the run.
        """
        check_every = _check_every(X.device)
        optimizer = self._optimizer(lam, w1)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=_PATIENCE // check_every,
            min_lr=self.lr * _MIN_LR_RATIO,
        )
        penalized = optimizer.param_groups[0]
        hidden = len(self._linears) > 1
        tiny = torch.finfo(X.dtype).tiny

        history = torch.empty(n_epochs, dtype=X.dtype, device=X.device)
        change = math.inf
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss, s = self._backward(X, y, lam, w1)
            if hidden:
                s = self._retract(s, optimizer, w1)
            with torch.no_grad():
                penalty = lam * torch.dot(s, w1.abs().sum(dim=1))
            history[epoch] = value = loss + penalty

            converged = False
            if (epoch + 1) % check_every == 0:
                if epoch >= _WINDOW:
                    moved = (history[epoch - _WINDOW] - value).abs()
                    change = float(moved / value.abs().clamp_min(tiny))
                converged = change <= tol
                scheduler.step(float(value))

            last = converged or epoch + 1 == n_epochs
            if self.verbose and (epoch == 0 or last or (epoch + 1) % self.log_every == 0):
                live = self._reaching_output()[0].unsqueeze(-1)
                print(
                    f"    {epoch + 1:>5d}  {float(loss):>10.4f}"
                    f"  {float(penalty):>10.4f}  {float(value):>11.4f}"
                    f"  {penalized['lr']:>9.1e}  {change:>9.2e}"
                    f"  {int(((w1 != 0.0) & live).any(dim=0).sum()):>4d}"
                )
            if converged:
                if self.verbose:
                    print(f"    converged at epoch {epoch + 1}")
                return history[: epoch + 1]

            penalized["penalty_weights"] = s.unsqueeze(-1)
            optimizer.step()

        if hidden:
            # The budget ran out right after a step: hand back the representative
            # on the constraint all the same.
            optimizer.zero_grad()
            self._retract(self._sensitivity(X, w1), optimizer, w1)
        warnings.warn(
            f"phase at lambda = {lam:.3e} stopped after {n_epochs} epochs without "
            f"converging: relative change of the objective {change:.2e} over "
            f"{_WINDOW} epochs, tol {tol:.1e}",
            RuntimeWarning,
            stacklevel=5,
        )
        return history

    def _check_data(self, X: Tensor, y: Tensor) -> Tensor:
        r"""Check the data handed to :meth:`fit` and bring the targets to the
        dtype and device of ``X``.

        Only the shapes and the finiteness of the targets are checked here; a
        subclass whose loss is defined on fewer values restricts them further,
        after calling this method.

        :param X: raw design matrix.
        :param y: targets.
        :return: targets, of shape ``(n,)``, in the dtype and on the device of
            ``X``.
        :raises ValueError: if ``X`` is not of shape ``(n, p)``, or if ``y`` is
            not of shape ``(n,)`` or holds a non-finite value.
        """
        p = self.selector.in_features
        if X.ndim != 2 or X.shape[1] != p:
            raise ValueError(f"X must be of shape (n, {p}), got {tuple(X.shape)}")
        n = X.shape[0]
        if y.shape != (n,):
            raise ValueError(
                f"y must be of shape ({n},), one target per row of X, "
                f"got {tuple(y.shape)}"
            )
        y = y.to(X)
        if not torch.isfinite(y).all():
            raise ValueError("y must hold finite values only")
        return y

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        n_epochs: int = 1000,
        generator: Generator | None = None,
    ) -> Self:
        r"""Calibrate :math:`\lambda` unless one was supplied, then walk the
        phases up to it.

        Phase :math:`m` runs at :math:`\lambda^{(m)}`, on an increasing geometric
        grid from ``lambda_ratio`` times :math:`\lambda` to :math:`\lambda`, each
        phase starting from the weights the previous one reached. Every phase
        but the last stops at ``100 * tol``: it only has to get the weights in
        position for the next level.

        Optimisation is full-batch: :math:`\lambda` is calibrated against the
        gradient the whole sample produces under :math:`H_0`, and mini-batches
        would put the two on different scales.

        With :attr:`standardize`, the statistics of ``X`` are estimated first and
        kept: the calibration then sees the standardised design matrix, on which
        :math:`\lambda` is comparable across variables, and :meth:`predict`
        replays the same transformation. A column with no variance is left
        centred, its scale held at one rather than dividing by zero.

        :param X: raw design matrix, of shape ``(n, p)``.
        :param y: targets, of shape ``(n,)``.
        :param n_epochs: number of gradient steps each phase may take.
        :param generator: pseudo-random generator, for the calibration draws.
        :return: this model.
        :raises ValueError: if ``X`` or ``y`` fails :meth:`_check_data`.
        """
        y = self._check_data(X, y)
        if self.standardize:
            with torch.no_grad():
                self.input_mean = X.mean(dim=0)
                scale = X.std(dim=0, unbiased=False)
                self.input_scale = scale.masked_fill(scale == 0.0, 1.0)
        X = self.standardize_inputs(X)

        level = float(self.calibrate(X, generator=generator))
        path = geometric_path(level, self.n_phases, self.lambda_ratio).tolist()
        if self.verbose:
            origin = "supplied" if self.fixed_lambda else "calibrated"
            print(
                f"lambda {origin}: {level:.6e}\n"
                f"Problem     lasso on W1 subject to s_k = 1, restored after "
                f"every step\n"
                f"Path        {self.n_phases} phase{'s' if self.n_phases > 1 else ''}"
                f", warm-started, from lambda = {path[0]:.6e}\n"
                f"Optimizer   ProxGenAdam, one instance per phase\n"
                f"LR          {self.lr:.2e}, halved on plateau, down to "
                f"{self.lr * _MIN_LR_RATIO:.2e}\n"
                f"Stop        relative change of the objective over {_WINDOW} "
                f"epochs <= tol ({100 * self.tol:.1e}, last phase "
                f"{self.tol:.1e}), checked every {_check_every(X.device)} "
                f"epoch(s)\n"
                f"Columns     zero columns dropped from the next phase, taken "
                f"back if their first-order condition fails\n"
            )

        self.train()
        for phase, lam in enumerate(path, start=1):
            tol = self.tol if phase == self.n_phases else 100 * self.tol
            if self.verbose:
                print(
                    f"  phase {phase}/{self.n_phases}"
                    f"  lambda {lam:.6e}  tol {tol:.1e}\n{_trace_header()}"
                )
            self._fit_phase(X, y, lam, tol, n_epochs, final=phase == self.n_phases)
        return self

    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        r"""Predict the mean of the response.

        :param X: raw design matrix, of shape ``(n, p)``, standardised on the
            way in with the statistics of the fit.
        :return: :math:`\mu = g(\eta)`, of shape ``(n,)``.
        """
        return self.loss.link(self(self.standardize_inputs(X)))
