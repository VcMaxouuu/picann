"""Shared fitting machinery for the sparse estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..calibration import lambda_pdb
from ..optim import GroupL1, L1, Loss, ProxAdam
from ..utils import gauge_scale, lambda_path
from .base import VariableSelectionMLP

__all__ = ["SparseEstimator"]

#: Final learning rate of each phase's cosine anneal, as a fraction of the
#: rate that phase started from. Nothing certifies a single ProxAdam step,
#: so a phase is made to settle by shrinking what a step is allowed to do:
#: it explores early and barely moves late, and the support read at the end
#: of a phase is one a near-stationary point carries rather than one
#: momentum happened to leave behind.
_ANNEAL_FLOOR = 1e-3

#: Iterations Adam is allowed to spend without improving on its best
#: objective before training is declared over. Adam is not a descent
#: method -- momentum makes it overshoot -- so a single flat step means
#: nothing and a one-step criterion would stop it far too early.
_ADAM_PATIENCE = 20

#: Factor by which each phase's initial learning rate shrinks relative to
#: the previous one. Each phase starts from the solution of the last and
#: has a shorter distance to travel, so it needs a shorter step; without
#: the decay every phase restarts at the full rate, which throws the
#: iterate out of the basin the previous one settled in. Restarting from
#: a multiple of the rate a phase *ended* at was tried and measured
#: worse: the plateau criterion stops phases mid-anneal, so the reached
#: rate carries no information beyond its own starting rate.
_PHASE_LR_DECAY = 0.75

#: Iterations between two rows of the verbose trace.
_TRACE_EVERY = 20

#: Columns of that trace, and the width each is printed in.
_TRACE_COLUMNS = (
    ("iter", 4), ("loss", 10), ("penalty", 10), ("objective", 11),
    ("dF/F", 9), ("gauge", 7), ("sel", 4),
)


def _trace_header() -> str:
    """Header and rule of the per-iteration trace."""
    cells = "  ".join(f"{name:>{width}}" for name, width in _TRACE_COLUMNS)
    rule = "  ".join("-" * width for _, width in _TRACE_COLUMNS)
    return f"    {cells}\n    {rule}"


def _trace_row(
    n_iter: int,
    loss: float,
    penalty: float,
    rel_err: float,
    gauge: float,
    selected: int,
) -> str:
    """One row of that trace.

    ``dF/F`` is the quantity the tolerance is tested against: the objective's
    relative move across the window, ``inf`` until the window has filled.
    """
    return (
        f"    {n_iter:>4d}  {loss:>10.4f}  {penalty:>10.4f}  "
        f"{loss + penalty:>11.4f}  {rel_err:>9.2e}  "
        f"{gauge:>7.4f}  {selected:>4d}"
    )


class SparseEstimator(nn.Module, ABC):
    r"""Penalized MLP with calibrated variable selection.

    Wraps the full training procedure around a
    :class:`~deeppic.models.VariableSelectionMLP`: standardisation of the
    inputs, calibration of the regularisation parameter, gauge retraction,
    and preconditioned proximal-gradient optimisation with
    :class:`~deeppic.optim.ProxAdam` along a continuation path.
    Subclasses fix the data-fidelity loss and the prediction semantics.

    A fitted estimator is a regular :class:`torch.nn.Module`: calling it,
    ``estimator(X)``, standardises ``X`` with the statistics stored at fit
    time and evaluates the fitted network, gradients included. The
    ``predict``-style methods wrap this forward pass in ``no_grad``.
    After a pruned fit the network only knows the selected variables, so
    the forward pass (and hence ``predict``) expects the reduced matrix
    returned by :meth:`transform`, not the full feature matrix.

    ``.to(device)`` and ``.to(dtype)`` behave as on any module, before or
    after fitting: the estimator remembers where it lives, so a later
    :meth:`fit` builds its network there and moves ``X`` and ``y`` along,
    running the calibration and the whole descent on that device.

    Parameters
    ----------
    hidden_dims : tuple of int or None, default=None
        Hidden layer widths. ``None`` fits a linear model.
    bias : bool, default=True
        Whether the last linear layer include a bias.
    lam : float or None, default=None
        Optional user-supplied regularization parameter. When None, selected
        automatically by calibration under the null.
    alpha : float, default=0.05
        Nominal level of the test for the null hypothesis. Used only if ``lam`` is None.
    n_mc : int, default=1000
        Number of Monte-Carlo replicates for the calibration.
    lr : float, default=0.01
        Learning rate the first phase starts from. Each phase cosine
        anneals to a thousandth of its own starting rate, and each starts
        from a rate below the last. Drives the whole network, penalized
        first layer included.
    adam_lr : float, default=0.01
        Learning rate of the unpenalized Adam refit on the selected
        support, which only runs when ``fit`` is called with ``prune``.
    n_phases : int, default=5
        Number of phases of the continuation path; see
        :func:`~deeppic.utils.lambda_path`. Each one warm-starts the next,
        the last carrying the calibrated weight and producing the
        selection.
    max_iter : int, default=500
        Maximum number of iterations per phase.
    min_iter : int, default=100
        Iterations a phase always runs before it is allowed to stop,
        capped at ``max_iter``. The learning rate is annealed over the
        phase, so an early quiet spell says nothing about stationarity.
    window : int, default=50
        Iterations the objective is compared across. A value at or above
        ``max_iter`` never fills and so disables early stopping.
    tol : float, default=1e-4
        Convergence tolerance. A phase stops once
        ``|F(t - window) - F(t)| / max(1, |F(t)|)`` falls below it, with
        ``F`` the penalized objective, and once ``min_iter`` iterations
        have passed. The comparison spans a window rather than a single
        step because the descent is not monotone: two consecutive
        objectives can agree to five digits halfway up a climb that a
        window sees whole. Fitting also stops early if the network
        collapses to a constant, leaving an empty support that the larger
        weights ahead cannot refill.
    standardize : bool, default=True
        Whether to centre and scale the columns of ``X`` before fitting.
        The calibration assumes standardised columns.

    Attributes
    ----------
    model_ : VariableSelectionMLP
        The fitted network. After a pruned fit its input width is the
        number of selected variables, and inputs are expected in that
        reduced space; use :meth:`transform` to build them.
    lambda_ : float
        Regularisation parameter actually used.
    lambda_data_ : float
        Zero-thresholding statistic of the observed sample: the smallest
        weight at which the null iterate is stationary. Purely diagnostic
        -- it anchors the continuation path, it does not select. Compared
        with ``lambda_``, it says how much signal the data carries: well
        above, and the selection is comfortable; below, and the calibrated
        weight already kills every variable.
    selected_ : list of int
        Indices selected by the penalized fit, frozen before any pruning.
    pruned_ : bool
        Whether ``model_`` is the unpenalized refit on the support.
    """

    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        bias: bool = True,
        lam: float | None = None,
        alpha: float = 0.05,
        n_mc: int = 1_000,
        lr: float = 0.01,
        adam_lr: float = 0.01,
        n_phases: int = 5,
        max_iter: int = 500,
        min_iter: int = 100,
        window: int = 50,
        tol: float = 1e-4,
        standardize: bool = True,
    ) -> None:
        super().__init__()

        if lam is not None and float(lam) < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam!r}.")

        n_phases = int(n_phases)
        if n_phases < 1:
            raise ValueError(f"n_phases must be at least 1, got {n_phases!r}.")

        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter!r}.")

        min_iter = int(min_iter)
        if min_iter < 0:
            raise ValueError(f"min_iter must be non-negative, got {min_iter!r}.")

        window = int(window)
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window!r}.")

        tol = float(tol)
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative, got {tol!r}.")

        self.hidden_dims = hidden_dims
        self.bias = bias
        self.lambda_ = lam
        self.alpha = float(alpha)
        self.n_mc = int(n_mc)
        self.lr = float(lr)
        self.adam_lr = float(adam_lr)
        self.n_phases = n_phases
        self.max_iter = max_iter
        # A floor above the ceiling would silence early stopping entirely.
        self.min_iter = min(min_iter, max_iter)
        self.window = window
        self.tol = tol
        self.lambda_data_: float | None = None
        self.standardize = bool(standardize)

        # Set by fit; the helpers below read it rather than take a flag.
        self._verbose = False

        self.loss = self._build_loss()
        self.pruned_ = False

        # Filled at fit time; buffers so they follow state_dict and .to().
        self.register_buffer("input_mean_", None)
        self.register_buffer("input_scale_", None)
        self.register_buffer("_support", None)

        # An unfitted estimator owns no parameter, so nothing would record
        # a .to(device) call. This empty buffer does: it is moved and cast
        # like any other, and fit reads the placement off it.
        self.register_buffer("_marker", torch.empty(0), persistent=False)

    # ------------------------------------------------------------------ #
    # Placement
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        """Device the estimator lives on. Follows ``.to()``."""
        return self._marker.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype the estimator computes in. Follows ``.to()``."""
        return self._marker.dtype

    def _as_tensor(self, value) -> Tensor:
        """Convert to the estimator's device and floating dtype."""
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _new_network(self, in_features: int) -> VariableSelectionMLP:
        """Build a network of the configured shape, on the right device."""
        return VariableSelectionMLP(
            in_features=in_features,
            hidden_dims=self.hidden_dims,
            bias=self.bias,
        ).to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_loss(self) -> Loss:
        """Return the data-fidelity loss defining the model."""

    def _check_target(self, y) -> Tensor:
        """Validate and convert the target. Subclasses tighten this."""
        y = self._as_tensor(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.ndim != 1:
            raise ValueError(
                f"y must be one-dimensional, got shape {tuple(y.shape)}."
            )
        return y

    def _calibrate(
        self, X: Tensor, y: Tensor, generator: "torch.Generator | None"
    ) -> float:
        """Calibrated regularisation parameter

        Returns
        -------
        lambda_ : float
            The calibrated threshold.
        """
        lambda_ = lambda_pdb(
            X, self.loss, alpha=self.alpha, n_mc=self.n_mc, generator=generator
        )
        return lambda_

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        y,
        generator: "torch.Generator | None" = None,
        prune: bool = False,
        verbose: bool = False,
    ):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like
            Target, in the format expected by the subclass.
        generator : torch.Generator, optional
            Randomness of the calibration, for reproducibility.
        prune : bool, default=False
            After the penalized fit, drop the zeroed first-layer columns
            and refit the reduced network on the selected variables only,
            without penalty and without gauge retraction. The support is
            frozen before the refit. Skipped when the selection is empty.
        verbose : bool, default=False
            Trace the fit: the calibrated weights, then one table per
            phase carrying the loss, the penalty, their sum, the relative
            move across the window that the tolerance is tested against,
            the gauge scale and the size of the support.

        Returns
        -------
        self
        """
        self._verbose = bool(verbose)
        X = self._as_tensor(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a matrix, got shape {tuple(X.shape)}.")

        y = self._check_target(y)
        if len(y) != len(X):
            raise ValueError(
                f"X and y carry different sample counts: {len(X)} and {len(y)}."
            )

        if self.standardize:
            mean = X.mean(dim=0)
            scale = X.std(dim=0, unbiased=False)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        else:
            mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            scale = torch.ones(X.shape[1], dtype=X.dtype, device=X.device)
        self.input_mean_ = mean
        self.input_scale_ = scale
        Xs = (X - mean) / scale

        model = self._new_network(X.shape[1])

        n_features = X.shape[1]

        if self.lambda_ is None:
            if self._verbose:
                print(
                    f"Calibrating lambda at level alpha = {self.alpha:g} "
                    f"({self.n_mc} replicates)..."
                )
            self.lambda_ = self._calibrate(Xs, y, generator)
            lambda_source = "calibrated"
        else:
            self.lambda_ = float(self.lambda_)
            lambda_source = "user-supplied"

        # The statistic the path is anchored on: the smallest weight for
        # which the null iterate is stationary. One number, closed form,
        # no sampling -- unlike the calibrated threshold above.
        self.lambda_data_ = float(self.loss.zero_thresholding(Xs, y).reshape(()))

        if self._verbose:
            print(
                f"lambda      {self.lambda_:.6f}  ({lambda_source})\n"
                f"lambda_data {self.lambda_data_:.6f}  "
                f"(the null is stationary above this)"
            )

        objective, total_iterations = self._optimize(model, Xs, y)

        self.model_ = model
        self.n_iter_ = total_iterations
        self._support = torch.where(
            model.penalized_weight.detach().norm(dim=0) > 0
        )[0]
        self.pruned_ = False

        if self._verbose:
            selected = self._support.tolist()
            print(
                f"\nSelection  {len(selected)}/{n_features} features "
                f"{selected}\n"
                f"           {total_iterations} iterations, "
                f"objective {objective:.6f}"
            )

        if prune and self._support.numel() > 0:
            self._refit_pruned(Xs, y)

        return self

    def _optimize(
        self,
        model: VariableSelectionMLP,
        Xs: Tensor,
        y: Tensor,
    ) -> tuple[float, int]:
        """Walk the continuation path with ProxAdam, in place.

        One optimizer drives the whole path: raising ``lam`` between
        phases warm-starts each from the previous solution, and keeps the
        moment estimates, which are what makes the descent cheap. Every
        iteration re-fixes the gauge and realigns those moments with it.
        Returns the objective of the last phase and the total number of
        iterations.

        A phase stops when it has run ``min_iter`` iterations and the
        penalized objective has moved by less than ``tol`` in relative
        terms across the last ``window`` of them. ``min_iter`` covers the
        opening iterations, where the second moments are still warming up
        and the anneal has not yet done real work; the window covers the
        fact that nothing here is monotone, so a single quiet step is not
        evidence of anything.
        """
        penalty = L1()
        w1, others = model.parameter_groups()

        # The penalty only sees W1, while the network is free to move
        # scale between the first two layers without changing its
        # function. Fixing the gauge before the first step, and after
        # every one of them, is what makes a weight on W1 mean anything.
        model.retract(gauge_scale(model, penalty, Xs))

        criterion = self.loss
        groups: list[dict] = [
            {"params": [w1], "penalty": penalty, "lam": 0.0}
        ]
        if others:
            groups.append({"params": others, "lam": 0.0})
        optimizer = ProxAdam(groups, lr=self.lr, eps=1e-4)

        # `fit` measures the statistic before calling; a plain geometric
        # ramp is the fallback if anything ever calls this without it.
        lambda_data = self.lambda_data_
        path = lambda_path(
            float(self.lambda_),
            0.0 if lambda_data is None else lambda_data,
            n_phases=self.n_phases,
        )
        n_phases = len(path)
        if self._verbose:
            print(
                f"Path        {n_phases} phase{'s' if n_phases > 1 else ''}, "
                f"warm-started, from lam = {path[0]:.6f}\n"
                f"Anneal      one cosine per phase, starting from "
                f"{self.lr:.2e} and shrinking by {_PHASE_LR_DECAY:g}\n"
                f"            each phase, down to {_ANNEAL_FLOOR:g} of that "
                f"within the phase\n"
                f"Stop        at least {self.min_iter} iterations, and the "
                f"objective moving by\n"
                f"            less than {self.tol:g} in relative terms over "
                f"{self.window} iterations"
            )

        objective = float("inf")
        total_iterations = 0
        for phase, lam in enumerate(path, start=1):
            optimizer.param_groups[0]["lam"] = lam

            # Each phase gets its own anneal, from a rate that decays along
            # the path down to a thousandth of it. The scheduler snapshots
            # the learning rate it finds, so the previous anneal has to be
            # rewound first.
            phase_lr = self.lr * _PHASE_LR_DECAY ** (phase - 1)
            for group in optimizer.param_groups:
                group["lr"] = phase_lr
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_iter,
                eta_min=phase_lr * _ANNEAL_FLOOR,  # type: ignore[arg-type]
            )

            if self._verbose:
                print(f"\nPhase {phase}/{n_phases}  lam = {lam:.6f}  "
                      f"lr = {phase_lr:.2e}")
                print(_trace_header())

            # Holds the objective of the last `window` iterations plus the
            # current one, so its head is what the tolerance compares
            # against once it is full.
            history: deque[float] = deque(maxlen=self.window + 1)
            collapsed = False
            n_iter = 0
            reason = f"ran out at the {self.max_iter}-iteration ceiling"
            for n_iter in range(1, self.max_iter + 1):
                optimizer.zero_grad()
                smooth = criterion(model(Xs), target=y)
                smooth.backward()
                with torch.no_grad():
                    smooth_value = float(smooth)
                    penalty_value = float(penalty.value(w1, lam))
                objective = smooth_value + penalty_value

                optimizer.step()

                # The step moved W1, so the gauge it was measured in is
                # stale: re-fix it and carry the moment estimates over,
                # or they spend the horizon of their exponential windows
                # describing a parametrisation the model has left.
                t = gauge_scale(model, penalty, Xs)
                model.retract(t)
                optimizer.transform_state(model.retraction_scales(t))
                scheduler.step()

                # Against the objective `window` iterations back, relative
                # to the objective itself and floored at one so that a
                # near-zero objective does not make the test unreachable.
                # Infinite until the window has filled: the descent is not
                # monotone, and a pair of neighbouring iterates can agree
                # to five digits halfway up a climb that the window sees
                # whole.
                history.append(objective)
                rel_err = (
                    abs(history[0] - objective) / max(1.0, abs(objective))
                    if len(history) > self.window
                    else float("inf")
                )

                selected = int(
                    torch.count_nonzero(torch.linalg.vector_norm(w1, dim=0))
                )

                collapsed = model.is_collapsed()
                converged = n_iter >= self.min_iter and rel_err < self.tol
                if collapsed:
                    reason = (
                        "the network collapsed to a constant; no column of "
                        "W1 survives, and the remaining phases only raise "
                        "the penalty weight further"
                    )
                elif converged:
                    reason = (
                        f"objective moved by {rel_err:.1e} in relative terms "
                        f"over {self.window} iterations"
                    )

                if self._verbose and (
                    (n_iter - 1) % _TRACE_EVERY == 0 or collapsed or converged
                ):
                    print(
                        _trace_row(
                            n_iter, smooth_value, penalty_value, rel_err,
                            float(t), selected,
                        )
                    )

                if collapsed or converged:
                    break

            total_iterations += n_iter

            # One extra forward, so the phase reports the iterate it
            # actually leaves behind rather than the one it stepped from.
            with torch.no_grad():
                objective = float(criterion(model(Xs), target=y)) + float(
                    penalty.value(w1, lam)
                )

            if self._verbose:
                print(f"    stopped at {n_iter}: {reason}")
                print(
                    f"    objective {objective:.6f} | "
                    f"selected {selected}/{w1.shape[1]}"
                )

            # A collapsed network cannot revive a variable at a larger
            # penalty weight either, so the remaining phases are moot.
            if collapsed:
                if self._verbose and phase < n_phases:
                    print(f"    skipping the remaining {n_phases - phase} phase(s)")
                break

        return objective, total_iterations

    def _optimize_unpenalized(
        self,
        model: VariableSelectionMLP,
        Xs: Tensor,
        y: Tensor,
        tol: float,
    ) -> tuple[float, int]:
        """Unpenalized full-batch training, in place.
        """
        # The default eps, not the fitting one: with no proximal step to
        # bound there is nothing to buy by blunting the preconditioner.
        optimizer = ProxAdam(
            [{"params": list(model.parameters()), "lam": 0.0}], lr=self.adam_lr
        )
        criterion = self.loss

        best = float("inf")
        flat_for = 0
        value = float("inf")
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            optimizer.zero_grad()
            objective = criterion(model(Xs), target=y)
            objective.backward()
            optimizer.step()
            value = float(objective.detach())

            gain = (best - value) / max(1.0, abs(value))
            flat_for = 0 if gain > tol else flat_for + 1
            best = min(best, value)

            if self._verbose and (n_iter - 1) % _TRACE_EVERY == 0:
                print(
                    f"    {n_iter:>4d}  loss {value:>12.6f}  "
                    f"gain {gain:>9.2e}  flat for {flat_for:>3d}"
                )

            if flat_for >= _ADAM_PATIENCE:
                if self._verbose:
                    print(
                        f"    stopped at {n_iter}: no improvement over "
                        f"{_ADAM_PATIENCE} iterations"
                    )
                break

        return value, n_iter

    def _refit_pruned(self, Xs: Tensor, y: Tensor) -> None:
        """Refit on the selected variables only, without penalty.

        The reduced network is trained on the smooth loss alone -- no
        penalty, no gauge retraction -- from a fresh initialisation.
        The support is frozen: :attr:`selected_` keeps the selection of
        the penalized fit.
        """
        support = self._support
        if self._verbose:
            print(
                f"\nPruning     unpenalized refit on {int(support.numel())} "
                "variable(s), fresh initialisation"
            )

        pruned = self._new_network(int(support.numel()))
        objective, n_iter = self._optimize_unpenalized(
            pruned, Xs[:, support], y, self.tol
        )

        if self._verbose:
            print(f"    {n_iter} iterations | loss {objective:.6f}")

        self.model_ = pruned
        self.pruned_ = True

    # ------------------------------------------------------------------ #
    # Fitted state
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet; "
                "call fit first."
            )

    @property
    def selected_(self) -> list[int]:
        """Selection of the penalized fit, frozen before any pruning."""
        self._check_fitted()
        return self._support.tolist()

    def _validate_inputs(self, x) -> Tensor:
        x = self._as_tensor(x)
        if x.ndim != 2 or x.shape[1] != self.input_mean_.numel():
            raise ValueError(
                f"X must have shape (n, {self.input_mean_.numel()}), "
                f"got {tuple(x.shape)}."
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Standardise ``x`` and evaluate the fitted network.

        An unpruned model takes the full feature matrix. A pruned model
        only knows the selected variables: pass the reduced matrix, as
        returned by :meth:`transform`. Differentiable, like any module
        forward; invoked as ``estimator(x)``.
        """
        self._check_fitted()
        if self.pruned_:
            x = self._as_tensor(x)
            n_selected = self._support.numel()
            if x.ndim != 2 or x.shape[1] != n_selected:
                raise ValueError(
                    f"This model was pruned: it expects the {n_selected} "
                    f"selected columns, e.g. transform(X); got shape "
                    f"{tuple(x.shape)}."
                )
            mean = self.input_mean_[self._support]
            scale = self.input_scale_[self._support]
        else:
            x = self._validate_inputs(x)
            mean, scale = self.input_mean_, self.input_scale_
        return self.model_((x - mean) / scale)

    def transform(self, X) -> Tensor:
        """Return ``X`` restricted to the selected columns.

        The columns are returned as given, without standardisation, in
        increasing index order; shape ``(n_samples, n_selected)``.
        """
        self._check_fitted()
        return self._validate_inputs(X)[:, self._support]

    def fit_transform(
        self,
        X,
        y,
        generator: "torch.Generator | None" = None,
        prune: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Fit the model, then return ``transform(X)``."""
        return self.fit(
            X, y, generator=generator, prune=prune, verbose=verbose
        ).transform(X)

    @torch.no_grad()
    def _decision(self, X) -> Tensor:
        """Forward pass without gradients, backing the predict methods."""
        return self(X).detach()

    def __repr__(self) -> str:
        parts = [f"hidden_dims={self.hidden_dims!r}"]
        if hasattr(self, "model_"):
            parts.append(f"lambda={self.lambda_:.4g}")
            parts.append(f"selected={len(self.selected_)}")
        return f"{type(self).__name__}({', '.join(parts)})"
