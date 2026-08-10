from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = ["Penalty", "Zero", "L1", "GroupL1"]


def soft_thresholding(W: Tensor, step: Tensor | float) -> Tensor:
    r"""Elementwise :math:`\operatorname{sign}(W)\,(|W| - t)_+`, with **exact** zeros."""
    return torch.sign(W) * (W.abs() - step).clamp_min(0.0)


class Penalty(ABC):
    r"""Base class for sparsity-inducing penalties.

    .. math::  P_\lambda(W) = \sum_g \rho_\lambda\!\left(W^{(g)}\right)

    where the groups :math:`g` are single entries for an elementwise penalty
    and blocks of entries for a group penalty. Subclasses provide the profile
    :math:`\rho_\lambda` and the associated proximal operator.

    Subclasses satisfy :math:`\rho_\lambda'(0^+) = \lambda`, which is what makes
    the calibrated threshold :math:`\lambda` comparable across penalties.

    Notes
    -----
    A preconditioned proximal step solves

    .. math::
        \min_z \tfrac12 \|z - \hat z\|_D^2 + \alpha P_\lambda(z)

    for a positive diagonal :math:`D`. When the penalty is separable in every
    coordinate this splits into scalar problems and any :math:`D` admits a
    closed form. A group penalty does not split that way, and the closed form
    generally requires :math:`D` to be constant within each block; see
    :meth:`condition`.
    """

    def __str__(self) -> str:
        """Return a concise representation of the penalty."""
        params = self._param_str()
        name = self.__class__.__name__
        return f"{name}({params})" if params else name

    def __repr__(self) -> str:
        """Return the developer-facing representation of the penalty."""
        return self.__str__()

    def _param_str(self) -> str:
        """Return penalty-specific parameters for string formatting."""
        return ""

    @abstractmethod
    def rho(self, W: Tensor, lam: float) -> Tensor:
        r"""The profile :math:`\rho_\lambda`, applied to each group of ``W``."""

    def value(self, W: Tensor, lam: float) -> Tensor:
        r"""Return :math:`P_\lambda(W) = \sum_g \rho_\lambda(W^{(g)})` as a 0-dim tensor."""
        return self.rho(W, lam).sum()

    @abstractmethod
    def _prox(self, W: Tensor, lam: float, step: Tensor) -> Tensor:
        """Closed form, assuming ``lam >= 0`` and ``step > 0``."""

    def prox(self, W: Tensor, lam: float, step: float | Tensor) -> Tensor:
        r"""Proximal operator.

        Parameters
        ----------
        W : Tensor
            Point to be shrunk, any shape.
        lam : float
            The regularisation parameter :math:`\lambda`.
        step : float or Tensor
            The proximal step size, either a scalar or a tensor broadcastable
            to ``W``. A tensor step encodes a preconditioner; it must be
            admissible for this penalty, which :meth:`condition` guarantees.

        Returns
        -------
        Tensor
            Same shape as ``W``, with **exact** zeros (``== 0``) below the
            threshold.

        Raises
        ------
        ValueError
            If ``lam < 0``, or if any step size is non-positive.
        """
        lam = float(lam)
        if lam < 0.0:
            raise ValueError(f"{type(self).__name__} requires lam >= 0, got {lam}.")

        step = torch.as_tensor(step, dtype=W.dtype, device=W.device)
        if bool((step <= 0.0).any()):
            raise ValueError(
                f"{type(self).__name__} requires positive step sizes, got a minimum "
                f"of {step.min().item():.6g}."
            )

        return self._prox(W, lam, step)

    def condition(self, denom: Tensor) -> Tensor:
        r"""Coerce a preconditioner into a metric in which :meth:`_prox` is exact.

        Parameters
        ----------
        denom : Tensor
            The diagonal of :math:`C_t + \delta I`, shaped like the parameter.

        Returns
        -------
        Tensor
            A positive tensor broadcastable to ``denom``, to be used in place of
            it. The default returns ``denom`` untouched: an elementwise penalty
            is separable in every coordinate, so every positive diagonal metric
            admits the same closed form.
        """
        return denom

    def sensitivity_scale(self, a: Tensor) -> Tensor:
        r"""Gauge scale :math:`\max_i \|a_i\|_*`, the dual of the penalty's
        within-group norm over the hidden-unit axis.

        The default is :math:`\ell_\infty`, dual to the :math:`\ell_1` geometry
        that :class:`L1` has at the origin.
        """
        return a.abs().amax()


class Zero(Penalty):
    r"""Zero penalty :math:`P_\lambda(W) = 0`, for an unpenalized group."""

    def rho(self, W: Tensor, lam: float) -> Tensor:
        return torch.zeros((), dtype=W.dtype, device=W.device)

    def _prox(self, W: Tensor, lam: float, step: Tensor) -> Tensor:
        return W


class L1(Penalty):
    r"""Lasso penalty :math:`\rho_\lambda(t) = \lambda|t|`.

    Separable in every coordinate, so its proximal operator is elementwise
    soft-thresholding under any positive diagonal metric.
    """

    def rho(self, W: Tensor, lam: float) -> Tensor:
        return float(lam) * W.abs()

    def _prox(self, W: Tensor, lam: float, step: Tensor) -> Tensor:
        return soft_thresholding(W, step * lam)


class GroupL1(Penalty):
    r"""Group lasso penalty.

    .. math::
        P_\lambda(W) = \lambda \sum_j \bigl\| W^{(j)} \bigr\|_2

    Parameters
    ----------
    dim : int or tuple of int, default=0
        Axes reduced to form the blocks. The default groups the columns of
        :math:`W^1`, one block per input variable.

    Notes
    -----
    The closed form below is the proximal operator in a metric that is constant
    within each block, where the objective reduces to a scalar multiple of the
    Euclidean one and the solution is block soft-thresholding. Under a general
    diagonal metric the group proximal mapping has no closed form and calls for
    a root search (Yun et al., Lemma 1). :meth:`condition` sidesteps this by
    averaging the preconditioner over each block, which is itself a valid
    positive diagonal preconditioner and so leaves the convergence guarantee
    intact, at the cost of adaptivity *within* a block. Adaptivity *between*
    blocks, the part that drives selection, is preserved.
    """

    def __init__(self, dim: int | tuple[int, ...] = 0) -> None:
        if isinstance(dim, int):
            axes: tuple[int, ...] = (dim,)
        elif isinstance(dim, tuple) and all(isinstance(d, int) for d in dim):
            axes = dim
        else:
            raise ValueError(f"dim must be an int or a tuple of ints, got {dim!r}.")
        if not axes or len(set(axes)) != len(axes):
            raise ValueError(
                f"dim must be non-empty and free of duplicates, got {dim!r}."
            )
        self.dim = dim
        self._axes = axes

    def _param_str(self) -> str:
        return f"dim={self.dim}"

    def _block_shape(self, W: Tensor) -> tuple[int, ...]:
        """Shape of a per-block quantity, with the reduced axes kept as 1."""
        return tuple(1 if d in self._axes else s for d, s in enumerate(W.shape))

    def norms(self, W: Tensor, keepdim: bool = False) -> Tensor:
        r"""Block norms :math:`\|W^{(j)}\|_2`, with the axes ``dim`` reduced."""
        return torch.linalg.vector_norm(W, dim=self.dim, keepdim=keepdim)

    def rho(self, W: Tensor, lam: float) -> Tensor:
        return float(lam) * self.norms(W)

    def condition(self, denom: Tensor) -> Tensor:
        """Average the preconditioner over each block, keeping the reduced axes."""
        return denom.mean(dim=self.dim, keepdim=True)

    def _prox(self, W: Tensor, lam: float, step: Tensor) -> Tensor:
        step = self._as_block_step(W, step)
        r = self.norms(W, keepdim=True)
        scale = soft_thresholding(r, step * lam) / r.clamp_min(
            torch.finfo(W.dtype).tiny
        )
        return W * scale

    def _as_block_step(self, W: Tensor, step: Tensor) -> Tensor:
        """Validate the step and reduce it to one value per block.

        A step that varies within a block would threshold each entry against
        the *same* block norm with a *different* cutoff, leaving stray survivors
        in blocks that should die outright. That is not the proximal operator of
        anything, so it is rejected rather than silently broadcast.
        """
        if not step.ndim:
            return step

        block = self._block_shape(W)
        if step.shape == block:
            return step
        if step.shape != W.shape:
            raise ValueError(
                f"{type(self).__name__} expects a step size of shape {block} or "
                f"{tuple(W.shape)}, got {tuple(step.shape)}."
            )

        collapsed = step.amin(dim=self.dim, keepdim=True)
        if not torch.allclose(step, collapsed.expand_as(step)):
            raise ValueError(
                f"{type(self).__name__} requires a step size that is constant "
                "within each block; route the preconditioner through "
                "Penalty.condition first."
            )
        return collapsed

    def sensitivity_scale(self, a: Tensor) -> Tensor:
        if self.dim not in (0, (0,)):
            raise NotImplementedError(
                "sensitivity_scale assumes groups are the columns of W1, "
                f"i.e. dim=0; got dim={self.dim!r}."
            )
        return torch.linalg.vector_norm(a, dim=-1).amax()
