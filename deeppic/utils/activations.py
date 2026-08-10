"""Activation functions."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LeakyELU"]


class LeakyELU(nn.Module):
    r"""The leaky ReLU with its corner rounded: a convex combination of
    the identity and the ELU,

    .. math::
        \sigma(x) = m\,x + (1 - m)\,\mathrm{ELU}(x),

    i.e. the identity for :math:`x \ge 0` and
    :math:`m\,x + (1 - m)(e^{x} - 1)` below, which interpolates between
    slope :math:`1` at the origin and slope :math:`m` far in the negative
    range. Same asymptotes as ``LeakyReLU(m)``, same zero at zero, same
    active regime -- only the derivative changes character, and that is
    the point.

    It exists for the hidden layers *above* the first one, whose
    activation pattern enters the gauge sensitivity
    :math:`a = \partial f / \partial h_1` as one factor
    :math:`\sigma'(h_\ell)` per layer. With a piecewise-linear activation
    that factor is two-valued, :math:`\{m, 1\}`, and near the null the
    pre-activations of a deep layer are almost constant across the
    sample, so the whole layer crosses zero together: the measured gauge
    scale then jumps by up to :math:`1/m` in a single optimizer step, and
    the retraction that chases the jump destabilises the very pattern
    that caused it -- the reciprocal oscillation that wrecks the
    selection. Three properties of this activation close that route:

    * **The derivative is continuous** (:math:`C^1` map: the left limit
      :math:`m + (1 - m)e^{0} = 1` matches the right one). A layer
      drifting across zero moves the sensitivity *continuously*, so the
      gauge target cannot jump between two iterations; the per-iteration
      retraction tracks it exactly and the constraint
      :math:`\|a\|_* = 1` is enforceable at every step.
    * **The derivative is floored**,
      :math:`\sigma'(x) = m + (1 - m)e^{x} > m` everywhere. The ratio
      between the pattern factors of any two states is bounded by
      :math:`1/m` per layer -- the gauge cannot collapse further, however
      far apart the states -- and a unit never stops passing gradient, so
      no unit is unrecoverable. The plain ELU loses exactly this:
      :math:`\sigma' = e^{x}` vanishes in the tail, and a unit parked far
      negative re-enters with an unbounded factor.
    * **The log-derivative is 1-Lipschitz**,
      :math:`0 \le (\log \sigma')'(x) < 1`, so one optimizer step moving
      the pre-activations by :math:`\Delta h` moves each factor by at
      most :math:`e^{|\Delta h|}`. That is the quantitative form of "the
      retraction only ever has to correct :math:`1 + O(\alpha_t)` per
      iteration".

    Parameters
    ----------
    m : float, default=0.1
        Weight of the identity part, in ``[0, 1)``: the infimum of the
        derivative, hence the :math:`1/m` cap on the pattern factors.
        The stability argument rests on ``m > 0``; ``m = 0`` degenerates
        to the plain ELU and forfeits the floor. Matching the first
        layer's leak slope keeps a single constant for the whole network.
    """

    def __init__(self, m: float = 0.1) -> None:
        super().__init__()
        m = float(m)
        if not 0.0 <= m < 1.0:
            raise ValueError(f"m must lie in [0, 1); received {m!r}.")
        self.m = m

    def forward(self, x: Tensor) -> Tensor:
        return self.m * x + (1.0 - self.m) * F.elu(x)

    def extra_repr(self) -> str:
        return f"m={self.m}"
