r"""Variable-selection models for a regression.

The response distribution is picked by name rather than by handing over a loss:
a family fixes the loss, and with it the link the linear output goes through and
the transformation its value gets. Adding one is adding an entry to
:attr:`Regressor.families`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from deeppic.loss.loss import Loss, SqrtMSELoss
from deeppic.models.base import SelectionMLP

__all__ = ["Regressor"]


class Regressor(SelectionMLP):
    r"""Variable-selection model for a regression.

    ``"gaussian"`` fits the square root of the mean squared error through an
    identity link, the square-root Lasso objective, whose calibrated level does
    not depend on the noise scale. A family maps to a loss and to nothing else,
    so the network, the penalty and the phased fit are the ones of
    :class:`~deeppic.models.base.SelectionMLP`.

    :param input_dim: number of input variables :math:`p`.
    :param hidden_dims: width of each hidden layer, from the first to the last;
        ``None`` or empty reduces the network to a linear model.
    :param family: distribution of the response, a key of :attr:`families`.
    :param kwargs: forwarded to :class:`~deeppic.models.base.SelectionMLP`,
        which documents the level, the penalty and the phases.
    :raises ValueError: if ``family`` is not a key of :attr:`families`.
    """

    families: ClassVar[dict[str, Callable[[], Loss]]] = {
        "gaussian": SqrtMSELoss,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        family: str = "gaussian",
        **kwargs: Any,
    ) -> None:
        if family not in self.families:
            known = ", ".join(repr(name) for name in sorted(self.families))
            raise ValueError(f"family must be one of {known}, got {family!r}")

        super().__init__(input_dim, hidden_dims, self.families[family](), **kwargs)
        self.family = family
