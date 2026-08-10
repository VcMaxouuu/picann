import torch
import torch.nn as nn
from torch import Tensor
from ..optim import Penalty

__all__ = ["gauge_scale"]


def gauge_scale(model: nn.Module, penalty: Penalty, x: Tensor) -> Tensor:
    r"""Return the gauge scale :math:`t = \max_i \|a_i\|_*`.

    Parameters
    ----------
    model : nn.Module
        Network exposing ``hidden_dims`` and ``_sensitivity``.
    penalty : Penalty
        Supplies the dual norm through ``sensitivity_scale``.
    x : Tensor
        Batch at which the sensitivity is evaluated. Activation patterns are
        data dependent, so ``t`` is too; use the fitting sample.

    Returns
    -------
    Tensor
        Scalar, strictly positive. Exactly ``1`` for a linear model, which has
        no gauge freedom.

    Raises
    ------
    RuntimeError
        If the scale is not finite, which means the network itself diverged.

    Notes
    -----
    The sensitivity :math:`a = \partial f / \partial h_1` runs through the
    whole network, so :math:`t` reports what the output still owes the first
    hidden layer. Every retraction normalises it to ``1``; it drifts from
    there by a factor that measures how well conditioned the gauge is.

    It cannot reach zero. The sensitivity is a product of the derivative of
    every activation above the first layer, and
    :class:`~deeppic.models.VariableSelectionMLP` uses a leaky activation
    throughout precisely so that no factor in that product can vanish. A hard
    ReLU could: a hidden layer whose units are all off on the sample sends
    :math:`a` to zero exactly, leaving nothing for the retraction to
    normalise and no gradient reaching :math:`W_1`. That is why the choice of
    activation is not configurable.
    """
    if not model.hidden_dims:
        return torch.ones((), device=x.device, dtype=x.dtype)

    t = penalty.sensitivity_scale(model._sensitivity(x)).detach()

    if not torch.isfinite(t):
        raise RuntimeError("The gauge scale is not finite.")

    return t
