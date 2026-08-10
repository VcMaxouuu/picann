from __future__ import annotations

import math
from typing import Optional, Tuple, Any

import torch
from torch import Tensor, nn

from ..optim import L1

__all__ = ["VariableSelectionMLP"]


class VariableSelectionMLP(nn.Module):
    r"""MLP carrying a penalty on its first weight matrix.

    Parameters
    ----------
    in_features : int
        Number of input variables.

    hidden_dims : tuple of int or None, default=None
        Dimensions of the hidden layers. If ``None`` or empty, the model is
        linear.

    bias : bool, default=True
        Whether the last linear layer include a bias.

    Attributes
    ----------
    input_mean, input_scale : Tensor
        Buffers used to standardize inputs before the forward pass.

    Notes
    -----
    The activation is :class:`~torch.nn.LeakyReLU` after every hidden layer
    and is not configurable, for two reasons that pull the same way.

    It is positively homogeneous, which the gauge retraction requires: it
    scales the first two linear layers against each other and needs the
    module between them to satisfy :math:`\sigma(cx) = c\,\sigma(x)` for
    :math:`c > 0`, or the retraction would change the function instead of
    reparametrising it.

    And its derivative never vanishes, which keeps the network recoverable.
    The sensitivity the retraction normalises,
    :math:`a = \partial f / \partial h_1`, is a product of the derivatives
    of every activation above the first layer; a single one of them going
    to zero everywhere sends :math:`a` to zero exactly, and with it the
    gradient reaching :math:`W_1`. A hard ReLU does that whenever a hidden
    layer's units are all off on the sample -- an absorbing state, since a
    unit with no gradient cannot be switched back on -- and a saturating
    activation such as tanh approaches it. The leak makes that product
    incapable of vanishing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("`in_features` must be a positive integer.")

        hidden_dims = tuple(hidden_dims or ())

        if any(not isinstance(dim, int) or dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers.")

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.bias = bias

        self.network = self._build_network()

        self.register_buffer("input_mean", torch.zeros(in_features))
        self.register_buffer("input_scale", torch.ones(in_features))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the network."""
        return self.network(x).squeeze(-1)

    @property
    def penalized_weight(self) -> nn.Parameter:
        """Weight matrix of the first linear layer."""
        first_layer = self.network[0]

        if not isinstance(first_layer, nn.Linear):
            raise RuntimeError("The first network module is not a linear layer.")

        return first_layer.weight

    def _build_network(self) -> nn.Sequential:
        """Construct the sequence of linear and activation layers."""
        dimensions = (self.in_features, *self.hidden_dims, 1)
        modules: list[nn.Module] = []

        for layer_index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            # `bias` only governs the output layer; hidden layers always
            # carry one.
            is_output_layer = layer_index == len(dimensions) - 2
            modules.append(
                nn.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                    bias=self.bias if is_output_layer else True,
                )
            )

            # No activation after the output layer; LeakyReLU after every
            # other one. See the class notes: the retraction needs positive
            # homogeneity, and the sensitivity needs a derivative that
            # cannot vanish.
            if not is_output_layer:
                modules.append(nn.LeakyReLU())

        return nn.Sequential(*modules)

    def _set_input_scaling(
        self,
        mean: Tensor,
        scale: Tensor,
    ) -> None:
        """Set the input centering and scaling buffers."""
        mean = torch.as_tensor(
            mean,
            device=self.input_mean.device,
            dtype=self.input_mean.dtype,
        )
        scale = torch.as_tensor(
            scale,
            device=self.input_scale.device,
            dtype=self.input_scale.dtype,
        )

        expected_shape = (self.in_features,)

        if mean.shape != expected_shape:
            raise ValueError(
                f"mean must have shape {expected_shape}, got {tuple(mean.shape)}."
            )

        if scale.shape != expected_shape:
            raise ValueError(
                f"scale must have shape {expected_shape}, got {tuple(scale.shape)}."
            )

        if torch.any(scale <= 0):
            raise ValueError("All input scales must be strictly positive.")

        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale)

    def standardize_input(self, x: Tensor) -> Tensor:
        """Apply the stored input centering and scaling."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"The last input dimension must be {self.in_features}, "
                f"got {x.shape[-1]}."
            )
        return (x - self.input_mean) / self.input_scale


    def parameter_groups(self) -> tuple[nn.Parameter, list[nn.Parameter]]:
        """Return the penalized weight matrix and all remaining parameters.

        Returns
        -------
        W1 : nn.Parameter
            Weight matrix of the first linear layer.
        others : list of nn.Parameter
            All remaining trainable parameters.
        """
        W1 = self.penalized_weight
        others = [
            p
            for p in self.parameters()
            if p.requires_grad and p is not W1
        ]
        return W1, others


    def _sensitivity(self, x: Tensor) -> Tensor:
        r"""Compute sensitivity vector :math:`a = \partial f / \partial h_1`,
        of shape ``(n, H1)``."""
        if not self.hidden_dims:
            return torch.ones(
                (),
                device=x.device,
                dtype=x.dtype,
            )

        first_layer = self.network[0]

        if not isinstance(first_layer, nn.Linear):
            raise RuntimeError("The first module must be nn.Linear.")

        with torch.enable_grad():
            h1 = first_layer(x).detach().requires_grad_(True)
            output = self.network[1:](h1)

            (a,) = torch.autograd.grad(
                outputs=output.sum(),
                inputs=h1,
                create_graph=False,
                retain_graph=False,
            )

        return a.detach()


    def is_collapsed(self) -> bool:
        r"""Whether the network computes a constant function.

        True when every entry of :math:`W_1` is zero. The first
        pre-activation is then :math:`h_1 = b_1`, the same for every
        observation, and a leaky activation passes that constant on
        unchanged whatever the sign of the biases, so nothing downstream
        can make the output depend on :math:`x` again. Always ``False``
        for a linear model, whose :math:`W_1` being zero *is* the null fit
        rather than a degenerate parametrisation of it.

        Not an absorbing state, unlike its ReLU counterpart. A hard ReLU
        with negative biases would leave :math:`W_1` with no gradient at
        all; the leak keeps :math:`\partial L / \partial W_1` nonzero, so
        the descent can put a variable back. What makes the fit stop here
        anyway is the continuation path, which only ever raises the
        penalty weight: a support the proximal step has emptied at one
        weight will not survive the next.
        """
        if not self.hidden_dims:
            return False

        return not bool(self.penalized_weight.detach().any())

    def retract(self, t: float | Tensor):
        r"""Apply a gauge retraction. A linear model has no gauge freedom,
        so the call is a no-op.

        Raises
        ------
        ValueError
            If ``t`` is not finite and strictly positive. The second layer is
            divided by it, so a null or negative scale does not reparametrise
            the network, it destroys it.
        """
        if not self.hidden_dims:
            return t

        scale = float(t)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"The retraction scale must be finite and strictly positive, "
                f"got {scale!r}."
            )

        first_layer = self.network[0]
        second_layer = self.network[2]

        if not isinstance(first_layer, nn.Linear):
            raise RuntimeError("The first module must be nn.Linear.")

        if not isinstance(second_layer, nn.Linear):
            raise RuntimeError("The third module must be nn.Linear.")

        with torch.no_grad():
            first_layer.weight.mul_(t)

            if first_layer.bias is not None:
                first_layer.bias.mul_(t)

            second_layer.weight.div_(t)

        return t

    def retraction_scales(self, t: float | Tensor) -> dict:
        r"""Per-parameter scales matching :meth:`retract`, in the format
        expected by ``ProxAdam.transform_state``.

        Empty for a linear model, whose retraction is the identity.
        """
        if not self.hidden_dims:
            return {}

        first_layer = self.network[0]
        second_layer = self.network[2]

        scales: dict = {first_layer.weight: t}
        if first_layer.bias is not None:
            scales[first_layer.bias] = t
        scales[second_layer.weight] = 1 / t
        return scales
