r"""DeepPINK (Lu et al., 2018) with model-X Gaussian knockoffs, through knockpy.

knockpy 1.3.5 implements the DeepPINK network (``knockpy.kpytorch.deeppink``) but
three of its code paths fail with the settings used here, and are patched below:

1. ``DeepPinkModel.feature_importances`` squeezes the product of the weight
   matrices inside its loop, so any network with two hidden layers or more
   raises ``ValueError``;
2. ``train_deeppink`` hands ``y.unsqueeze(-1)`` to ``CrossEntropyLoss``, so a
   binary response raises ``RuntimeError``;
3. ``DeepPinkStatistic.fit`` always scores the model in-sample with a squared
   error, which breaks on the two logits of a binary response; the score is not
   used by the filter.

Also, ``DeepPinkStatistic.fit`` does not infer the response type: ``y_dist``
must be passed. The filter uses knockoff+ (``offset=1``), which cannot select
fewer than about ``1 / fdr`` variables.

::

    from deeppink import deeppink_select
    selected = deeppink_select(X, y, hidden=(16, 16), fdr=0.1, Sigma=Sigma)
"""

from __future__ import annotations

import numpy as np
import torch
from knockpy import knockoff_stats
from knockpy.knockoff_filter import KnockoffFilter
from knockpy.kpytorch import deeppink
from torch import nn


def _feature_importances(self, weight_scores: bool = True) -> np.ndarray:
    r"""Lu et al.: :math:`Z_j = z_j w_j`, with :math:`w = W^{(0)} W^{(1)} \cdots
    W^{(L)}` the product of the weight matrices of the MLP, activations ignored.
    For a binary response, the two output logits enter through their
    difference."""
    with torch.no_grad():
        if weight_scores:
            linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
            w = linears[0].weight.T
            for layer in linears[1:]:
                w = w @ layer.weight.T
            w = (w[:, 0] if w.shape[1] == 1 else w[:, 1] - w[:, 0]).numpy()
        else:
            w = np.ones(self.p)
        z = self._fetch_Z_weight().numpy()
        return np.concatenate([z[self.feature_inds] * w, z[self.ko_inds] * w])


_train_gaussian = deeppink.train_deeppink


def _train_deeppink(model, features, y, **kwargs):
    """Train as knockpy does, with an integer target for a binary response."""
    if model.y_dist == "gaussian":
        return _train_gaussian(model, features, y, **kwargs)
    n, p = features.shape[0], features.shape[1] // 2
    lambda1 = kwargs.get("lambda1") or 10 * np.sqrt(np.log(p) / n)
    batchsize = min(n, kwargs.get("batchsize", 100))
    features = torch.tensor(features).float()
    target = torch.tensor(y).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for _ in range(kwargs.get("num_epochs", 50)):
        for Xb, yb in deeppink.create_batches(features, target, batchsize):
            loss = criterion(model(Xb), yb) + lambda1 * model.l1norm()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


deeppink.DeepPinkModel.feature_importances = _feature_importances
deeppink.train_deeppink = _train_deeppink
knockoff_stats.DeepPinkStatistic.cv_score_model = lambda self, features, y, cv_score: None


def deeppink_select(
    X: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...] = (16, 16),
    fdr: float = 0.1,
    Sigma: np.ndarray | None = None,
    binary: bool = False,
    num_epochs: int = 200,
    batchsize: int | None = None,
    normalize_Z: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    r"""Select variables with DeepPINK at false discovery rate ``fdr``.

    :param X: design matrix, of shape ``(n, p)``.
    :param y: response, of shape ``(n,)``; ``0/1`` when ``binary``.
    :param hidden: widths of the hidden layers of the MLP after the pairwise
        coupling layer.
    :param fdr: target false discovery rate of the knockoff+ filter.
    :param Sigma: covariance of the rows of ``X``; ``None`` estimates it
        (Ledoit-Wolf), as for real data.
    :param binary: whether ``y`` is binary (cross-entropy on two logits).
    :param num_epochs: training epochs.
    :param batchsize: mini-batch size; ``None`` trains full batch.
    :param normalize_Z: knockpy option normalizing each feature/knockoff pair
        of coupling weights to unit :math:`\ell_1` norm; ``False`` keeps the
        free coupling weights of Lu et al.
    :param seed: seed of NumPy (knockoffs) and PyTorch (network).
    :return: boolean mask of the selected variables, of shape ``(p,)``.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    kfilter = KnockoffFilter(ksampler="gaussian", fstat="deeppink")
    selected = kfilter.forward(
        X=X,
        y=y,
        Sigma=Sigma,
        fdr=fdr,
        fstat_kwargs={
            "hidden_sizes": list(hidden),
            "y_dist": "binomial" if binary else "gaussian",
            "normalize_Z": normalize_Z,
            "train_kwargs": {
                "num_epochs": num_epochs,
                "batchsize": X.shape[0] if batchsize is None else batchsize,
                "verbose": False,
            },
        },
    )
    return np.asarray(selected, dtype=bool)
