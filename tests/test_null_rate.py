r"""Monte-Carlo check of the calibration: under :math:`H_0`, the fitted network
selects nothing with probability close to :math:`1 - \alpha`.

The zero-thresholding condition makes :math:`\theta^0` a fixed point of the
fit exactly when :math:`\lambda_0(X, \mathbf y) \leq \lambda`, which happens
with probability :math:`1 - \alpha` at :math:`\lambda^{\mathrm{DB}}_\alpha`.
The fit may still stop elsewhere, so the observed rate is reported, not
tuned: the test only checks it against the Monte-Carlo error.
"""

from __future__ import annotations

import math

import pytest
import torch

from deeppic import BinaryClassifier, Regressor

ALPHA = 0.05
REPLICATES = 100


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["gaussian", "binary"])
def test_empty_selection_rate_under_the_null(kind):
    n, p, hidden = 100, 20, (16, 8)
    empty = 0
    for replicate in range(REPLICATES):
        generator = torch.Generator().manual_seed(10_000 + replicate)
        torch.manual_seed(replicate)
        X = torch.randn(n, p, generator=generator, dtype=torch.float64)
        if kind == "gaussian":
            model = Regressor(p, hidden, alpha=ALPHA)
            y = torch.randn(n, generator=generator, dtype=torch.float64)
        else:
            model = BinaryClassifier(p, hidden, alpha=ALPHA)
            y = torch.bernoulli(torch.full((n,), 0.5, dtype=torch.float64), generator=generator)
        model = model.double()
        model.fit(X, y, generator=generator)
        empty += not bool(model.selected.any())

    rate = empty / REPLICATES
    error = math.sqrt(ALPHA * (1.0 - ALPHA) / REPLICATES)
    print(f"\n{kind}: P(empty selection) = {rate:.3f} over {REPLICATES} draws, "
          f"target {1 - ALPHA:.3f} +- {error:.3f}")
    assert abs(rate - (1.0 - ALPHA)) <= 3.0 * error
