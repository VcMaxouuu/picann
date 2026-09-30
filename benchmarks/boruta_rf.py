r"""Boruta with a random forest (Kursa and Rudnicki, 2010), through BorutaPy.

Every iteration adds a shuffled copy ("shadow") of each variable, fits the forest
and records which variables beat the best shadow in importance. A variable is
confirmed once it has done so significantly more often than chance, rejected
once significantly less, and left tentative otherwise; the tests are corrected
for multiplicity at level ``alpha``. Only the confirmed variables are selected.

::

    from boruta_rf import boruta_select
    selected = boruta_select(X, y, alpha=0.05, seed=0)
"""

from __future__ import annotations

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def boruta_select(
    X: np.ndarray,
    y: np.ndarray,
    binary: bool = False,
    alpha: float = 0.05,
    max_iter: int = 100,
    max_depth: int | None = 5,
    include_tentative: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    r"""Select variables with Boruta on a random forest.

    :param X: design matrix, of shape ``(n, p)``.
    :param y: response, of shape ``(n,)``; ``0/1`` when ``binary``.
    :param binary: whether ``y`` is binary (classification forest, balanced
        class weights).
    :param alpha: level of the multiplicity-corrected tests of Boruta.
    :param max_iter: maximum number of Boruta iterations; the variables still
        undecided then are tentative.
    :param max_depth: depth of the trees, 3 to 7 being the range BorutaPy
        recommends.
    :param include_tentative: whether the tentative variables count as
        selected.
    :param seed: seed of the forest and of the shadow permutations.
    :return: boolean mask of the selected variables, of shape ``(p,)``.
    """
    if binary:
        forest = RandomForestClassifier(
            max_depth=max_depth, class_weight="balanced", n_jobs=1
        )
    else:
        forest = RandomForestRegressor(max_depth=max_depth, n_jobs=1)
    selector = BorutaPy(
        forest,
        n_estimators="auto",
        alpha=alpha,
        max_iter=max_iter,
        random_state=seed,
    )
    selector.fit(np.asarray(X), np.asarray(y).ravel())
    selected = np.asarray(selector.support_, dtype=bool)
    if include_tentative:
        selected = selected | np.asarray(selector.support_weak_, dtype=bool)
    return selected
