"""Continuation path for the regularisation parameter."""

from __future__ import annotations

__all__ = ["lambda_path"]


def _geomspace(start: float, stop: float, n: int) -> list[float]:
    """``n`` points from ``start`` to ``stop``, geometrically spaced.

    Both ends must be strictly positive. A single point is placed at
    ``stop``, not at ``start``: every segment of a continuation path is
    named by where it has to arrive.
    """
    if n <= 0:
        return []
    if n == 1:
        return [stop]
    ratio = (stop / start) ** (1.0 / (n - 1))
    # The last entry is written out rather than accumulated, so the path
    # ends exactly on `stop` whatever the rounding of the ratio.
    return [start * ratio**k for k in range(n - 1)] + [stop]


def lambda_path(
    lambda_target: float,
    lambda_data: float,
    n_phases: int = 5,
    floor: float = 0.2,
    margin: float = 0.1,
) -> list[float]:
    r"""Ascending geometric path of penalty weights, ending at
    ``lambda_target``.

    Anchored on ``lambda_data`` when the target sits above it, since that is
    where the null becomes a local minimum: for
    :math:`\lambda \ge \lambda_{\rm data}` the zero iterate satisfies the
    first-order condition of the penalized problem, so a descent started too
    close to zero, or landed straight on a large target, is free to stay
    there. Crossing that value with points on both sides -- the last below at
    :math:`(1 - \text{margin})\lambda_{\rm data}`, the first above at
    :math:`(1 + \text{margin})\lambda_{\rm data}` -- carries a fitted iterate
    across it instead, and the support that survives is the one the data
    actually supports.

    Parameters
    ----------
    lambda_target : float
        Weight of the final phase, the one producing the selection.
        Typically the calibrated :func:`~deeppic.calibration.lambda_pdb`.
    lambda_data : float
        Zero-thresholding statistic of the observed sample, i.e. the smallest
        weight for which the null iterate is stationary; see
        :meth:`~deeppic.optim.Loss.zero_thresholding`. Values that are not
        strictly positive, or that ``lambda_target`` does not clear by more
        than ``margin``, make the anchor moot -- there is then no crossing
        left to stage before the last phase -- and the path reduces to a
        plain geometric ramp up to ``lambda_target``.
    n_phases : int, default=5
        Number of phases, i.e. length of the returned path.
    floor : float, default=0.2
        First weight of the path, as a fraction of the value it ramps up
        towards. Small enough to leave the network free to fit, large enough
        that the first phase already shrinks something.
    margin : float, default=0.1
        Relative half-width of the gap left around ``lambda_data``. Landing
        exactly on it would put the descent at the very point where the null
        turns stationary.

    Returns
    -------
    list of float
        Non-decreasing, of length ``n_phases``, ending at ``lambda_target``.

    Raises
    ------
    ValueError
        If ``lambda_target`` is not strictly positive, if ``n_phases`` is
        below one, if ``floor`` is outside ``(0, 1]``, or if ``margin`` is
        outside ``[0, 1)``.

    Examples
    --------
    >>> lambda_path(0.30, 0.10)  # target above the data statistic
    [0.02, 0.042..., 0.09, 0.11, 0.3]
    >>> lambda_path(0.05, 0.10)  # target below it: plain ramp
    [0.01, 0.014..., 0.021..., 0.031..., 0.05]
    """
    lambda_target = float(lambda_target)
    lambda_data = float(lambda_data)

    if not lambda_target > 0.0:
        raise ValueError(
            f"lambda_target must be strictly positive, got {lambda_target!r}."
        )

    n_phases = int(n_phases)
    if n_phases < 1:
        raise ValueError(f"n_phases must be at least 1, got {n_phases!r}.")

    floor = float(floor)
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must lie in (0, 1], got {floor!r}.")

    margin = float(margin)
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must lie in [0, 1), got {margin!r}.")

    # A single phase has no room to ramp; and a target that does not clear
    # the upper margin has no crossing left to stage -- the last phase is
    # the crossing -- so a plain ramp does.
    if n_phases == 1:
        return [lambda_target]
    if lambda_data <= 0.0 or lambda_target <= (1.0 + margin) * lambda_data:
        return _geomspace(floor * lambda_target, lambda_target, n_phases)

    # Roughly three phases below the crossing for every two above: what the
    # path buys is a fitted iterate at the crossing, and that is bought on
    # the way up.
    n_below = max(1, min(round(0.6 * n_phases), n_phases - 1))
    below = _geomspace(
        floor * lambda_data, (1.0 - margin) * lambda_data, n_below
    )
    above = _geomspace(
        (1.0 + margin) * lambda_data, lambda_target, n_phases - n_below
    )
    return below + above
