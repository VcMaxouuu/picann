"""Is the calibrated threshold the right one, for every network size?

Monte-Carlo study over ``n``, at fixed ``p``, whose primary axis is the
regularisation weight: once per replicate, :math:`\\lambda^{\\rm PDB}` is
calibrated on the standardised design -- exactly as the estimator would
internally -- and every architecture is fitted at the three weights
``LAM_FACTORS`` :math:`\\times\\ \\lambda`, 0.8, 1.0 and 1.2 by default.
Two quantities are recorded per architecture, per factor and per ``n``:
the probability of exact support recovery (``pesr``) and the mean squared
error on an independent test set, with its standard error.

The expected verdict on the weight: 0.8 fails by false positives -- the
calibration puts the null maximum's 95th percentile exactly at
:math:`\\lambda`, so four fifths of it lets noise columns through at any
``n`` -- while 1.2 fails by misses wherever the signal sits near the
threshold, and the calibrated 1.0 is the only one of the three dominated
nowhere. The threshold depends only on the design and the loss, not on
the network, so finding the same verdict for every architecture is the
point: :math:`\\lambda^{\\rm PDB}` is not a per-architecture tuning knob.

The architectures differ in *prediction*, not in selection::

    (32,)          one hidden layer
    (32, 16)       two hidden layers
    (32, 16, 8)    three hidden layers

The target
----------
Only four of the ``p`` variables carry signal, through the composed hinge

.. math::
    \\mu(x) = \\mathrm{relu}\\bigl(\\mathrm{relu}(x_0 + x_1)\\,
    \\mathrm{relu}(x_2 - x_3) - 1\\bigr),

standardised and scaled to the requested signal-to-noise ratio, with unit
variance Gaussian noise. The noise variance being one, a test MSE of one is
the Bayes floor and ``snr**2 + 1`` is what a constant predictor achieves.

Three composition levels -- linear forms, their product, a thresholded
hinge of it -- and a support inside the positive quadrant, so the target
is not an even function: every signal variable keeps a linear trace and
*every* architecture can select, which the weight analysis requires.
Measured behaviour (signal MSE at SNR 2.5): the one-hidden-layer model
recovers the support essentially always yet plateaus around 0.19 by
``n = 4000``, where the deeper models reach ~0.07; the deeper the model,
the more samples before depth pays -- both trail the shallow one at
``n = 500`` -- and two and three layers draw level by ``n = 4000``.
Depth buys prediction here, never selection.

Fairness
--------
Within a replicate the threshold is calibrated once and shared: every
architecture and every factor scales the very same :math:`\\lambda`, and
each architecture restarts from the same weights whatever the factor, so
a difference across factors is attributable to the weight alone and a
difference across architectures to the structure alone. The penalized fit
is followed by the unpenalized refit on the selected support (``PRUNE``):
comparing test errors only makes sense once the shrinkage bias, which
depends on the threshold rather than on the depth, has been removed.

Usage
-----
``python simulations/regression/depth_vs_sample_size.py``. The script
takes no argument; change the settings block below to change the study.

Replicates run in parallel over processes. The CSV is appended after every
``n``, so an interrupted run keeps its results, and re-running the script
skips the values of ``n`` already present in the file. Delete the file to
start over.
"""

from __future__ import annotations

import os

# Each replicate is single-threaded; the parallelism is across processes.
# Set before torch is imported, so the workers inherit it too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import math
import statistics
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import torch

# The project is not installed as a package; make it importable from here
# (two levels up: simulations/regression/ -> project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deeppic import SparseRegressor, SqrtMSELoss, lambda_pdb  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings. Edit them here; the script takes no argument.
# --------------------------------------------------------------------------- #
#: Sample sizes swept, in order: dense at the small sizes, where the
#: weight sweep discriminates, and reaching where the depth gap matures.
SAMPLE_SIZES = range(100, 3001, 100)

#: Architectures compared, keyed by the label used in the CSV: one, two
#: and three hidden layers. All select the same support; they differ in
#: how well they can represent the composed target.
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "32": (32,),
    "32-16": (32, 16),
    "32-16-8": (32, 16, 8),
}

#: Multiples of the calibrated threshold at which every architecture is
#: fitted. The threshold itself is computed once per replicate and shared.
LAM_FACTORS = (0.8, 1.0, 1.2)

#: Number of variables, of which N_SIGNAL carry signal.
P = 100
N_SIGNAL = 4

#: Replicates per sample size.
N_REPS = 100

#: Size of the independent test set, and signal-to-noise ratio of the
#: response. The noise has unit variance, so a test MSE of one is the Bayes
#: floor and ``SNR ** 2 + 1`` is what a constant predictor achieves.
N_TEST = 5000
SNR = 1.0

#: Level and Monte-Carlo budget of the threshold calibration.
ALPHA = 0.05
N_MC = 1000

#: Iteration cap of each penalized descent.
MAX_ITER = 500

#: Follow the penalized fit by the unpenalized refit on the selected
#: support. Kept on: comparing test errors across architectures only makes
#: sense once the shrinkage bias, which depends on the threshold rather
#: than on the depth, has been removed.
PRUNE = True

#: Base seed. The seed of a replicate is derived from it and from ``n``.
SEED = 0

#: Parallel processes. Set to 0 to run sequentially, for debugging.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

#: Results file, appended after every sample size.
OUTPUT = Path(__file__).with_name("depth_vs_sample_size.csv")

CSV_FIELDS = [
    "n",
    "p",
    "snr",
    "hidden_dims",
    "lam_factor",
    "n_reps",
    "n_failed",
    "pesr",
    "test_mse",
    "test_mse_se",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def target(x: torch.Tensor) -> torch.Tensor:
    """Composed signal on the four columns of ``x``: a thresholded hinge
    of a product of hinges, three composition levels deep."""
    return torch.relu(
        torch.relu(x[:, 0] + x[:, 1]) * torch.relu(x[:, 2] - x[:, 3]) - 1.0
    )


def make_dataset(n: int, seed: int):
    """Draw a train/test pair and the indices of the signal variables.

    ``mu`` is centred and scaled with the *training* statistics, so train and
    test responses live on the same scale and the test MSE is comparable
    across ``n``.
    """
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, P, generator=g)
    X_test = torch.randn(N_TEST, P, generator=g)
    true = torch.randperm(P, generator=g)[:N_SIGNAL]

    mu = target(X[:, true])
    mu_test = target(X_test[:, true])
    centre = mu.mean()
    spread = mu.std(unbiased=False).clamp_min(1e-12)
    mu = (mu - centre) / spread
    mu_test = (mu_test - centre) / spread

    signal_test = SNR * mu_test
    y = SNR * mu + torch.randn(n, generator=g)
    y_test = signal_test + torch.randn(N_TEST, generator=g)

    return X, y, X_test, y_test, signal_test, sorted(true.tolist())


# --------------------------------------------------------------------------- #
# One replicate
# --------------------------------------------------------------------------- #
def calibrate(X: torch.Tensor, seed: int) -> float:
    """The replicate's threshold, computed as the estimator would.

    The estimator standardises the design before calibrating and before
    penalising, so a threshold meant to be passed as ``lam`` must be
    measured on the standardised columns too.
    """
    mean = X.mean(dim=0)
    scale = X.std(dim=0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return float(
        lambda_pdb(
            (X - mean) / scale,
            SqrtMSELoss(),
            alpha=ALPHA,
            n_mc=N_MC,
            generator=torch.Generator().manual_seed(seed),
        )
    )


def run_replicate(task: tuple[int, int]) -> list[dict]:
    """Fit every architecture at every weight on one dataset.

    One row per (architecture, factor) pair. The threshold is calibrated
    once and every fit scales it; each architecture restarts from the same
    weights whatever the factor, so the factor is the only thing that
    changes between its fits.

    The workers are spawned, so they re-import this module and read the
    settings above directly; only ``n`` and the replicate index travel.

    Returns rows with ``failed=True`` rather than raising, so that a single
    pathological draw cannot take down a long run.
    """
    n, rep = task
    torch.set_num_threads(1)

    seed = SEED + 1_000_003 * rep + n
    X, y, X_test, y_test, _, true = make_dataset(n, seed)
    truth = set(true)
    lam_base = calibrate(X, seed + 7)

    rows = []
    for label, dims in ARCHITECTURES.items():
        for factor in LAM_FACTORS:
            row = {
                "n": n,
                "rep": rep,
                "hidden_dims": label,
                "lam_factor": factor,
                "failed": False,
            }
            try:
                # The weight seed is derived from the label rather than
                # from a position, so adding or removing an architecture
                # or a factor leaves the others bit-for-bit reproducible.
                # crc32, not hash(), which is salted per process and would
                # differ across workers.
                torch.manual_seed(seed * 31 + zlib.crc32(label.encode()))
                model = SparseRegressor(
                    hidden_dims=dims,
                    lam=factor * lam_base,
                    max_iter=MAX_ITER,
                )
                model.fit(X, y, prune=PRUNE)

                selected = set(model.selected_)
                # A pruned network only knows the columns it kept.
                inputs = model.transform(X_test) if model.pruned_ else X_test
                prediction = model.predict(inputs)

                row.update(
                    exact=float(selected == truth),
                    test_mse=float((prediction - y_test).square().mean()),
                )
            except Exception as error:  # noqa: BLE001
                row["failed"] = True
                row["error"] = f"{type(error).__name__}: {error}"

            rows.append(row)

    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean_se(values: list[float]) -> tuple[float, float]:
    """Mean and standard error, ``nan`` when the sample is empty."""
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def summarise(n: int, label: str, factor: float, rows: list[dict]) -> dict:
    """Collapse one (n, architecture, factor) cell into a CSV row."""
    done = [r for r in rows if not r["failed"]]

    test_mse, test_mse_se = _mean_se([r["test_mse"] for r in done])

    return {
        "n": n,
        "p": P,
        "snr": SNR,
        "hidden_dims": label,
        "lam_factor": factor,
        "n_reps": len(done),
        "n_failed": len(rows) - len(done),
        "pesr": (
            statistics.fmean(r["exact"] for r in done) if done else math.nan
        ),
        "test_mse": test_mse,
        "test_mse_se": test_mse_se,
    }


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def completed_sizes(path: Path, expected_rows: int) -> set[int]:
    """Values of ``n`` already fully recorded in ``path``."""
    if not path.exists():
        return set()

    counts: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                counts[int(row["n"])] = counts.get(int(row["n"]), 0) + 1
            except (KeyError, ValueError):
                continue
    return {n for n, count in counts.items() if count >= expected_rows}


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows, writing the header the first time. Flushed on exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    sizes = list(SAMPLE_SIZES)
    done = completed_sizes(OUTPUT, len(ARCHITECTURES) * len(LAM_FACTORS))
    todo = [n for n in sizes if n not in done]

    n_fits = len(todo) * N_REPS * len(ARCHITECTURES) * len(LAM_FACTORS)
    print(
        f"p = {P}, snr = {SNR}, {N_REPS} replicates, "
        f"{len(ARCHITECTURES)} architectures x "
        f"lam factors {list(LAM_FACTORS)}"
    )
    print(
        f"{len(sizes)} sample sizes from {sizes[0]} to {sizes[-1]} "
        f"-> {n_fits} fits to run"
    )
    if done:
        print(f"resuming: {len(done)} sample size(s) already in {OUTPUT}")
    print(f"writing to {OUTPUT}\n", flush=True)

    pool = None
    if N_WORKERS > 0:
        pool = ProcessPoolExecutor(
            max_workers=N_WORKERS, mp_context=get_context("spawn")
        )
    try:
        for n in todo:
            started = time.perf_counter()
            tasks = [(n, rep) for rep in range(N_REPS)]
            if pool is None:
                batches = [run_replicate(task) for task in tasks]
            else:
                batches = list(pool.map(run_replicate, tasks, chunksize=1))

            rows = [row for batch in batches for row in batch]
            summary = [
                summarise(
                    n, label, factor,
                    [
                        r for r in rows
                        if r["hidden_dims"] == label
                        and r["lam_factor"] == factor
                    ],
                )
                for label in ARCHITECTURES
                for factor in LAM_FACTORS
            ]
            append_rows(OUTPUT, summary)

            elapsed = time.perf_counter() - started
            print(f"n = {n:5d}  ({elapsed:6.1f}s)")
            for label in ARCHITECTURES:
                cells = [
                    row for row in summary if row["hidden_dims"] == label
                ]
                parts = " | ".join(
                    f"{row['lam_factor']:.1f}: pesr {row['pesr']:.2f} "
                    f"mse {row['test_mse']:7.3f}"
                    for row in cells
                )
                n_failed = sum(row["n_failed"] for row in cells)
                failed = f"  [{n_failed} failed]" if n_failed else ""
                print(f"    {label:>9s} | {parts}{failed}")
            print(flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    print(f"done -> {OUTPUT}")


if __name__ == "__main__":
    main()
