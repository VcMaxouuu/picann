"""Recovery probability on the additive benchmark of Hardle et al. (2004).

Monte-Carlo study at fixed ``n`` and ``p``: how often does each of three
architectures recover exactly the four signal variables of

.. math::
    Y_i = f_1(x_{i1}) + f_2(x_{i2}) + f_3(x_{i3}) + f_4(x_{i4})
          + \\epsilon_i,

with

.. math::
    f_1(x) = -2\\sin(2x), \\qquad
    f_2(x) = x^2 - \\tfrac{1}{3}, \\qquad
    f_3(x) = x - \\tfrac{1}{2}, \\qquad
    f_4(x) = e^{-x} + e^{-1} - 1.

The covariates are uniform on ``(0, 1)`` -- the constants above are exactly
their means of ``x^2``, ``x`` and ``e^{-x}`` there, so every component is
centred -- and the noise is Gaussian with standard deviation ``SIGMA``. The
four signal variables sit at random positions among the ``p`` columns; the
remaining ones are pure noise.

One number is the deliverable -- the probability of exact support recovery
per architecture -- so the fits stop at the penalized phase: the refit
never changes the selection, only polishes the fit. Within a replicate the
three architectures see the same data and the same calibration draw, hence
exactly the same threshold; differences are attributable to the
architecture alone.

Usage
-----
``python simulations/hardle.py``. The script takes no argument; change the
settings block below to change the study. Results go to ``OUTPUT`` as one
CSV row per architecture.
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

# The project is not installed as a package; make it importable from here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeppic import SparseRegressor  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings. Edit them here; the script takes no argument.
# --------------------------------------------------------------------------- #
#: Sample size and number of variables, of which N_SIGNAL carry signal.
N = 150
P = 50
N_SIGNAL = 4

#: The three structures compared, keyed by the label used in the CSV.
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "linear": (),
    "32": (32,),
    "64-32": (64, 32),
}

#: Replicates of the Monte-Carlo study.
N_REPS = 100

SIGMA = 1.0

#: Level and Monte-Carlo budget of the threshold calibration.
ALPHA = 0.05
N_MC = 1000

#: Iteration cap of each penalized phase.
MAX_ITER = 500

#: Base seed. The seed of a replicate is derived from it.
SEED = 0

#: Parallel processes. Set to 0 to run sequentially, for debugging.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

#: Results file, one row per architecture.
OUTPUT = Path(__file__).with_name("hardle.csv")

CSV_FIELDS = [
    "model",
    "n",
    "p",
    "sigma",
    "n_reps",
    "n_failed",
    "exact_recovery",
    "exact_recovery_se",
    "tpr",
    "mean_false_positives",
    "mean_n_selected",
    "lambda_mean",
    "seconds_per_fit",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def components(x: torch.Tensor) -> torch.Tensor:
    """The four additive components, evaluated columnwise on ``(n, 4)``."""
    return torch.stack(
        [
            -2.0 * torch.sin(2.0 * x[:, 0]),
            x[:, 1] ** 2 - 1.0 / 3.0,
            x[:, 2] - 0.5,
            torch.exp(-x[:, 3]) + math.exp(-1.0) - 1.0,
        ],
        dim=1,
    )


def make_dataset(seed: int):
    """One draw of the design, the response and the signal positions."""
    g = torch.Generator().manual_seed(seed)

    X = torch.rand(N, P, generator=g)
    true = torch.randperm(P, generator=g)[:N_SIGNAL]
    y = components(X[:, true]).sum(dim=1) + SIGMA * torch.randn(N, generator=g)

    return X, y, sorted(true.tolist())


# --------------------------------------------------------------------------- #
# One replicate
# --------------------------------------------------------------------------- #
def run_replicate(rep: int) -> list[dict]:
    """Fit every architecture on one dataset; one row per architecture.

    The workers are spawned, so they re-import this module and read the
    settings above directly; only the replicate index travels.

    Returns a row with ``failed=True`` rather than raising, so that a single
    pathological draw cannot take down a long run.
    """
    torch.set_num_threads(1)

    seed = SEED + 1_000_003 * rep
    X, y, true = make_dataset(seed)
    truth = set(true)

    rows = []
    for label, dims in ARCHITECTURES.items():
        started = time.perf_counter()
        row = {"rep": rep, "model": label, "failed": False}
        try:
            # The weight seed is derived from the label rather than from a
            # position, so adding or removing an architecture leaves the
            # others bit-for-bit reproducible. crc32, not hash(), which is
            # salted per process and would differ across workers.
            torch.manual_seed(seed * 31 + zlib.crc32(label.encode()))
            # The calibration draw, in contrast, does not depend on the
            # architecture at all: every model gets the same threshold.
            model = SparseRegressor(
                hidden_dims=dims,
                alpha=ALPHA,
                n_mc=N_MC,
                max_iter=MAX_ITER,
            )
            model.fit(X, y, generator=torch.Generator().manual_seed(seed + 7))

            selected = set(model.selected_)
            # `lambda_` is typed optional because it is None before fit;
            # a successful fit always leaves a float in it.
            lam_value = model.lambda_
            row.update(
                exact=float(selected == truth),
                n_selected=len(selected),
                true_positives=len(selected & truth),
                false_positives=len(selected - truth),
                lam=math.nan if lam_value is None else float(lam_value),
            )
        except Exception as error:  # noqa: BLE001 - a run must survive one draw
            row["failed"] = True
            row["error"] = f"{type(error).__name__}: {error}"

        row["seconds"] = time.perf_counter() - started
        rows.append(row)

    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(label: str, rows: list[dict]) -> dict:
    """Collapse one architecture's replicates into a CSV row."""
    done = [r for r in rows if not r["failed"]]
    n_done = len(done)

    recovery = (
        statistics.fmean(r["exact"] for r in done) if done else math.nan
    )
    # Binomial standard error of the recovery probability.
    recovery_se = (
        math.sqrt(recovery * (1.0 - recovery) / n_done)
        if n_done > 1
        else math.nan
    )

    return {
        "model": label,
        "n": N,
        "p": P,
        "sigma": SIGMA,
        "n_reps": n_done,
        "n_failed": len(rows) - n_done,
        "exact_recovery": recovery,
        "exact_recovery_se": recovery_se,
        "tpr": (
            statistics.fmean(r["true_positives"] / N_SIGNAL for r in done)
            if done
            else math.nan
        ),
        "mean_false_positives": (
            statistics.fmean(r["false_positives"] for r in done)
            if done
            else math.nan
        ),
        "mean_n_selected": (
            statistics.fmean(r["n_selected"] for r in done)
            if done
            else math.nan
        ),
        "lambda_mean": (
            statistics.fmean(r["lam"] for r in done) if done else math.nan
        ),
        "seconds_per_fit": statistics.fmean(r["seconds"] for r in rows),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    print(
        f"Hardle et al. (2004): n = {N}, p = {P}, sigma = {SIGMA}, "
        f"{N_REPS} replicates, {len(ARCHITECTURES)} architectures "
        f"-> {N_REPS * len(ARCHITECTURES)} fits"
    )
    print(f"writing to {OUTPUT}\n", flush=True)

    started = time.perf_counter()
    if N_WORKERS > 0:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS, mp_context=get_context("spawn")
        ) as pool:
            batches = list(
                pool.map(run_replicate, range(N_REPS), chunksize=1)
            )
    else:
        batches = [run_replicate(rep) for rep in range(N_REPS)]

    rows = [row for batch in batches for row in batch]
    summary = [
        summarise(label, [r for r in rows if r["model"] == label])
        for label in ARCHITECTURES
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    elapsed = time.perf_counter() - started
    print(f"done in {elapsed:.1f}s")
    for row in summary:
        failed = f"  [{row['n_failed']} failed]" if row["n_failed"] else ""
        print(
            f"  {row['model']:>7s} | "
            f"P(exact recovery) {row['exact_recovery']:.2f} "
            f"+/- {row['exact_recovery_se']:.2f} | "
            f"TPR {row['tpr']:.2f} | "
            f"FP {row['mean_false_positives']:.2f}{failed}"
        )
    print(f"\nresults -> {OUTPUT}")


if __name__ == "__main__":
    main()
