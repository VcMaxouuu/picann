"""Recovery of an additive absolute-value signal, over ``n`` and ``s``.

Monte-Carlo study of one architecture, ``hidden_dims=(32,)``, on

.. math::
    y_i = \\gamma \\sum_{j \\in S} |x_{ij}| + \\sigma \\, \\epsilon_i,
    \\qquad x \\sim N(0, I_p),

with a random support ``S`` of size ``s``. Since
:math:`\\mathrm{Var}(|x|) = 1 - 2/\\pi \\approx 0.3634`, the response
carries :math:`\\mathrm{Var}(y) = \\gamma^2 (1 - 2/\\pi)\\, s + \\sigma^2`;
at :math:`\\gamma = 5` each variable contributes an amplitude of about
``3.01``. Unlike a single-index target, the per-variable magnitude is
constant in ``s``: growing the support adds signal instead of diluting
it.

Replicates run in parallel over processes. The CSV -- same name as this
file -- gains one row per ``(n, s)`` cell as soon as the cell finishes, so
an interrupted run keeps its results, and re-running the script skips the
cells already present in the file. Delete the file to start over.
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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import torch

# The project is not installed as a package; make it importable from here
# (two levels up: simulations/regression/ -> project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deeppic import SparseRegressor  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings. Edit them here; the script takes no argument.
# --------------------------------------------------------------------------- #
#: Number of variables, sample sizes swept, and support sizes swept.
P = 50
N_LIST = (400, 800, 1200)
S_LIST = range(0, 16)

#: The single architecture fitted.
HIDDEN_DIMS = (32,)

#: Replicates per (n, s) cell.
N_REPS = 100

#: Per-variable amplitude and noise standard deviation: each support
#: variable contributes a variance of GAMMA**2 * (1 - 2/pi), about 9.09
#: at the default, whatever the support size.
GAMMA = 5.0
SIGMA = 1.0

#: Level and Monte-Carlo budget of the threshold calibration.
ALPHA = 0.05
N_MC = 1000

#: Iteration cap of each penalized phase.
MAX_ITER = 500

#: Base seed. The seed of a replicate is derived from it and the cell.
SEED = 0

#: Parallel processes. Set to 0 to run sequentially, for debugging.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

#: Results file: same name as this script, one row per (n, s) cell.
OUTPUT = Path(__file__).with_suffix(".csv")

CSV_FIELDS = [
    "n",
    "p",
    "s",
    "gamma",
    "sigma",
    "n_reps",
    "n_failed",
    "pesr",
    "fdr",
    "tpr",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_dataset(n: int, s: int, seed: int):
    """One draw of the design, the response and the support.

    ``y = gamma * sum_{j in S} |x_j| + sigma * eps``; ``s = 0`` draws pure
    noise and an empty support.
    """
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, P, generator=g)
    true = torch.randperm(P, generator=g)[:s]

    if s > 0:
        signal = GAMMA * X[:, true].abs().sum(dim=1)
    else:
        signal = torch.zeros(n)

    y = signal + SIGMA * torch.randn(n, generator=g)
    return X, y, sorted(true.tolist())


# --------------------------------------------------------------------------- #
# One replicate
# --------------------------------------------------------------------------- #
def run_replicate(task: tuple[int, int, int]) -> dict:
    """Fit the model on one draw of one (n, s) cell.

    The workers are spawned, so they re-import this module and read the
    settings above directly; only the cell and the replicate index travel.

    Returns a row with ``failed=True`` rather than raising, so that a single
    pathological draw cannot take down a long run.
    """
    n, s, rep = task
    torch.set_num_threads(1)

    seed = SEED + 1_000_003 * rep + 7_919 * s + n
    X, y, true = make_dataset(n, s, seed)
    truth = set(true)

    row = {"n": n, "s": s, "rep": rep, "failed": False}
    try:
        torch.manual_seed(seed * 31 + 1)
        model = SparseRegressor(
            hidden_dims=HIDDEN_DIMS,
            alpha=ALPHA,
            n_mc=N_MC,
            max_iter=MAX_ITER,
        )
        model.fit(X, y, generator=torch.Generator().manual_seed(seed + 7))

        selected = set(model.selected_)
        row.update(
            exact=float(selected == truth),
            n_selected=len(selected),
            true_positives=len(selected & truth),
            false_positives=len(selected - truth),
        )
    except Exception as error:  # noqa: BLE001 - a run must survive one draw
        row["failed"] = True
        row["error"] = f"{type(error).__name__}: {error}"

    return row


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(n: int, s: int, rows: list[dict]) -> dict:
    """Collapse the replicates of one (n, s) cell into a CSV row.

    Three probabilities, each a mean of per-replicate proportions:

    - ``pesr``: probability of exact support recovery.
    - ``fdr``: mean false discovery proportion ``FP / |selected|``, with
      the usual convention that an empty selection discovers nothing
      falsely, so its proportion is zero -- the denominator is
      ``max(1, |selected|)``.
    - ``tpr``: mean recovered fraction of the support ``TP / s``. At
      ``s = 0`` the ratio is undefined; a replicate then scores 1 when
      the model indeed returned no variable -- the empty support fully
      recovered -- and 0 otherwise, which makes ``tpr`` coincide with
      ``pesr`` there.
    """
    done = [r for r in rows if not r["failed"]]

    return {
        "n": n,
        "p": P,
        "s": s,
        "gamma": GAMMA,
        "sigma": SIGMA,
        "n_reps": len(done),
        "n_failed": len(rows) - len(done),
        "pesr": (
            statistics.fmean(r["exact"] for r in done) if done else math.nan
        ),
        "fdr": (
            statistics.fmean(
                r["false_positives"] / max(1, r["n_selected"]) for r in done
            )
            if done
            else math.nan
        ),
        "tpr": (
            statistics.fmean(
                r["true_positives"] / s
                if s > 0
                else float(r["n_selected"] == 0)
                for r in done
            )
            if done
            else math.nan
        ),
    }


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def completed_cells(path: Path) -> set[tuple[int, int]]:
    """The (n, s) cells already recorded in ``path``."""
    if not path.exists():
        return set()

    cells: set[tuple[int, int]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                cells.add((int(row["n"]), int(row["s"])))
            except (KeyError, ValueError):
                continue
    return cells


def append_row(path: Path, row: dict) -> None:
    """Append one row, writing the header the first time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    cells = [(n, s) for n in N_LIST for s in S_LIST]
    done = completed_cells(OUTPUT)
    todo = [cell for cell in cells if cell not in done]

    print(
        f"Additive |x|: p = {P}, gamma = {GAMMA}, sigma = {SIGMA}, "
        f"hidden_dims = {HIDDEN_DIMS}, {N_REPS} replicates"
    )
    print(
        f"{len(cells)} (n, s) cells, n in {list(N_LIST)}, "
        f"s in [{S_LIST[0]}, {S_LIST[-1]}] "
        f"-> {len(todo) * N_REPS} fits to run"
    )
    if done:
        print(f"resuming: {len(done)} cell(s) already in {OUTPUT}")
    print(f"writing to {OUTPUT}\n", flush=True)

    pool = None
    if N_WORKERS > 0:
        pool = ProcessPoolExecutor(
            max_workers=N_WORKERS, mp_context=get_context("spawn")
        )
    try:
        for n, s in todo:
            started = time.perf_counter()
            tasks = [(n, s, rep) for rep in range(N_REPS)]
            if pool is None:
                rows = [run_replicate(task) for task in tasks]
            else:
                rows = list(pool.map(run_replicate, tasks, chunksize=1))

            summary = summarise(n, s, rows)
            append_row(OUTPUT, summary)

            elapsed = time.perf_counter() - started
            failed = (
                f"  [{summary['n_failed']} failed]"
                if summary["n_failed"]
                else ""
            )
            print(
                f"n = {n:4d}  s = {s:2d}  ({elapsed:6.1f}s) | "
                f"PESR {summary['pesr']:.2f} | "
                f"FDR {summary['fdr']:.2f} | "
                f"TPR {summary['tpr']:.2f}{failed}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()

    print(f"\ndone -> {OUTPUT}")


if __name__ == "__main__":
    main()
