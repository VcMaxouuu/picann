r"""Time the fit on the benchmark scenarios.

For every scenario and seed, one full :meth:`~deeppic.Regressor.fit`
(calibration included) is timed, and the number of epochs of every phase, the
peak memory and the selected variables are recorded. Every scenario runs in a
fresh process, so that its peak resident memory is its own.

The package is imported from ``PYTHONPATH``, so that the same script measures
any revision::

    PYTHONPATH=. python benchmarks/bench.py --out benchmarks/results/after.json
    git worktree add /tmp/base <commit>
    PYTHONPATH=/tmp/base python benchmarks/bench.py --out benchmarks/results/before.json
    python benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SEEDS = {"linear_p100": 5, "linear_p1000": 5, "linear_p5000": 3, "nonlinear_p100": 5}


def _worker(name: str, seeds: list[int]) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scenarios import make

    from deeppic import Regressor

    # Warm-up: thread pools and lazy initialisations stay out of the timings.
    X, y, _, hidden = make(name, seed=10_000)
    torch.manual_seed(0)
    Regressor(X.shape[1], hidden, n_phases=1).fit(X, y, n_epochs=20)

    for seed in seeds:
        X, y, support, hidden = make(name, seed)
        torch.manual_seed(seed)
        model = Regressor(X.shape[1], hidden)
        epochs: list[int] = []
        # The phases fit runs: fit_phase, or _fit_phase in the versions that had it.
        method = "_fit_phase" if hasattr(model, "_fit_phase") else "fit_phase"
        phase = getattr(model, method)

        def counted(*args, **kwargs):
            history = phase(*args, **kwargs)
            epochs.append(len(history))
            return history

        setattr(model, method, counted)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = time.perf_counter()
        model.fit(X, y, generator=torch.Generator().manual_seed(seed))
        elapsed = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            "scenario": name,
            "seed": seed,
            "time": elapsed,
            "epochs": epochs,
            "selected": model.selected_indices,
            "support": support,
            "lambda": float(model.lambda_),
            "peak_rss_mb": peak / 1024.0,
            "rss_growth_mb": (peak - rss) / 1024.0,
            "threads": torch.get_num_threads(),
        }
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--seeds", type=int, default=None, help="seeds per scenario")
    parser.add_argument("--out", type=Path, default=None, help="JSON file for the records")
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker[0], [int(seed) for seed in args.worker[1:]])
        return

    records = []
    for name in args.scenarios:
        count = args.seeds or DEFAULT_SEEDS.get(name, 3)
        command = [sys.executable, __file__, "--worker", name, *map(str, range(count))]
        output = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ)
        for line in output.stdout.splitlines():
            if line.startswith("{"):
                records.append(json.loads(line))
                record = records[-1]
                print(
                    f"{record['scenario']:>16}  seed {record['seed']:>2}  "
                    f"{record['time']:8.2f} s  epochs {sum(record['epochs']):>5d} "
                    f"{record['epochs']}  peak {record['peak_rss_mb']:7.1f} MB  "
                    f"selected {record['selected']}  support {record['support']}",
                    flush=True,
                )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
