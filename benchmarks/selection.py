r"""Selection harness: fit a scenario over many seeds and report how well the
support is recovered.

For every seed: exact recovery (selected set equal to the support), true and
false positives, and the time of the fit. Seeds run in parallel processes of
one thread each, so that the selections do not depend on how many seeds share
the machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fit(task: tuple[str, int, float | None]) -> dict:
    name, seed, tol = task
    import torch

    torch.set_num_threads(1)
    from scenarios import make

    from deeppic import Regressor

    X, y, support, hidden = make(name, seed)
    torch.manual_seed(seed)
    kwargs = {} if tol is None else {"tol": tol}
    model = Regressor(X.shape[1], hidden, **kwargs)
    epochs: list[int] = []
    # The phases fit runs: fit_phase, or _fit_phase in the versions that had it.
    method = "_fit_phase" if hasattr(model, "_fit_phase") else "fit_phase"
    phase = getattr(model, method)

    def counted(*args, **kwargs):
        history = phase(*args, **kwargs)
        epochs.append(len(history))
        return history

    setattr(model, method, counted)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        model.fit(X, y, generator=torch.Generator().manual_seed(seed))
    elapsed = time.perf_counter() - start
    selected = model.selected_indices
    return {
        "scenario": name,
        "seed": seed,
        "time": elapsed,
        "epochs": epochs,
        "unconverged": sum(issubclass(w.category, RuntimeWarning) for w in caught),
        "selected": selected,
        "support": support,
        "tp": len(set(selected) & set(support)),
        "fp": len(set(selected) - set(support)),
        "exact": set(selected) == set(support),
    }


def run(name: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tol", type=float, default=None, help="tolerance of the fit")
    parser.add_argument("--scenario", default=name, help="scenario of scenarios.py")
    parser.add_argument("--out", type=Path, default=None, help="JSON file for the records")
    args = parser.parse_args()
    name = args.scenario

    seeds = range(args.first_seed, args.first_seed + args.seeds)
    with ProcessPoolExecutor(args.jobs) as pool:
        records = list(pool.map(_fit, [(name, seed, args.tol) for seed in seeds]))

    for record in records:
        flag = "" if record["exact"] else "   <-"
        print(
            f"seed {record['seed']:>3}  {record['time']:6.2f} s  "
            f"epochs {sum(record['epochs']):>5d}  unconverged {record['unconverged']}  "
            f"tp {record['tp']}  fp {record['fp']}  selected {record['selected']}  "
            f"support {record['support']}{flag}"
        )
    size = len(records[0]["support"])
    print(
        f"\n{name}: {len(records)} seeds, exact recovery "
        f"{sum(r['exact'] for r in records)}/{len(records)}, "
        f"mean TP {statistics.mean(r['tp'] for r in records):.2f}/{size}, "
        f"mean FP {statistics.mean(r['fp'] for r in records):.2f}, "
        f"seeds with FP {sum(r['fp'] > 0 for r in records)}, "
        f"unconverged phases {sum(r['unconverged'] for r in records)}, "
        f"mean epochs {statistics.mean(sum(r['epochs']) for r in records):.0f}, "
        f"mean time {statistics.mean(r['time'] for r in records):.2f} s"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=1))
