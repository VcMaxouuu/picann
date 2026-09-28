r"""Compare two runs of :mod:`bench`: time, epochs, memory and selections.

::

    python benchmarks/compare.py before.json after.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def _load(path: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for record in json.load(open(path)):
        runs[record["scenario"]][record["seed"]] = record
    return runs


def main() -> None:
    before, after = _load(sys.argv[1]), _load(sys.argv[2])
    print(
        "| scenario | fits | time before (s) | time after (s) | speed-up "
        "| epochs before | epochs after | peak RSS before (MB) | peak RSS after (MB) "
        "| same selections |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in before:
        seeds = sorted(set(before[name]) & set(after.get(name, {})))
        if not seeds:
            continue
        old = [before[name][seed] for seed in seeds]
        new = [after[name][seed] for seed in seeds]
        t_old = statistics.mean(r["time"] for r in old)
        t_new = statistics.mean(r["time"] for r in new)
        e_old = statistics.mean(sum(r["epochs"]) for r in old)
        e_new = statistics.mean(sum(r["epochs"]) for r in new)
        m_old = max(r["peak_rss_mb"] for r in old)
        m_new = max(r["peak_rss_mb"] for r in new)
        same = sum(a["selected"] == b["selected"] for a, b in zip(old, new))
        print(
            f"| {name} | {len(seeds)} | {t_old:.2f} | {t_new:.2f} | {t_old / t_new:.2f}x "
            f"| {e_old:.0f} | {e_new:.0f} | {m_old:.0f} | {m_new:.0f} | {same}/{len(seeds)} |"
        )
        for a, b in zip(old, new):
            if a["selected"] != b["selected"]:
                print(
                    f"|   seed {a['seed']}: {a['selected']} -> {b['selected']} "
                    f"(support {a['support']}) | | | | | | | | | |"
                )


if __name__ == "__main__":
    main()
