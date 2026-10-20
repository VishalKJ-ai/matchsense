"""Run several seeds and summarise how often each realism check passes (for tuning)."""
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from matchsense.sim.engine import simulate
from matchsense.sim.report import realism_checks
from matchsense.sim.squads import load_default_teams


def run(seed: int):
    home, away = load_default_teams()
    r = simulate(home, away, seed=seed, storyline=sys.argv[2] if len(sys.argv) > 2 else "balanced")
    checks, _ = realism_checks(r.metadata, r.events, r.tracking)
    return seed, (r.metadata["score"]["home"], r.metadata["score"]["away"]), [(c.name, c.value, c.ok, c.low, c.high) for c in checks]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    values = defaultdict(list)
    fails = defaultdict(int)
    with ProcessPoolExecutor() as ex:
        for seed, score, checks in ex.map(run, range(100, 100 + n)):
            print(seed, score, f"{sum(c[2] for c in checks)}/{len(checks)}")
            for name, value, ok, lo, hi in checks:
                key = name.replace("IRV ", "").replace("WMA ", "")
                values[(key, lo, hi)].append(value)
                fails[(key, lo, hi)] += not ok
    print()
    for (key, lo, hi), vals in values.items():
        flag = "  <-- " if fails[(key, lo, hi)] else ""
        print(f"{key:<34} mean {np.mean(vals):8.2f}  range {min(vals):7.2f}-{max(vals):7.2f}   target {lo}-{hi}  fails {fails[(key, lo, hi)]}{flag}")
