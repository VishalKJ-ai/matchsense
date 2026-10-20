"""Evaluate tactical-shift detection against the simulator's scripted storyline changes.

    python scripts/evaluate_shifts.py [matches_per_storyline]

On "comeback" matches, Westmarsh switch to a high press at 58' and Ironvale drop
off at 62'. A detection counts if a high-confidence pressing shift for the right
team, in the right direction, arrives within 20 minutes. On "balanced" matches
nothing is scripted, so every high-confidence pressing shift is a false alarm.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from matchsense.analytics.engine import analyse
from matchsense.sim import load_default_teams, simulate
from matchsense.sim.io import StoredMatch

TRUTH = {"comeback": [("WMA", 58, "up"), ("IRV", 62, "down")], "balanced": []}


def run(args):
    story, seed = args
    home, away = load_default_teams()
    r = simulate(home, away, seed=seed, storyline=story)
    a = analyse(StoredMatch(r.metadata, r.events, r.tracking, r.truth))
    found = []
    for s in a.emitted:
        if s.kind == "tactical_shift" and s.facts["metric"] == "press":
            found.append((s.team, s.t / 60.0, "up" if s.facts["pressing_harder"] else "down", s.significance))
    return story, seed, found


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(story, 7000 + i) for story in TRUTH for i in range(n)]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run, jobs))
    hits, delays, false_alarms, truth_total = 0, [], 0, 0
    balanced_alarms = 0
    for story, seed, found in results:
        matched = set()
        for team, minute, direction in TRUTH[story]:
            truth_total += 1
            cands = [f for f in found if f[0] == team and f[2] == direction and minute <= f[1] <= minute + 20]
            if cands:
                hits += 1
                delays.append(min(c[1] for c in cands) - minute)
                matched.update(cands)
        extra = [f for f in found if f not in matched]
        false_alarms += len(extra)
        if story == "balanced":
            balanced_alarms += len(found)
        print(f"{story:<9} seed {seed}: " + ", ".join(f"{t} {d} @{m:.0f}'" for t, m, d, _ in found))
    n_bal = sum(1 for s, _, _ in results if s == "balanced")
    print()
    print(f"recall: {hits}/{truth_total} scripted changes detected"
          + (f", median delay {np.median(delays):.1f} min" if delays else ""))
    print(f"false alarms on unscripted matches: {balanced_alarms} in {n_bal} matches")
    print(f"unmatched pressing shifts overall: {false_alarms}")


if __name__ == "__main__":
    main()
