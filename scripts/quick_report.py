"""Simulate one match in memory and print the realism report (for tuning).

    python scripts/quick_report.py [seed] [storyline]
"""

import sys
import time
from collections import Counter

from matchsense.sim.engine import simulate
from matchsense.sim.report import format_report, realism_checks
from matchsense.sim.squads import load_default_teams

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
story = sys.argv[2] if len(sys.argv) > 2 else "balanced"
home, away = load_default_teams()
start = time.time()
result = simulate(home, away, seed=seed, storyline=story)
checks, detail = realism_checks(result.metadata, result.events, result.tracking)
print(format_report(result.metadata, checks, detail))
for tid, st in detail["teams"].items():
    print(tid, {k: st[k] for k in ("dribbles", "carries", "clearances")})
print("restarts", Counter(e["attributes"].get("restart") for e in result.events if e["type"] == "ball_out"))
print(f"simulated in {time.time() - start:.1f}s")
