"""Break down shots and xG by what created the chance (for tuning)."""
import sys
from collections import defaultdict

from matchsense.sim.engine import simulate
from matchsense.sim.squads import load_default_teams

home, away = load_default_teams()
agg = defaultdict(lambda: [0, 0.0])
n = 0
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    r = simulate(home, away, seed=200 + seed)
    n += 1
    ev = r.events
    by_poss = defaultdict(list)
    for e in ev:
        by_poss[e["possession"]].append(e)
    for e in ev:
        if e["type"] != "shot":
            continue
        sp = e["attributes"].get("set_piece")
        if sp:
            src = sp
        else:
            prior = [x for x in by_poss[e["possession"]] if x["index"] < e["index"] and x["type"] in ("pass", "carry", "dribble", "ball_recovery", "interception", "tackle", "goalkeeper", "block")]
            last_pass = next((x for x in reversed(prior) if x["type"] == "pass"), None)
            if e["attributes"]["body_part"] == "head":
                src = "header"
            elif prior and prior[-1]["type"] in ("block", "goalkeeper"):
                src = "rebound"
            elif last_pass is not None and last_pass["attributes"]["kind"] in ("cross", "through"):
                src = last_pass["attributes"]["kind"]
            elif len([x for x in prior if x["type"] == "pass"]) <= 2:
                src = "quick (<=2 passes)"
            else:
                src = "build-up"
        agg[src][0] += 1
        agg[src][1] += e["attributes"]["xg"]
total_s = sum(v[0] for v in agg.values())
total_x = sum(v[1] for v in agg.values())
print(f"per team per match: shots {total_s / n / 2:.1f}, xG {total_x / n / 2:.2f}")
for k, (c, x) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:<20} shots/team {c / n / 2:5.1f}  xG/team {x / n / 2:5.2f}  xG/shot {x / max(c, 1):.3f}")
