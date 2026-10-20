"""Realism report: checks a simulated match against typical top-flight ranges.

Everything here is computed from the published outputs (events and
tracking), not from simulator internals, so it also checks the data contract.
The ranges are approximate per-team or per-player spreads for elite men's
league football, wide enough to allow for normal match-to-match variation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

FPS = 10
SPRINT_SPEED = 7.0  # m/s (25.2 km/h)


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    low: float
    high: float
    unit: str = ""

    @property
    def ok(self) -> bool:
        return self.low <= self.value <= self.high


TEAM_RANGES = {
    "shots": (5, 24),
    "shots_on_target": (1, 10),
    "xg": (0.3, 3.2),
    "goals": (0, 5),
    "passes": (280, 750),
    "pass_completion_pct": (70, 92),
    "possession_pct": (28, 72),
    "corners": (1, 11),
    "fouls": (5, 17),
    "yellow_cards": (0, 5),
    "offsides": (0, 6),
    "tackles_won": (5, 25),
    "interceptions": (3, 18),
    "pressures": (90, 260),
    "crosses": (5, 30),
}

MATCH_RANGES = {
    "ball_in_play_min": (50, 64),
    "match_length_min": (93, 102),
}

PLAYER_RANGES = {
    "distance_km_full_match": (8.8, 13.0),
    "top_speed_ms": (7.4, 10.0),
    "sprints_full_match": (5, 60),
}


def team_stats(events: list[dict], tracking: dict, team_ids: tuple[str, str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    owner = tracking["owner"]
    owned = owner[owner >= 0]
    for idx, tid in enumerate(team_ids):
        mine = [e for e in events if e["team"] == tid]
        c = Counter(e["type"] for e in mine)
        passes = [e for e in mine if e["type"] == "pass"]
        shots = [e for e in mine if e["type"] == "shot"]
        complete = sum(1 for e in passes if e["outcome"] == "complete")
        on_target = sum(1 for e in shots if e["outcome"] in ("goal", "saved"))
        corners = sum(1 for e in events if e["type"] == "ball_out"
                      and e["attributes"].get("restart") == "corner" and e["team"] != tid)
        stats[tid] = {
            "goals": c["goal"],
            "shots": len(shots),
            "shots_on_target": on_target,
            "xg": round(sum(e["attributes"].get("xg", 0.0) for e in shots), 2),
            "passes": len(passes),
            "pass_completion_pct": round(100.0 * complete / max(len(passes), 1), 1),
            "possession_pct": round(100.0 * float(np.mean((owned // 11) == idx)) if len(owned) else 0.0, 1),
            "corners": corners,
            "fouls": c["foul"],
            "yellow_cards": sum(1 for e in mine if e["type"] == "card" and e["attributes"].get("card") == "yellow"),
            "offsides": c["offside"],
            "tackles_won": sum(1 for e in mine if e["type"] == "tackle" and e["outcome"] == "won"),
            "interceptions": c["interception"],
            "pressures": c["pressure"],
            "crosses": sum(1 for e in passes if e["attributes"].get("kind") == "cross"),
            "dribbles": c["dribble"],
            "carries": c["carry"],
            "clearances": c["clearance"],
        }
    return stats


def player_physical(tracking: dict) -> dict[int, dict[str, float]]:
    """Distance, top speed, sprints and minutes per player, from tracking alone."""
    xy = tracking["xy"].astype(np.float64)
    ids = tracking["player_index"]
    period = tracking["period"]
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1))  # (n-1, 22)
    same = (ids[1:] == ids[:-1]) & (period[1:] == period[:-1])[:, None]
    step = np.where(same & np.isfinite(step) & (step < 1.3), step, np.nan)
    speed = step * FPS
    out: dict[int, dict[str, float]] = {}
    for pid in np.unique(ids):
        mask = ids[1:] == pid
        sp = np.where(mask, speed, np.nan)
        col = np.nanmax(np.where(np.isfinite(sp), sp, -1), axis=1)  # one slot per player at a time
        valid = col >= 0
        if not valid.any():
            continue
        series = np.where(valid, col, 0.0)
        smooth = np.convolve(series, np.ones(5) / 5, mode="same")
        fast = (smooth > SPRINT_SPEED) & valid
        edges = np.diff(np.concatenate([[0], fast.astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        sprints = int(np.sum((ends - starts) >= FPS))
        out[int(pid)] = {
            "minutes": round(valid.sum() / FPS / 60.0, 1),
            "distance_km": round(float(np.nansum(col[valid])) / FPS / 1000.0, 2),
            "top_speed_ms": round(float(smooth[valid].max()), 2),
            "sprints": sprints,
        }
    return out


def realism_checks(metadata: dict, events: list[dict], tracking: dict) -> tuple[list[Check], dict]:
    home, away = metadata["teams"]["home"]["id"], metadata["teams"]["away"]["id"]
    stats = team_stats(events, tracking, (home, away))
    checks: list[Check] = []
    for tid in (home, away):
        for key, (lo, hi) in TEAM_RANGES.items():
            checks.append(Check(f"{tid} {key}", stats[tid][key], lo, hi))
    state = tracking["state"]
    checks.append(Check("ball in play (min)", round(float(np.sum(state != 3)) / FPS / 60.0, 1),
                        *MATCH_RANGES["ball_in_play_min"]))
    checks.append(Check("match length (min)", round(len(state) / FPS / 60.0, 1), *MATCH_RANGES["match_length_min"]))
    phys = player_physical(tracking)
    players = {p["index"]: p for p in metadata["players"]}
    full = {k: v for k, v in phys.items() if v["minutes"] >= 88 and players[k]["position"] != "GK"}
    if full:
        dist = [v["distance_km"] for v in full.values()]
        checks.append(Check("outfield distance, median (km)", round(float(np.median(dist)), 2),
                            *PLAYER_RANGES["distance_km_full_match"]))
        checks.append(Check("outfield distance, min (km)", min(dist), 7.5, 13.5))
        checks.append(Check("outfield distance, max (km)", max(dist), 8.5, 14.0))
        tops = [v["top_speed_ms"] for v in full.values()]
        checks.append(Check("top speed, median (m/s)", round(float(np.median(tops)), 2), *PLAYER_RANGES["top_speed_ms"]))
        checks.append(Check("top speed, max (m/s)", max(tops), 8.0, 10.2))
        spr = [v["sprints"] for v in full.values()]
        checks.append(Check("sprints per player, median", float(np.median(spr)), *PLAYER_RANGES["sprints_full_match"]))
    return checks, {"teams": stats, "players": phys}


def format_report(metadata: dict, checks: list[Check], detail: dict) -> str:
    home, away = metadata["teams"]["home"], metadata["teams"]["away"]
    s = metadata["score"]
    lines = [
        f"{home['name']} {s['home']} - {s['away']} {away['name']}  "
        f"(HT {metadata['half_time_score']['home']}-{metadata['half_time_score']['away']}, "
        f"seed {metadata['seed']}, storyline {metadata['storyline']['name']})",
        "",
    ]
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "ok " if c.ok else "OUT"
        lines.append(f"  [{mark}] {c.name:<{width}}  {c.value:>8}   expected {c.low}-{c.high}")
    passed = sum(c.ok for c in checks)
    lines += ["", f"{passed}/{len(checks)} checks inside the expected range"]
    return "\n".join(lines)
