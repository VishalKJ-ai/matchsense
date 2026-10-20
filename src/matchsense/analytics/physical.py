"""Speed, distance and sprints from tracking, with broadcast-style triggers.

Speeds are smoothed over half a second, as optical tracking providers do, so
one noisy frame never fires an on-screen speed tag. The triggers follow the
brief: a player's speed indicator appears when they pass a set threshold,
and a distance tag appears at set milestones.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .context import MatchContext, clock_label
from .signals import Signal, SignalFactory

FPS = 10
SPRINT_SPEED = 7.0  # m/s, 25.2 km/h: the usual high-speed sprint threshold
BURST_SPEED = 8.6  # m/s, 31 km/h: shows an on-screen speed tag
BURST_COOLDOWN = 4 * 60 * FPS  # frames before the same player can trigger again
DISTANCE_MILESTONES_KM = (10.0, 12.0)


class PhysicalTracker:
    def __init__(self, ctx: MatchContext, signals: SignalFactory):
        self.ctx = ctx
        self.signals = signals
        self.prev_xy: np.ndarray | None = None
        self.prev_ids: np.ndarray | None = None
        self.prev_period = 0
        self.ring: deque[np.ndarray] = deque(maxlen=5)
        self.smooth = np.zeros(22)
        self.sprint_run = np.zeros(22, dtype=int)
        self.stats: dict[str, dict] = {}
        self.match_top = 0.0
        self.next_burst: dict[str, int] = {}
        self.milestones: set[tuple[str, float]] = set()

    def player(self, pid: str) -> dict:
        if pid not in self.stats:
            self.stats[pid] = {"distance_m": 0.0, "top_speed_ms": 0.0, "sprints": 0, "frames": 0,
                               "high_speed_m": 0.0}
        return self.stats[pid]

    def update(self, frame) -> list[Signal]:
        xy = frame.xy.astype(np.float64)
        ids = frame.player_index
        out: list[Signal] = []
        if self.prev_xy is None or frame.period != self.prev_period:
            self.prev_xy, self.prev_ids, self.prev_period = xy, ids.copy(), frame.period
            self.ring.clear()
            return out
        step = np.hypot(*(xy - self.prev_xy).T)
        valid = (ids == self.prev_ids) & np.isfinite(step) & (step < 1.3)
        speed = np.where(valid, step * FPS, np.nan)
        self.ring.append(speed)
        stacked = np.vstack(self.ring)
        count = np.isfinite(stacked).sum(axis=0)
        smooth = np.where(count > 0, np.nansum(stacked, axis=0) / np.maximum(count, 1), 0.0)
        previous = self.smooth
        self.smooth = np.nan_to_num(smooth)
        for slot in np.flatnonzero(valid):
            pid = self.ctx.by_index[int(ids[slot])]["id"]
            st = self.player(pid)
            st["distance_m"] += float(step[slot])
            st["frames"] += 1
            v = float(self.smooth[slot])
            if v > SPRINT_SPEED:
                st["high_speed_m"] += float(step[slot])
                self.sprint_run[slot] += 1
                if self.sprint_run[slot] == FPS:
                    st["sprints"] += 1
            else:
                self.sprint_run[slot] = 0
            if v > st["top_speed_ms"]:
                st["top_speed_ms"] = v
            if previous[slot] < BURST_SPEED <= v and frame.index >= self.next_burst.get(pid, 0):
                self.next_burst[pid] = frame.index + BURST_COOLDOWN
                fastest = v > self.match_top + 0.05 and frame.t > 300
                out.append(self.signals.make(
                    "fastest_sprint" if fastest else "speed_burst",
                    frame=frame.index, t=frame.t, period=frame.period, clock=clock_label(frame.period, frame.t),
                    team=self.ctx.player_team(pid), players=[pid],
                    facts={"speed_kmh": round(v * 3.6, 1), "distance_km": round(st["distance_m"] / 1000, 2),
                           "sprints": st["sprints"], "match_top_kmh": round(max(self.match_top, v) * 3.6, 1)},
                    significance=0.45 if fastest else 0.25))
            self.match_top = max(self.match_top, v)
            km = st["distance_m"] / 1000.0
            for mark in DISTANCE_MILESTONES_KM:
                if km >= mark and (pid, mark) not in self.milestones:
                    self.milestones.add((pid, mark))
                    out.append(self.signals.make(
                        "distance_milestone", frame=frame.index, t=frame.t, period=frame.period,
                        clock=clock_label(frame.period, frame.t), team=self.ctx.player_team(pid), players=[pid],
                        facts={"distance_km": mark, "minutes_played": round(st["frames"] / FPS / 60, 1),
                               "sprints": st["sprints"]},
                        significance=0.2))
        self.prev_xy, self.prev_ids = xy, ids.copy()
        return out

    def current_speed_kmh(self, slot: int) -> float:
        return round(float(self.smooth[slot]) * 3.6, 1)
