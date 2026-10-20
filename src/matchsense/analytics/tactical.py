"""Tactical state in rolling windows, and detection of when it changes.

Raw counts are kept in one-minute buckets. Any window metric is then a ratio
of sums over buckets, which stays stable even in quiet minutes. All metrics
are standard in analytics departments:

- possession: share of frames each team has the ball
- pressing: pressures in the opponent's half per minute of opponent possession
- PPDA: opponent passes in their own 60% per defensive action there (lower = more intense press)
- line height: how far from its own goal a team's back line holds out of possession (tracking)
- tempo: passes per minute of possession; directness: metres gained per pass
- field tilt: share of final-third passes
- momentum: exponentially weighted balance of threat created (xT and xG)
- control vs chaos: 0 (calm, long controlled possessions) to 100 (turnovers, loose balls, duels)

Tactical shifts are found by comparing the last few minutes with the baseline
before them. A change must be large in absolute and relative terms, and it
must persist, before it is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.models import expected_threat
from .context import MatchContext, clock_label
from .signals import Signal, SignalFactory

FPS = 10
FRAMES_PER_MIN = 60 * FPS


@dataclass
class Bucket:
    period: int
    minute: int
    frame: int
    t: float
    frames: int = 0
    inplay: int = 0
    loose: int = 0
    poss: list = field(default_factory=lambda: [0, 0])
    line_sum: list = field(default_factory=lambda: [0.0, 0.0])
    line_n: list = field(default_factory=lambda: [0, 0])
    high_line_sum: list = field(default_factory=lambda: [0.0, 0.0])
    press_gap_sum: list = field(default_factory=lambda: [0.0, 0.0])
    build_n: list = field(default_factory=lambda: [0, 0])
    pressures_high: list = field(default_factory=lambda: [0, 0])
    passes: list = field(default_factory=lambda: [0, 0])
    completed: list = field(default_factory=lambda: [0, 0])
    forward_m: list = field(default_factory=lambda: [0.0, 0.0])
    final_third: list = field(default_factory=lambda: [0, 0])
    def_actions_high: list = field(default_factory=lambda: [0, 0])
    opp_build_passes: list = field(default_factory=lambda: [0, 0])
    turnovers: list = field(default_factory=lambda: [0, 0])
    xt: list = field(default_factory=lambda: [0.0, 0.0])
    xg: list = field(default_factory=lambda: [0.0, 0.0])
    shots: list = field(default_factory=lambda: [0, 0])
    duels: int = 0
    possessions: int = 0
    stoppages: int = 0
    evidence: dict = field(default_factory=dict)

    def note(self, key: str, event_id: str) -> None:
        self.evidence.setdefault(key, []).append(event_id)


# metric: (label, minimum absolute change, minimum relative change, word if up, word if down, press polarity)
# Thresholds are about 2.5 standard deviations of the window-to-window noise measured on
# simulated matches; press polarity is +1 when an increase means pressing harder.
SHIFT_METRICS = {
    "high_line": ("line height while the opponent builds from the back", 5.0, 0.0,
                  "a higher line", "a deeper line", +1),
    "press_gap": ("distance from the nearest presser to the ball carrier", 1.8, 0.0,
                  "closing down further", "pressing tighter", -1),
    "possession": ("possession share", 18.0, 0.0, "more of the ball", "less of the ball", 0),
    "tempo": ("passing tempo", 6.0, 0.35, "quicker passing", "slower passing", 0),
    "directness": ("directness", 5.0, 0.0, "more direct", "more patient", 0),
}
# A coach's pressing change shows up as line height and press distance moving together.
PRESS_FAMILY = ("high_line", "press_gap")
RECENT, BASELINE, MIN_BUCKETS, PERSIST = 8, 18, 20, 3

# Reference ranges for the control-vs-chaos index: the 10th and 90th percentiles of
# 5-minute windows measured over simulated matches.
CHAOS_RANGES = {
    "turnovers_per_min": (1.0, 3.0),
    "loose_share": (0.01, 0.075),
    "duels_per_min": (1.8, 5.6),
    "possession_s": (15.0, 35.0),
}


def _norm(x: float, lo: float, hi: float) -> float:
    return min(max((x - lo) / (hi - lo), 0.0), 1.0)


class TacticalTracker:
    def __init__(self, ctx: MatchContext, signals: SignalFactory):
        self.ctx = ctx
        self.signals = signals
        self.buckets: list[Bucket] = []
        self.momentum = 0.0
        self.momentum_series: list[tuple[int, float]] = []  # (bucket index, value -100..100)
        self.chaos_series: list[tuple[int, float]] = []
        self._shift: dict[tuple[int, str], dict] = {}
        self._spell_cooldown = -1
        self._swing_cooldown = -1

    # ------------------------------------------------------------ ingestion

    def _bucket(self, period: int, t: float, frame: int) -> tuple[Bucket, bool]:
        minute = int(t // 60)
        if self.buckets and self.buckets[-1].period == period and self.buckets[-1].minute == minute:
            return self.buckets[-1], False
        self.buckets.append(Bucket(period, minute, frame, t))
        return self.buckets[-1], True

    def on_frame(self, frame) -> list[Signal]:
        b, new = self._bucket(frame.period, frame.t, frame.index)
        out = self._close_minute(len(self.buckets) - 2) if new and len(self.buckets) > 1 else []
        b.frames += 1
        if frame.state != 3:
            b.inplay += 1
            if frame.state == 2:
                b.loose += 1
        if frame.owner >= 0:
            b.poss[frame.owner // 11] += 1
        if frame.index % FPS == 0 and frame.state != 3:
            for ti, tid in enumerate(self.ctx.team_ids):
                if frame.owner >= 0 and frame.owner // 11 == ti:
                    continue
                pts = frame.xy[self.ctx.slots(tid)]
                pts = pts[np.isfinite(pts[:, 0])]
                rel = self.ctx.to_rel(pts, tid, frame.period)
                order = np.argsort(rel[:, 0])
                if len(rel) < 5:
                    continue
                back_line = float(rel[order[1:5], 0].mean())  # skip the goalkeeper
                b.line_sum[ti] += back_line
                b.line_n[ti] += 1
                # While the opponent builds from the back: how high is the line, how close is the press?
                ball_rel = self.ctx.to_rel(frame.ball[:2], tid, frame.period)[0]
                if frame.owner >= 0 and ball_rel > 52.5:
                    carrier = frame.xy[frame.owner]
                    outfield = pts[order[1:]]
                    b.high_line_sum[ti] += back_line
                    b.press_gap_sum[ti] += float(np.hypot(*(outfield - carrier).T).min())
                    b.build_n[ti] += 1
        return out

    def on_event(self, ev: dict) -> None:
        if not self.buckets:
            return
        b = self.buckets[-1]
        etype = ev["type"]
        team = ev["team"]
        ti = self.ctx.team_index(team) if team else None
        loc = ev.get("location")
        rel = self.ctx.to_rel([loc["x"], loc["y"]], team, ev["period"]) if (loc and team) else None
        if etype == "pass":
            b.passes[ti] += 1
            end = ev["end_location"]
            rel_end = self.ctx.to_rel([end["x"], end["y"]], team, ev["period"])
            if rel[0] < 63.0:
                b.opp_build_passes[1 - ti] += 1
            if ev["outcome"] == "complete":
                b.completed[ti] += 1
                b.forward_m[ti] += float(rel_end[0] - rel[0])
                if rel_end[0] > 70.0:
                    b.final_third[ti] += 1
                b.xt[ti] += max(0.0, expected_threat(*rel_end) - expected_threat(*rel))
        elif etype == "carry":
            end = ev["end_location"]
            rel_end = self.ctx.to_rel([end["x"], end["y"]], team, ev["period"])
            b.xt[ti] += max(0.0, expected_threat(*rel_end) - expected_threat(*rel))
        elif etype == "pressure":
            if rel[0] > 52.5:
                b.pressures_high[ti] += 1
                b.note(f"pressure_high:{ti}", ev["id"])
        elif etype in ("tackle", "interception", "foul"):
            if etype != "tackle" or ev["outcome"] == "won":
                if rel[0] >= 42.0:
                    b.def_actions_high[ti] += 1
            if etype == "tackle":
                b.duels += 1
            if etype == "foul":
                b.stoppages += 1
        elif etype in ("aerial_duel", "dribble"):
            b.duels += 1
        elif etype == "possession_change":
            b.possessions += 1
            if ev["attributes"].get("cause") in ("interception", "tackle", "loose"):
                b.turnovers[self.ctx.team_index(ev["attributes"]["from_team"])] += 1
        elif etype in ("ball_out", "offside"):
            b.stoppages += 1
        elif etype == "shot":
            b.shots[ti] += 1
            b.xg[ti] += float(ev["attributes"].get("xg", 0.0))
            b.note(f"shot:{ti}", ev["id"])

    # -------------------------------------------------------------- windows

    def window(self, n: int, end: int | None = None) -> dict:
        end = len(self.buckets) if end is None else end
        bs = self.buckets[max(0, end - n): end]
        if not bs:
            return {}

        def tot(name):
            return np.sum([getattr(b, name) for b in bs], axis=0)

        poss = tot("poss").astype(float)
        frames, inplay, loose = tot("frames"), tot("inplay"), tot("loose")
        passes, completed = tot("passes"), tot("completed")
        line_sum, line_n = tot("line_sum"), tot("line_n")
        high_line, press_gap, build_n = tot("high_line_sum"), tot("press_gap_sum"), tot("build_n")
        final_third = tot("final_third")
        teams = {}
        for ti, tid in enumerate(self.ctx.team_ids):
            opp_minutes = poss[1 - ti] / FRAMES_PER_MIN
            own_minutes = poss[ti] / FRAMES_PER_MIN
            teams[tid] = {
                "possession": round(float(100.0 * poss[ti] / max(poss.sum(), 1.0)), 1),
                "pressing": round(float(tot("pressures_high")[ti] / opp_minutes), 2) if opp_minutes >= 1.0 else None,
                "ppda": round(float(tot("opp_build_passes")[ti] / max(tot("def_actions_high")[ti], 1)), 1),
                "pass_completion": round(float(100.0 * completed[ti] / max(passes[ti], 1)), 1),
                "tempo": round(float(passes[ti] / own_minutes), 1) if own_minutes >= 2.0 else None,
                "directness": round(float(tot("forward_m")[ti] / max(completed[ti], 1)), 1) if completed[ti] >= 15 else None,
                "field_tilt": round(float(100.0 * final_third[ti] / max(final_third.sum(), 1)), 1),
                "line_height": round(float(line_sum[ti] / line_n[ti]), 1) if line_n[ti] >= 60 else None,
                "high_line": round(float(high_line[ti] / build_n[ti]), 1) if build_n[ti] >= 25 else None,
                "press_gap": round(float(press_gap[ti] / build_n[ti]), 2) if build_n[ti] >= 25 else None,
                "turnovers": int(tot("turnovers")[ti]),
                "xt": round(float(tot("xt")[ti]), 3),
                "xg": round(float(tot("xg")[ti]), 2),
                "shots": int(tot("shots")[ti]),
                "passes": int(passes[ti]),
            }
        inplay_min = max(inplay / FRAMES_PER_MIN, 0.5)
        match = {
            "minutes": len(bs),
            "ball_in_play_pct": round(float(100.0 * inplay / max(frames, 1)), 1),
            "turnovers_per_min": round(float(tot("turnovers").sum() / inplay_min), 2),
            "loose_share": round(float(loose / max(inplay, 1)), 3),
            "duels_per_min": round(float(tot("duels") / inplay_min), 2),
            "possession_s": round(float((inplay / FPS) / max(tot("possessions"), 1)), 1),
            "stoppages": int(tot("stoppages")),
        }
        match["chaos"] = self._chaos(match)
        match["rhythm"] = self._rhythm(match)
        return {"teams": teams, "match": match}

    @staticmethod
    def _chaos(m: dict) -> int:
        parts = [
            _norm(m["turnovers_per_min"], *CHAOS_RANGES["turnovers_per_min"]),
            _norm(m["loose_share"], *CHAOS_RANGES["loose_share"]),
            _norm(m["duels_per_min"], *CHAOS_RANGES["duels_per_min"]),
            1.0 - _norm(m["possession_s"], *CHAOS_RANGES["possession_s"]),
        ]
        return int(round(100.0 * float(np.mean(parts))))

    @staticmethod
    def _rhythm(m: dict) -> str:
        if m["ball_in_play_pct"] < 50:
            return "stop-start"
        if m["chaos"] >= 65:
            return "frantic"
        if m["chaos"] <= 35:
            return "patient"
        return "flowing"

    def momentum_value(self) -> int:
        return self.momentum_series[-1][1] if self.momentum_series else 0

    # -------------------------------------------------------- minute close

    def _close_minute(self, idx: int) -> list[Signal]:
        b = self.buckets[idx]
        out: list[Signal] = []
        net = (b.xt[0] - b.xt[1]) + 0.8 * (b.xg[0] - b.xg[1])
        decay = 0.5 ** (1 / 3)  # three-minute half-life
        self.momentum = self.momentum * decay + net * (1 - decay)
        value = int(round(100 * math.tanh(self.momentum / 0.08)))
        self.momentum_series.append((idx, value))
        recent = self.window(5, end=idx + 1)
        self.chaos_series.append((idx, recent["match"]["chaos"]))
        at = dict(frame=b.frame + b.frames, t=b.t + 60.0, period=b.period, clock=clock_label(b.period, b.t + 60.0))
        out += self._detect_swing(idx, value, at)
        out += self._detect_spell(idx, at)
        out += self._detect_shifts(idx, at)
        return out

    def _detect_swing(self, idx: int, value: int, at: dict) -> list[Signal]:
        if idx < self._swing_cooldown or len(self.momentum_series) < 6:
            return []
        past = [v for _, v in self.momentum_series[-9:-1]]
        peak = max(past, key=abs)
        if abs(value) >= 45 and abs(peak) >= 40 and np.sign(peak) != np.sign(value):
            self._swing_cooldown = idx + 10
            team = self.ctx.home if value > 0 else self.ctx.away
            return [self.signals.make("momentum_swing", team=team, facts={
                "momentum_now": value, "momentum_before": int(peak),
                "explanation": "momentum is a decaying balance of threat created (xT from passes and carries, plus xG)",
                "window_minutes": 8}, significance=0.45 + min(abs(value - peak) / 500, 0.25), **at)]
        return []

    def _detect_spell(self, idx: int, at: dict) -> list[Signal]:
        if idx < self._spell_cooldown or len(self.chaos_series) < 4:
            return []
        last = [c for _, c in self.chaos_series[-3:]]
        w = self.window(5, end=idx + 1)
        facts = {"chaos_index": last[-1], **{k: w["match"][k] for k in CHAOS_RANGES}, "window_minutes": 5}
        if min(last) >= 70:
            self._spell_cooldown = idx + 10
            return [self.signals.make("chaos_spell", facts=facts, significance=0.45, **at)]
        if max(last) <= 25:
            self._spell_cooldown = idx + 10
            teams = w["teams"]
            leader = max(teams, key=lambda tid: teams[tid]["possession"])
            facts["possession"] = teams[leader]["possession"]
            return [self.signals.make("control_spell", team=leader, facts=facts, significance=0.4, **at)]
        return []

    def _detect_shifts(self, idx: int, at: dict) -> list[Signal]:
        """Compare the last RECENT minutes with the BASELINE minutes before them, per team."""
        if idx + 1 < MIN_BUCKETS:
            return []
        out = []
        recent = self.window(RECENT, end=idx + 1)
        base = self.window(BASELINE, end=idx + 1 - RECENT)
        for ti, tid in enumerate(self.ctx.team_ids):
            team_state = self._shift.setdefault((ti, "*"), {"cooldown": -1})
            moves: dict[str, tuple[float, float, float, float]] = {}  # metric -> (before, after, change, strength)
            ready = []
            for metric, (label, min_abs, min_rel, up, down, _) in SHIFT_METRICS.items():
                st = self._shift.setdefault((ti, metric), {"pending": 0, "cooldown": -1})
                before, after = base["teams"][tid].get(metric), recent["teams"][tid].get(metric)
                if before is None or after is None:
                    st["pending"] = 0
                    continue
                change = after - before
                rel = abs(change) / max(abs(before), 1e-6)
                strength = min(abs(change) / min_abs, rel / min_rel if min_rel else 9.0)
                moves[metric] = (before, after, change, strength)
                st["pending"] = st["pending"] + 1 if strength >= 1.0 else 0
                if st["pending"] >= PERSIST and idx >= st["cooldown"] and idx >= team_state["cooldown"]:
                    ready.append(metric)
            if not ready:
                continue
            metric = max(ready, key=lambda m: moves[m][3])
            before, after, change, strength = moves[metric]
            label, _, _, up, down, polarity = SHIFT_METRICS[metric]
            press_dir = polarity * np.sign(change)
            supporting = {}
            for m, v in moves.items():
                if m == metric or v[3] < 0.8:
                    continue
                pol = SHIFT_METRICS[m][5]
                agrees = (pol * np.sign(v[2]) == press_dir) if (pol and polarity) else np.sign(v[2]) == np.sign(change)
                if agrees:
                    supporting[m] = {"before": v[0], "after": v[1]}
            pressing_change = metric in PRESS_FAMILY and any(m in PRESS_FAMILY for m in supporting)
            for m in [metric, *supporting]:
                self._shift[(ti, m)].update(pending=0, cooldown=idx + 15)
            team_state["cooldown"] = idx + 5
            opp = self.ctx.other(tid)
            facts = {
                "metric": "press" if pressing_change else metric,
                "label": ("pushed up and pressed higher" if press_dir > 0 else "dropped into a deeper block")
                         if pressing_change else label,
                "summary": up if change > 0 else down,
                "pressing_harder": bool(press_dir > 0) if pressing_change else None,
                "measure": metric, "before": before, "after": after, "change": round(change, 2),
                "supporting": supporting,
                "recent_minutes": RECENT, "baseline_minutes": BASELINE,
                "context": {
                    "pressures_per_min_before": base["teams"][tid]["pressing"],
                    "pressures_per_min_after": recent["teams"][tid]["pressing"],
                    "possession_before": base["teams"][tid]["possession"],
                    "possession_after": recent["teams"][tid]["possession"],
                    "ppda_before": base["teams"][tid]["ppda"], "ppda_after": recent["teams"][tid]["ppda"],
                    "opponent_pass_completion_before": base["teams"][opp]["pass_completion"],
                    "opponent_pass_completion_after": recent["teams"][opp]["pass_completion"],
                    "shots_before": base["teams"][tid]["shots"], "shots_after": recent["teams"][tid]["shots"],
                },
            }
            evidence = []
            if pressing_change:
                for b in self.buckets[idx + 1 - RECENT: idx + 1]:
                    evidence += b.evidence.get(f"pressure_high:{ti}", [])
            significance = (0.75 if pressing_change else 0.4) + min(strength - 1.0, 1.0) * 0.15
            out.append(self.signals.make("tactical_shift", team=tid, facts=facts, evidence=evidence[:12],
                                         significance=significance, **at))
        return out
