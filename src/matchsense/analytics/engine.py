"""Live analytics: consume tracking frames and events as they arrive, keep match state, emit signals.

    analytics = LiveAnalytics(metadata)
    for kind, item in replay(match):
        new_signals = analytics.on_frame(item) if kind == "frame" else analytics.on_event(item)

Nothing here reads simulator internals: inputs are the event feed and the
tracking feed, exactly as a broadcast data provider would supply them.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..sim.models import expected_threat
from .context import MatchContext, clock_label
from .passing import PassDifficultyModel, pass_features
from .physical import PhysicalTracker
from .signals import Signal, SignalFactory
from .tactical import TacticalTracker

FRAME_MEMORY = 1200  # keep 2 minutes of tracking for late-arriving events
HARD_PASS = 65
SHOT_SPEED_TAG = 100 / 3.6  # m/s
BIG_CHANCE_XG = 0.25
PASS_MILESTONES = (50, 75)


def _player_stats() -> dict:
    return {
        "touches": 0, "passes": 0, "passes_completed": 0, "difficulty_completed_sum": 0.0, "hard_passes": 0,
        "progressive_passes": 0, "key_passes": 0, "assists": 0, "shots": 0, "shots_on_target": 0, "xg": 0.0,
        "goals": 0, "tackles_won": 0, "interceptions": 0, "pressures": 0, "recoveries": 0, "dribbles": 0,
        "dribbles_completed": 0, "fouls": 0, "xt": 0.0, "yellow_cards": 0, "red_cards": 0,
    }


class LiveAnalytics:
    def __init__(self, metadata: dict, pass_model: PassDifficultyModel | None = None):
        self.ctx = MatchContext(metadata)
        self.signals = SignalFactory()
        self.model = pass_model or PassDifficultyModel.load()
        self.physical = PhysicalTracker(self.ctx, self.signals)
        self.tactical = TacticalTracker(self.ctx, self.signals)
        self.frame = None
        self._xy: dict[int, np.ndarray] = {}
        self._order: deque[int] = deque()
        self.score = {self.ctx.home: 0, self.ctx.away: 0}
        self.players: dict[str, dict] = defaultdict(_player_stats)
        self.pass_info: dict[str, dict] = {}
        self.shots: dict[str, dict] = {}
        self.last_completed: dict[str, dict] = {}
        self.possession_start: tuple[int, float, str | None] = (0, 0.0, None)
        self.possession_passes = 0
        self.emitted: list[Signal] = []
        self.periods_done: list[int] = []

    # ------------------------------------------------------------ ingestion

    def on_frame(self, frame) -> list[Signal]:
        self.frame = frame
        self._xy[frame.index] = frame.xy
        self._order.append(frame.index)
        while len(self._order) > FRAME_MEMORY:
            self._xy.pop(self._order.popleft(), None)
        out = self.physical.update(frame) + self.tactical.on_frame(frame)
        self.emitted += out
        return out

    def on_event(self, ev: dict) -> list[Signal]:
        self.tactical.on_event(ev)
        handler = getattr(self, f"_on_{ev['type']}", None)
        out = handler(ev) or [] if handler else []
        self.emitted += out
        return out

    # -------------------------------------------------------------- helpers

    def _at(self, ev: dict) -> dict:
        return {"frame": ev["frame"], "t": ev["t"], "period": ev["period"], "clock": clock_label(ev["period"], ev["t"])}

    def _rel(self, point: dict, team: str, period: int) -> np.ndarray:
        return self.ctx.to_rel([point["x"], point["y"]], team, period)

    def _touch(self, ev: dict) -> dict:
        st = self.players[ev["player"]]
        st["touches"] += 1
        return st

    # --------------------------------------------------------------- events

    def _on_possession_change(self, ev):
        self.possession_start = (ev["possession"], ev["t"], ev["team"])
        self.possession_passes = 0

    def _on_pass(self, ev):
        st = self._touch(ev)
        st["passes"] += 1
        team, period = ev["team"], ev["period"]
        xy = self._xy.get(ev["frame"])
        difficulty, reasons = None, []
        if xy is not None:
            feats = pass_features(ev, xy, self.ctx)
            difficulty = self.model.difficulty(feats)
            reasons = self.model.reasons(feats)
        rel_s = self._rel(ev["location"], team, period)
        rel_e = self._rel(ev["end_location"], team, period)
        forward = float(rel_e[0] - rel_s[0])
        info = {"difficulty": difficulty, "reasons": reasons, "forward_m": round(forward, 1)}
        self.pass_info[ev["id"]] = info
        if ev["outcome"] != "complete":
            return []
        self.possession_passes += 1
        st["passes_completed"] += 1
        st["xt"] += max(0.0, expected_threat(*rel_e) - expected_threat(*rel_s))
        if difficulty is not None:
            st["difficulty_completed_sum"] += difficulty
        progressive = forward >= 10.0 and rel_e[0] > 52.5
        st["progressive_passes"] += int(progressive)
        self.last_completed[team] = ev
        out = []
        if difficulty is not None and difficulty >= HARD_PASS and (forward >= 8.0 or rel_e[0] > 70.0):
            st["hard_passes"] += 1
            out.append(self.signals.make(
                "hard_pass", team=team, players=[ev["player"], ev["attributes"]["recipient"]], evidence=[ev["id"]],
                facts={"difficulty": difficulty, "completion_chance_pct": 100 - difficulty, "reasons": reasons,
                       "length_m": ev["attributes"]["length_m"], "forward_m": round(forward, 1),
                       "ball_speed_kmh": round(ev["attributes"]["speed_ms"] * 3.6, 1),
                       "kind": ev["attributes"].get("kind"),
                       "start_xy": [ev["location"]["x"], ev["location"]["y"]],
                       "end_xy": [ev["end_location"]["x"], ev["end_location"]["y"]]},
                significance=0.3 + difficulty / 300, **self._at(ev)))
        done = st["passes_completed"]
        if done in PASS_MILESTONES:
            accuracy = round(100.0 * done / st["passes"], 1)
            if accuracy >= 85.0:
                out.append(self.signals.make(
                    "pass_milestone", team=team, players=[ev["player"]], evidence=[ev["id"]],
                    facts={"completed_passes": done, "accuracy_pct": accuracy,
                           "progressive_passes": st["progressive_passes"]},
                    significance=0.25, **self._at(ev)))
        return out

    def _on_carry(self, ev):
        st = self._touch(ev)
        rel_s = self._rel(ev["location"], ev["team"], ev["period"])
        rel_e = self._rel(ev["end_location"], ev["team"], ev["period"])
        st["xt"] += max(0.0, expected_threat(*rel_e) - expected_threat(*rel_s))

    def _on_shot(self, ev):
        st = self._touch(ev)
        a = ev["attributes"]
        xg = float(a.get("xg", 0.0))
        st["shots"] += 1
        st["xg"] += xg
        st["shots_on_target"] += int(ev["outcome"] in ("goal", "saved"))
        self.shots[ev["id"]] = ev
        out = []
        lp = self.last_completed.get(ev["team"])
        if lp is not None and lp["attributes"].get("recipient") == ev["player"] and lp["possession"] == ev["possession"]:
            self.players[lp["player"]]["key_passes"] += 1
            info = self.pass_info.get(lp["id"], {})
            out.append(self.signals.make(
                "key_pass", team=ev["team"], players=[lp["player"], ev["player"]], evidence=[lp["id"], ev["id"]],
                facts={"shot_xg": round(xg, 3), "pass_difficulty": info.get("difficulty"),
                       "pass_length_m": lp["attributes"]["length_m"], "pass_kind": lp["attributes"].get("kind"),
                       "shot_outcome": ev["outcome"]},
                significance=0.3 + xg, **self._at(ev)))
        speed = float(a.get("speed_ms", 0.0))
        if speed >= SHOT_SPEED_TAG and a.get("body_part") != "head":
            out.append(self.signals.make(
                "shot_speed", team=ev["team"], players=[ev["player"]], evidence=[ev["id"]],
                facts={"speed_kmh": round(speed * 3.6, 1), "distance_m": a.get("distance_m"), "outcome": ev["outcome"],
                       "shot_xy": [ev["location"]["x"], ev["location"]["y"]]},
                significance=0.3, **self._at(ev)))
        if ev["outcome"] != "goal" and xg >= BIG_CHANCE_XG:
            out.append(self.signals.make(
                "big_chance", team=ev["team"], players=[ev["player"]] + ([a["goalkeeper"]] if a.get("goalkeeper") else []),
                evidence=[ev["id"]],
                facts={"xg": round(xg, 3), "outcome": ev["outcome"], "distance_m": a.get("distance_m"),
                       "body_part": a.get("body_part"), "set_piece": a.get("set_piece"),
                       "score": dict(self.score), "shot_xy": [ev["location"]["x"], ev["location"]["y"]]},
                significance=0.5 + xg * 0.45, **self._at(ev)))
        return out

    def _on_goal(self, ev):
        a = ev["attributes"]
        team = ev["team"]
        before = dict(self.score)
        self.score = {self.ctx.home: a["score_home"], self.ctx.away: a["score_away"]}
        scorer = a["scorer"]
        self.players[scorer]["goals"] += 1
        if a.get("assist"):
            self.players[a["assist"]]["assists"] += 1
        shot = self.shots.get(a.get("shot_event"), {})
        sa = shot.get("attributes", {})
        mine, theirs = self.score[team], self.score[self.ctx.other(team)]
        prev_mine, prev_theirs = before[team], before[self.ctx.other(team)]
        if mine == theirs:
            situation = "equaliser"
        elif prev_mine == prev_theirs:
            situation = "takes the lead"
        elif prev_mine < prev_theirs:
            situation = "pulls one back"
        else:
            situation = "extends the lead"
        momentum = self.tactical.momentum_value()
        against_run = (momentum <= -30 and team == self.ctx.home) or (momentum >= 30 and team == self.ctx.away)
        build_up_s = round(ev["t"] - self.possession_start[1], 1) if self.possession_start[2] == team else None
        late = ev["minute"] >= 84
        facts = {
            "scorer": scorer, "assist": a.get("assist"), "score": dict(self.score), "situation": situation,
            "xg": sa.get("xg"), "shot_speed_kmh": round(sa["speed_ms"] * 3.6, 1) if sa.get("speed_ms") else None,
            "distance_m": sa.get("distance_m"), "body_part": sa.get("body_part"), "set_piece": sa.get("set_piece"),
            "passes_in_move": self.possession_passes, "move_duration_s": build_up_s,
            "momentum_before": momentum, "against_run_of_play": against_run, "late": late,
            "scorer_goals_this_match": self.players[scorer]["goals"],
            "shot_xy": [shot["location"]["x"], shot["location"]["y"]] if shot else None,
        }
        evidence = [ev["id"]] + ([shot["id"]] if shot else [])
        return [self.signals.make("goal", team=team, players=[p for p in (scorer, a.get("assist")) if p],
                                  facts=facts, evidence=evidence,
                                  significance=0.9 + 0.05 * late + 0.05 * (situation in ("equaliser", "takes the lead")),
                                  **self._at(ev))]

    def _on_goalkeeper(self, ev):
        a = ev["attributes"]
        if a.get("action") != "save":
            return []
        shot = self.shots.get(a.get("shot_event"))
        if shot is None or float(shot["attributes"].get("xg", 0)) < 0.12:
            return []
        sa = shot["attributes"]
        return [self.signals.make(
            "save", team=ev["team"], players=[ev["player"], shot["player"]], evidence=[shot["id"], ev["id"]],
            facts={"xg_faced": round(float(sa["xg"]), 3), "shot_speed_kmh": round(sa["speed_ms"] * 3.6, 1),
                   "distance_m": sa.get("distance_m"), "held": ev["outcome"] == "caught", "score": dict(self.score)},
            significance=0.45 + float(sa["xg"]) * 0.5, **self._at(ev))]

    def _on_tackle(self, ev):
        if ev["outcome"] == "won":
            self.players[ev["player"]]["tackles_won"] += 1

    def _on_interception(self, ev):
        self._touch(ev)["interceptions"] += 1

    def _on_pressure(self, ev):
        self.players[ev["player"]]["pressures"] += 1

    def _on_ball_recovery(self, ev):
        self._touch(ev)["recoveries"] += 1

    def _on_dribble(self, ev):
        st = self._touch(ev)
        st["dribbles"] += 1
        st["dribbles_completed"] += int(ev["outcome"] == "complete")

    def _on_foul(self, ev):
        self.players[ev["player"]]["fouls"] += 1

    def _on_card(self, ev):
        card = ev["attributes"]["card"]
        st = self.players[ev["player"]]
        st["yellow_cards" if card == "yellow" else "red_cards"] += 1
        return [self.signals.make("card", team=ev["team"], players=[ev["player"]], evidence=[ev["id"]],
                                  facts={"card": card, "score": dict(self.score)},
                                  significance=0.6 if card != "yellow" else 0.3, **self._at(ev))]

    def _on_substitution(self, ev):
        a = ev["attributes"]
        off = a["player_off"]
        phys = self.physical.stats.get(off, {})
        return [self.signals.make(
            "substitution", team=ev["team"], players=[a["player_on"], off], evidence=[ev["id"]],
            facts={"player_on": a["player_on"], "player_off": a["player_off"],
                   "off_distance_km": round(phys.get("distance_m", 0) / 1000, 2),
                   "off_touches": self.players[off]["touches"], "score": dict(self.score)},
            significance=0.25, **self._at(ev))]

    def _on_added_time(self, ev):
        return [self.signals.make("added_time", facts={"minutes": ev["attributes"]["minutes"], "score": dict(self.score)},
                                  evidence=[ev["id"]], significance=0.3, **self._at(ev))]

    def _on_period_end(self, ev):
        period = ev["period"]
        self.periods_done.append(period)
        n = sum(1 for b in self.tactical.buckets if b.period == period)
        w = self.tactical.window(n)
        best = {}
        for tid in self.ctx.team_ids:
            mine = [(pid, st) for pid, st in self.players.items() if self.ctx.player_team(pid) == tid]
            if mine:
                pid, st = max(mine, key=lambda kv: kv[1]["xt"] + 0.3 * kv[1]["key_passes"] + kv[1]["goals"])
                best[tid] = {"player": pid, "xt": round(st["xt"], 3), "key_passes": st["key_passes"], "goals": st["goals"]}
        return [self.signals.make(
            "period_summary", facts={"period": period, "score": dict(self.score), "teams": w.get("teams", {}),
                                     "match": w.get("match", {}), "standouts": best,
                                     "match_teams": self.tactical.window(len(self.tactical.buckets)).get("teams", {})},
            evidence=[ev["id"]], significance=0.7, **self._at(ev))]

    # ------------------------------------------------------------- snapshot

    def ball_carrier(self) -> dict | None:
        f = self.frame
        if f is None or f.owner < 0:
            return None
        pid = self.ctx.by_index[int(f.player_index[f.owner])]["id"]
        return {"player": pid, "name": self.ctx.player_name(pid), "team": self.ctx.player_team(pid),
                "speed_kmh": self.physical.current_speed_kmh(f.owner)}

    def player_table(self) -> dict[str, dict]:
        table = {}
        for pid in set(self.players) | set(self.physical.stats):
            st = dict(self.players[pid])
            phys = self.physical.stats.get(pid, {})
            completed = st["passes_completed"]
            st.update({
                "name": self.ctx.player_name(pid, short=False), "team": self.ctx.player_team(pid),
                "pass_accuracy_pct": round(100.0 * completed / st["passes"], 1) if st["passes"] else None,
                "avg_difficulty_completed": round(st["difficulty_completed_sum"] / completed, 1) if completed else None,
                "xg": round(st["xg"], 2), "xt": round(st["xt"], 3),
                "distance_km": round(phys.get("distance_m", 0.0) / 1000, 2),
                "top_speed_kmh": round(phys.get("top_speed_ms", 0.0) * 3.6, 1),
                "sprints": phys.get("sprints", 0),
                "minutes": round(phys.get("frames", 0) / 600, 1),
            })
            st.pop("difficulty_completed_sum", None)
            table[pid] = st
        return table

    def snapshot(self, window_minutes: int = 10) -> dict:
        f = self.frame
        return {
            "clock": clock_label(f.period, f.t) if f else "0'",
            "period": f.period if f else 0,
            "score": dict(self.score),
            "momentum": self.tactical.momentum_value(),
            "recent": self.tactical.window(window_minutes),
            "match_so_far": self.tactical.window(len(self.tactical.buckets)),
            "ball_carrier": self.ball_carrier(),
        }


def analyse(match, latency: float = 2.0) -> LiveAnalytics:
    """Run the full live pipeline over a stored match, as fast as possible."""
    from ..live.feed import replay

    analytics = LiveAnalytics(match.metadata)
    for kind, item in replay(match, latency=latency):
        if kind == "frame":
            analytics.on_frame(item)
        else:
            analytics.on_event(item)
    return analytics
