"""Tracking-grade football match simulator.

Every 0.1 s the engine:

1. works out where each player wants to be: a team shape that shifts with the
   ball, pressing and covering, marking, support angles and runs in behind;
2. advances the ball's state machine: at a player's feet, in flight, loose,
   or dead for a restart;
3. moves every player towards their target within their own top speed and
   acceleration, so distances, sprints and speeds are physically plausible;
4. records a tracking frame for all 22 players and the ball.

On-ball decisions use expected value. Each candidate pass is scored with a
geometric success model (the passer's accuracy, interception risk along the
lane, and who reaches the ball first) plus its expected-threat gain. Players
choose with a softmax whose temperature reflects their skill, so better
players make better choices but nobody is a robot. Events are written as
things happen, the way a live data feed would produce them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import __version__
from . import pitch
from .models import (
    expected_goals,
    expected_threat,
    logistic,
    pass_accuracy,
    pass_kinematics,
    rating_to_acceleration,
    rating_to_top_speed,
)
from .squads import FORMATIONS, ROLE_GROUP, PlayerSpec, TeamSpec, TeamStyle
from .storyline import STORYLINES, MatchOutline, Storyline

FPS = 10
DT = 1.0 / FPS
HALF = 45 * 60

OWNED, FLIGHT, LOOSE, DEAD = "owned", "flight", "loose", "dead"
STATE_CODE = {OWNED: 0, FLIGHT: 1, LOOSE: 2, DEAD: 3}

RUN_RATE = {"ST": 0.12, "RW": 0.09, "LW": 0.09, "RAM": 0.08, "LAM": 0.08, "CAM": 0.05, "RCM": 0.03, "LCM": 0.03}
CORNER_TAKERS = {"RW", "LW", "RAM", "LAM", "CAM", "RCM", "LCM", "RB", "LB"}
ON_BALL = {"pass", "carry", "dribble", "shot", "clearance"}
# How much players value moving the ball into dangerous areas relative to simply keeping it.
XT_WEIGHT = 1.6
REFEREES = ("Daniel Okafor", "Helen Marsh-Pryor", "Ivo Kranjc", "Simone Achterberg", "Rui Calado")

# Set-piece positions in the attacking team's relative frame.
CORNER_ATTACK = [(99.0, 31.0), (99.0, 37.5), (97.0, 34.0), (94.0, 28.0), (94.0, 41.0),
                 (88.5, 34.0), (76.0, 34.0), (68.0, 20.0), (68.0, 48.0)]
CORNER_DEFEND = [(101.5, 31.5), (101.5, 36.5), (99.5, 34.0), (97.5, 30.0), (97.5, 38.5),
                 (95.0, 34.0), (93.0, 27.0), (93.0, 41.0), (87.0, 34.0), (80.0, 24.0)]


@dataclass
class Spell:
    """One player's uninterrupted time on the ball."""

    slot: int
    start_frame: int
    start_pos: np.ndarray
    decision_frame: int
    tackle_ready: int
    pressers: set[int] = field(default_factory=set)
    carries: int = 0


@dataclass
class Flight:
    kind: str  # pass | clearance | shot
    event: dict
    team: int
    passer: int
    origin: np.ndarray
    target: np.ndarray
    start_frame: int
    duration: float
    v0: float
    v_end: float
    lofted: bool
    apex: float = 0.0
    z_end: float = 0.0
    receiver: int | None = None
    intercept: tuple[int, float, np.ndarray] | None = None
    shot_outcome: str | None = None
    blocker: int | None = None

    def ball_at(self, tau: float) -> np.ndarray:
        vec = self.target - self.origin
        dist = float(np.hypot(*vec))
        u = vec / dist if dist > 1e-6 else np.zeros(2)
        t = min(tau, self.duration)
        frac = t / self.duration if self.duration > 0 else 1.0
        if self.lofted or self.kind == "shot":
            s = dist * frac
        else:
            s = self.v0 * t - 0.5 * (self.v0 - self.v_end) / self.duration * t * t
        z = 4.0 * self.apex * frac * (1.0 - frac) if self.lofted else 0.0
        z += self.z_end * frac
        xy = self.origin + u * s
        return np.array([xy[0], xy[1], max(z, 0.0)])


@dataclass
class Restart:
    kind: str  # kick_off | throw_in | corner | goal_kick | free_kick | penalty
    team: int
    spot: np.ndarray
    taker: int | None
    resume_frame: int


@dataclass
class PendingPass:
    """A pass whose outcome is decided by whoever controls the ball next."""

    event: dict
    team: int
    deadline: int


@dataclass
class PassPlan:
    kind: str
    receiver: int | None
    target: np.ndarray
    distance: float
    lofted: bool
    v0: float
    v_end: float
    duration: float
    apex: float
    p_accuracy: float
    p_lane: float
    p_control: float
    interceptor: int | None
    intercept_time: float
    intercept_point: np.ndarray | None

    @property
    def p_success(self) -> float:
        return self.p_accuracy * self.p_lane * self.p_control

    @property
    def loss_point(self) -> np.ndarray:
        if self.intercept_point is not None and self.p_lane < self.p_control:
            return self.intercept_point
        return self.target


@dataclass
class SimulationResult:
    metadata: dict
    events: list[dict]
    truth: list[dict]
    tracking: dict[str, np.ndarray]

    def outline(self) -> MatchOutline:
        m = self.metadata
        goals = tuple((e["minute"], 0 if e["team"] == m["teams"]["home"]["id"] else 1)
                      for e in self.events if e["type"] == "goal")
        return MatchOutline(
            final_score=(m["score"]["home"], m["score"]["away"]),
            half_time_score=(m["half_time_score"]["home"], m["half_time_score"]["away"]),
            goals=goals,
        )


def _pt(xy) -> dict | None:
    if xy is None:
        return None
    return {"x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2)}


def _clean(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    return v


class MatchSimulator:
    def __init__(
        self,
        home: TeamSpec,
        away: TeamSpec,
        *,
        seed: int = 0,
        storyline: Storyline | None = None,
        referee: str | None = None,
        kickoff: str = "2026-10-24T15:00:00Z",
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.teams = (home, away)
        self.storyline = storyline or STORYLINES["balanced"]
        self.referee = referee or REFEREES[seed % len(REFEREES)]
        self.kickoff = kickoff

        self.registry: list[PlayerSpec] = [*home.all_players(), *away.all_players()]
        self.reg_index = {p.id: i for i, p in enumerate(self.registry)}
        self.bench = [list(home.bench), list(away.bench)]

        self.slot_player: list[PlayerSpec] = [*home.starters, *away.starters]
        self.slot_role: list[str] = [*home.roles, *away.roles]
        self.slot_reg = np.array([self.reg_index[p.id] for p in self.slot_player], dtype=np.int16)
        self.team_of = np.repeat([0, 1], 11)
        self.active = np.ones(22, dtype=bool)
        self.anchors = np.zeros((22, 2, 2))
        self.vmax = np.zeros(22)
        self.accel = np.zeros(22)
        self.stamina = np.zeros(22)
        for s in range(22):
            self._load_slot(s)

        self.pos = np.zeros((22, 2))
        self.vel = np.zeros((22, 2))
        self.smooth = np.zeros((22, 2))
        self.target = np.zeros((22, 2))
        self.urg = np.full(22, 0.5)
        self.covered = np.zeros(22)
        self.stun_until = np.zeros(22, dtype=np.int64)
        self.run_until = np.zeros(22, dtype=np.int64)
        self.run_target = np.zeros((22, 2))
        self.presser: list[int | None] = [None, None]

        self.direction = [1, -1]
        self.style: list[TeamStyle] = [home.style, away.style]
        self.score = [0, 0]
        self.half_time_score = (0, 0)
        self.goals: list[tuple[int, int]] = []
        self.subs_made = [0, 0]
        self.sub_windows = [self._plan_sub_windows(), self._plan_sub_windows()]
        self.cards: dict[str, int] = {}

        self.ball = np.array([pitch.LENGTH / 2, pitch.WIDTH / 2, 0.0])
        self.state = DEAD
        self.owner: int | None = None
        self.flight: Flight | None = None
        self.loose_v = np.zeros(2)
        self.restart: Restart | None = None
        self.spell: Spell | None = None
        self.pending: PendingPass | None = None
        self.last_pass: dict | None = None
        self.last_touch_team = 0
        self.possession_team = 0
        self.possession_id = 0

        self.events: list[dict] = []
        self.truth: list[dict] = []
        self._n_events = 0

        cap = 2 * (HALF + 12 * 60) * FPS
        self.f_t = np.zeros(cap, np.float32)
        self.f_period = np.zeros(cap, np.int8)
        self.f_state = np.zeros(cap, np.int8)
        self.f_owner = np.full(cap, -1, np.int16)
        self.f_ball = np.zeros((cap, 3), np.float32)
        self.f_xy = np.zeros((cap, 22, 2), np.float32)
        self.f_ids = np.zeros((cap, 22), np.int16)
        self.n = 0

        self.frame = 0
        self.period = 0
        self.clock = 0.0
        self.period_end = float(HALF)
        self._minute_seen = -1

    # ------------------------------------------------------------------ run

    def run(self) -> SimulationResult:
        for period in (1, 2):
            self._start_period(period)
            announced = False
            while True:
                if not announced and self.clock >= HALF:
                    self._announce_added_time()
                    announced = True
                penalty_due = self.state == DEAD and self.restart is not None and self.restart.kind == "penalty"
                if announced and self.clock >= self.period_end and self.state != FLIGHT and not penalty_due:
                    break
                self._tick()
            self._end_period()
        return self._result()

    def _tick(self) -> None:
        self._apply_storyline()
        self._compute_targets()
        if self.state == OWNED:
            self._update_owned()
        elif self.state == FLIGHT:
            self._update_flight()
        elif self.state == LOOSE:
            self._update_loose()
        else:
            self._update_dead()
        self._integrate()
        if self.state == OWNED:
            self._attach_ball()
        self._record()
        self.frame += 1
        self.clock += DT

    # -------------------------------------------------------------- helpers

    @property
    def match_time(self) -> float:
        return (self.period - 1) * HALF + self.clock

    def _load_slot(self, s: int) -> None:
        r = self.slot_player[s].ratings
        self.vmax[s] = rating_to_top_speed(r.pace)
        self.accel[s] = rating_to_acceleration(r.accel)
        self.stamina[s] = r.stamina
        team = self.teams[self.team_of[s]]
        self.anchors[s] = np.array(FORMATIONS[team.formation][self.slot_role[s]], dtype=float)

    def _plan_sub_windows(self) -> list[tuple[float, int]]:
        return [(self.rng.uniform(56, 66), 2), (self.rng.uniform(68, 77), 2), (self.rng.uniform(79, 86), 1)]

    def _r(self, s: int, attr: str) -> float:
        value = float(getattr(self.slot_player[s].ratings, attr))
        if attr in ("passing", "dribbling", "finishing", "defending", "keeping", "aerial"):
            value += self.style[self.team_of[s]].strength
        return value

    def _pid(self, s: int) -> str:
        return self.slot_player[s].id

    def _slots(self, team: int) -> np.ndarray:
        return np.flatnonzero((self.team_of == team) & self.active)

    def _gk(self, team: int) -> int | None:
        for s in self._slots(team):
            if self.slot_role[s] == "GK":
                return int(s)
        return None

    def _nearest_opp(self, s: int) -> tuple[int | None, float]:
        opp = self._slots(1 - self.team_of[s])
        if len(opp) == 0:
            return None, math.inf
        d = np.hypot(*(self.pos[opp] - self.pos[s]).T)
        i = int(np.argmin(d))
        return int(opp[i]), float(d[i])

    def _reach_time(self, slots, point, reaction: float = 0.25, reach: float = 1.0) -> np.ndarray:
        """Seconds for each player to reach a point, allowing for the way they are already moving.

        A player already running towards the point gains up to 0.35 s; one
        running the other way loses up to 0.35 s turning round.
        """
        slots = np.asarray(slots)
        vec = point - self.pos[slots]
        d = np.hypot(vec[:, 0], vec[:, 1])
        toward = np.einsum("ij,ij->i", self.vel[slots], vec) / np.maximum(d, 1e-6)
        momentum = 0.35 * np.clip(toward / self.vmax[slots], -1.0, 1.0)
        t = reaction + np.maximum(d - reach, 0.0) / (0.92 * self.vmax[slots]) - momentum
        return np.maximum(t, 0.05)

    def _event(self, etype: str, slot: int | None = None, *, team: int | None = None, location=None,
               end_location=None, outcome: str | None = None, frame: int | None = None, **attrs) -> dict:
        frame = self.frame if frame is None else frame
        t = float(self.f_t[frame]) if frame < self.n else self.match_time
        player = None
        pressured = False
        if slot is not None:
            team = int(self.team_of[slot])
            player = self._pid(slot)
            if etype in ON_BALL:
                pressured = self._nearest_opp(slot)[1] < 3.0
        ev = {
            "id": f"ev{self._n_events:05d}",
            "index": -1,
            "frame": int(frame),
            "period": self.period,
            "t": round(t, 1),
            "minute": int(t // 60),
            "second": int(t % 60),
            "type": etype,
            "team": self.teams[team].id if team is not None else None,
            "player": player,
            "possession": self.possession_id,
            "location": _pt(location),
            "end_location": _pt(end_location),
            "outcome": outcome,
            "under_pressure": pressured,
            "attributes": {k: _clean(v) for k, v in attrs.items() if v is not None},
            "related": [],
        }
        self._n_events += 1
        self.events.append(ev)
        return ev

    def _set_possession(self, team: int, cause: str) -> None:
        if team == self.possession_team:
            return
        previous = self.possession_team
        self.possession_team = team
        self.possession_id += 1
        self.last_pass = None
        self._event("possession_change", team=team, from_team=self.teams[previous].id, cause=cause)

    # -------------------------------------------------------- periods, time

    def _apply_storyline(self) -> None:
        minute = int(self.match_time // 60)
        if minute == self._minute_seen:
            return
        self._minute_seen = minute
        for team in (0, 1):
            self.style[team] = self.teams[team].style.with_overrides(self.storyline.styles_at(minute, team))

    def _start_period(self, period: int) -> None:
        self.period = period
        self.clock = 0.0
        self.period_end = float(HALF)
        if period == 2:
            self.direction = [-1, 1]
        kick_team = 0 if period == 1 else 1
        for team in (0, 1):
            d = self.direction[team]
            for s in self._slots(team):
                ax, ay = self.anchors[s, 1 if team == kick_team else 0]
                rel = np.array([min(ax, 49.5 if team == kick_team else 43.0), ay])
                self.pos[s] = pitch.to_rel(rel, d)
        self.vel[:] = 0.0
        self.smooth[:] = self.pos
        self.ball[:] = (pitch.LENGTH / 2, pitch.WIDTH / 2, 0.0)
        self._apply_storyline()
        self._event("period_start")
        self._dead("kick_off", kick_team, pitch.CENTER.copy(), 1.5)

    def _announce_added_time(self) -> None:
        second_half = self.period == 2
        goals = sum(1 for minute, _ in self.goals if (minute >= 45) == second_half)
        base = 1.5 + 0.5 * goals + (1.5 + 0.35 * sum(self.subs_made) if second_half else 0.0)
        minutes = int(np.clip(round(base + self.rng.normal(0, 0.8)), 1, 8))
        self._event("added_time", minutes=minutes)
        self.period_end = HALF + minutes * 60 + self.rng.uniform(0, 35)

    def _end_period(self) -> None:
        if self.state == OWNED and self.owner is not None:
            self._end_spell(self.owner)
        if self.pending is not None:
            self.pending.event["outcome"] = "incomplete"
            self.pending = None
        self._event("period_end", frame=self.frame - 1, score_home=self.score[0], score_away=self.score[1])
        if self.period == 1:
            self.half_time_score = (self.score[0], self.score[1])

    # -------------------------------------------------------------- targets

    def _attacking_team(self) -> int:
        if self.state == OWNED:
            return int(self.team_of[self.owner])
        if self.state == FLIGHT:
            return self.flight.team
        if self.state == DEAD and self.restart is not None:
            return self.restart.team
        return self.possession_team

    def _offside_line(self, team: int) -> float:
        """x of the offside line in `team`'s attacking frame."""
        d = self.direction[team]
        opp = self._slots(1 - team)
        xs = np.sort(pitch.to_rel(self.pos[opp], d)[:, 0])[::-1]
        second_last = xs[1] if len(xs) > 1 else pitch.HALFWAY
        ball_x = pitch.to_rel(self.ball[:2], d)[0]
        return float(max(pitch.HALFWAY, ball_x, second_last))

    def _compute_targets(self) -> None:
        attacking = self._attacking_team()
        ball = self.ball[:2]
        base = self.smooth.copy()
        for team in (0, 1):
            slots = self._slots(team)
            d = self.direction[team]
            style = self.style[team]
            in_poss = attacking == team
            rb = pitch.to_rel(ball, d)
            anchor = self.anchors[slots, 1 if in_poss else 0]
            x = anchor[:, 0].copy()
            y = anchor[:, 1].copy()
            groups = np.array([ROLE_GROUP[self.slot_role[s]] for s in slots])
            gk = groups == "GK"
            y = 34.0 + (y - 34.0) * (style.width if in_poss else 0.82)
            if in_poss:
                x += 0.45 * (rb[0] - 52.5)
                y += 0.15 * (rb[1] - 34.0)
                x = np.minimum(x, self._offside_line(team) - 0.7)
            else:
                x += 0.55 * (rb[0] - 52.5) + style.line_height
                y += 0.35 * (rb[1] - 34.0)
                back = (groups == "DEF") | (groups == "MID")
                x = np.where(back, np.minimum(x, rb[0] - 1.5), x)
                x = np.where(groups == "DEF", np.maximum(x, 6.0), x)
                x, y = self._apply_marking(team, slots, x, y, groups, style)
                x, y = self._goal_side(rb, x, y, groups)
            x[gk] = np.clip(3.5 + 0.12 * rb[0], 3.0, 16.0)
            y[gk] = 34.0 + 0.22 * (rb[1] - 34.0)
            x = np.where(gk, x, np.maximum(x, 4.0))
            rel = np.stack([np.clip(x, 1.0, 104.0), np.clip(y, 1.0, 67.0)], axis=1)
            base[slots] = pitch.to_rel(rel, d)
        self.smooth += 0.2 * (base - self.smooth)
        self.target[:] = self.smooth
        gap = np.hypot(*(self.target - self.pos).T)
        self.urg[:] = 0.36 + 0.5 * np.clip((gap - 5.0) / 15.0, 0.0, 1.0)
        defending = self.team_of != attacking
        self.urg[defending] += 0.04
        self._situational_targets()

    def _apply_marking(self, team, slots, x, y, groups, style):
        """Out of possession, pull defenders and midfielders goal-side of nearby opponents."""
        d = self.direction[team]
        opp = self._slots(1 - team)
        opp_rel = pitch.to_rel(self.pos[opp], d)
        tgt = np.stack([x, y], axis=1)
        dist = np.linalg.norm(tgt[:, None, :] - opp_rel[None, :, :], axis=2)
        j = dist.argmin(axis=1)
        dmin = dist[np.arange(len(slots)), j]
        weight = 0.15 + 0.2 * style.press_intensity
        w = np.where(groups == "ATT", 0.4 * weight, weight)
        w = np.where((groups != "GK") & (dmin < 9.0), w, 0.0)
        mark = opp_rel[j] - np.array([1.5, 0.0])
        return (1 - w) * x + w * mark[:, 0], (1 - w) * y + w * mark[:, 1]

    @staticmethod
    def _goal_side(rb, x, y, groups):
        """Near their own goal, defenders and midfielders squeeze into the ball-goal corridor."""
        dist_goal = float(np.hypot(rb[0], rb[1] - 34.0))
        w = 0.55 * float(np.clip((36.0 - dist_goal) / 16.0, 0.0, 1.0))
        if w == 0.0:
            return x, y
        goal = np.array([0.0, 34.0])
        seg = np.asarray(rb, dtype=float) - goal
        pts = np.stack([x, y], axis=1)
        t = np.clip(((pts - goal) @ seg) / float(seg @ seg), 0.05, 0.9)
        corridor = goal + t[:, None] * seg
        ww = np.where((groups == "DEF") | (groups == "MID"), w, 0.0)
        return (1 - ww) * x + ww * corridor[:, 0], (1 - ww) * y + ww * corridor[:, 1]

    def _situational_targets(self) -> None:
        if self.state == OWNED:
            c = self.owner
            team = int(self.team_of[c])
            self._press_targets(c, 1 - team)
            self._support_targets(c, team)
            self._run_targets(c, team)
            self._carrier_target(c)
        elif self.state == FLIGHT:
            f = self.flight
            if f.receiver is not None and self.active[f.receiver]:
                self.target[f.receiver] = f.target
                self.urg[f.receiver] = 1.0
            if f.intercept is not None:
                j, _, q = f.intercept
                self.target[j] = q
                self.urg[j] = 1.0
            opp = self._slots(1 - f.team)
            if f.kind == "shot":
                gk = self._gk(1 - f.team)
                if gk is not None:
                    self.target[gk] = f.target
                    self.urg[gk] = 1.0
            elif len(opp):
                j = int(opp[np.argmin(self._reach_time(opp, f.target))])
                if f.intercept is None or j != f.intercept[0]:
                    self.target[j] = f.target
                    self.urg[j] = 0.9
            for s in np.flatnonzero(self.run_until > self.frame):
                if s != f.receiver:
                    self.target[s] = self.run_target[s]
                    self.urg[s] = 0.9
        elif self.state == LOOSE:
            aim = pitch.clip(self.ball[:2] + self.loose_v * 0.4, 0.5)
            for team in (0, 1):
                sl = self._slots(team)
                order = np.argsort(self._reach_time(sl, aim))
                for k, idx in enumerate(order[:2]):
                    self.target[sl[idx]] = aim
                    self.urg[sl[idx]] = 1.0 if k == 0 else 0.75
        elif self.state == DEAD and self.restart is not None:
            self._restart_targets(self.restart)

    def _press_targets(self, c: int, dteam: int) -> None:
        style = self.style[dteam]
        d = self.direction[dteam]
        opp = np.array([s for s in self._slots(dteam) if self.slot_role[s] != "GK"])
        if len(opp) == 0:
            return
        carrier = self.pos[c]
        tt = self._reach_time(opp, carrier)
        order = np.argsort(tt)
        first = int(opp[order[0]])
        prev = self.presser[dteam]
        if prev is not None and prev != first and prev in opp:
            k = int(np.flatnonzero(opp == prev)[0])
            if tt[k] - tt[order[0]] < 0.5:
                first = prev
        self.presser[dteam] = first
        goal_side = pitch.unit(pitch.own_goal_center(d) - carrier)
        ball_rel = pitch.to_rel(carrier, d)[0]
        if ball_rel <= style.press_line or ball_rel < 35.0:
            self.target[first] = carrier + goal_side * 0.8
            self.urg[first] = 0.85 + 0.15 * style.press_intensity
            near_goal = float(np.hypot(*(carrier - pitch.own_goal_center(d)))) < 25.0
            if near_goal and len(order) > 1:
                second = int(opp[order[1]])
                self.target[second] = carrier + goal_side * 1.5
                self.urg[second] = 0.95
            elif style.press_intensity >= 0.6 and len(order) > 1:
                second = int(opp[order[1]])
                mates = np.array([m for m in self._slots(self.team_of[c]) if m != c])
                if len(mates):
                    m = mates[np.argmin(np.hypot(*(self.pos[mates] - carrier).T))]
                    self.target[second] = carrier + 0.45 * (self.pos[m] - carrier)
                    self.urg[second] = 0.8
        else:
            self.target[first] = carrier + goal_side * 7.0
            self.urg[first] = 0.6

    def _support_targets(self, c: int, team: int) -> None:
        mates = [int(m) for m in self._slots(team)
                 if m != c and self.slot_role[m] != "GK" and self.run_until[m] <= self.frame]
        if not mates:
            return
        carrier = self.pos[c]
        vec = self.target[mates] - carrier
        dist = np.hypot(vec[:, 0], vec[:, 1])
        for k in np.argsort(dist)[:2]:
            want = min(max(dist[k], 9.0), 17.0)
            self.target[mates[k]] = pitch.clip(carrier + vec[k] / max(dist[k], 0.1) * want, 1.0)
            self.urg[mates[k]] = 0.6

    def _run_targets(self, c: int, team: int) -> None:
        style = self.style[team]
        d = self.direction[team]
        rb = pitch.to_rel(self.ball[:2], d)[0]
        line = None
        for s in self._slots(team):
            if s == c:
                continue
            if self.run_until[s] > self.frame:
                self.target[s] = self.run_target[s]
                self.urg[s] = 0.95
                continue
            rate = RUN_RATE.get(self.slot_role[s], 0.0) * style.tempo
            if rate and rb > 35.0 and self.rng.random() < rate * DT:
                if line is None:
                    line = self._offside_line(team)
                rel = pitch.to_rel(self.pos[s], d)
                if rel[0] < line - 8.0 or line > 96.0:
                    continue
                tx = min(line + self.rng.uniform(7.0, 13.0), 102.0)
                ty = rel[1] + 0.3 * (34.0 - rel[1]) + self.rng.uniform(-4.0, 4.0)
                self.run_target[s] = pitch.to_rel(np.array([tx, np.clip(ty, 3.0, 65.0)]), d)
                self.run_until[s] = self.frame + int(self.rng.uniform(2.0, 3.0) * FPS)

    def _forward(self, c: int) -> np.ndarray:
        d = self.direction[self.team_of[c]]
        fwd = np.array([float(d), 0.0])
        if pitch.to_rel(self.pos[c], d)[0] > 75.0:
            fwd = pitch.unit(0.6 * pitch.unit(pitch.goal_center(d) - self.pos[c]) + 0.4 * fwd)
        return fwd

    def _space_ahead(self, c: int, fwd: np.ndarray, radius: float = 7.0) -> bool:
        opp = self._slots(1 - self.team_of[c])
        vec = self.pos[opp] - self.pos[c]
        dist = np.hypot(vec[:, 0], vec[:, 1]) + 1e-9
        return not np.any((dist < radius) & ((vec @ fwd) / dist > 0.5))

    def _carrier_target(self, c: int) -> None:
        p = self.pos[c]
        if self.slot_role[c] == "GK":
            self.target[c] = p
            self.urg[c] = 0.3
            return
        fwd = self._forward(c)
        j, dist = self._nearest_opp(c)
        if j is not None and dist < 2.5:
            away = pitch.unit(p - self.pos[j])
            self.target[c] = pitch.clip(p + pitch.unit(0.6 * away + 0.4 * fwd) * 3.0, 1.0)
            self.urg[c] = 0.4
        elif self._space_ahead(c, fwd):
            self.target[c] = pitch.clip(p + fwd * 6.0, 1.0)
            self.urg[c] = 0.62
        else:
            self.target[c] = pitch.clip(p + fwd * 2.0, 1.0)
            self.urg[c] = 0.38

    def _restart_targets(self, r: Restart) -> None:
        team = r.team
        d = self.direction[team]
        if r.kind == "kick_off":
            for t in (0, 1):
                sl = self._slots(t)
                rel = pitch.to_rel(self.target[sl], self.direction[t])
                # the defending side stays outside the centre circle
                rel[:, 0] = np.minimum(rel[:, 0], 50.5 if t == team else 43.0)
                self.target[sl] = pitch.to_rel(rel, self.direction[t])
        elif r.kind == "corner":
            att = sorted((int(s) for s in self._slots(team) if s != r.taker and self.slot_role[s] != "GK"),
                         key=lambda s: -self._r(s, "aerial"))
            for s, xy in zip(att, CORNER_ATTACK):
                self.target[s] = pitch.to_rel(np.array(xy), d)
            gk = self._gk(1 - team)
            if gk is not None:
                self.target[gk] = pitch.to_rel(np.array([104.2, 34.0]), d)
            dfd = sorted((int(s) for s in self._slots(1 - team) if s != gk), key=lambda s: -self._r(s, "aerial"))
            for s, xy in zip(dfd, CORNER_DEFEND):
                self.target[s] = pitch.to_rel(np.array(xy), d)
        elif r.kind == "penalty":
            gk = self._gk(1 - team)
            sl = [int(s) for s in range(22) if self.active[s] and s not in (r.taker, gk)]
            rel = pitch.to_rel(self.target[sl], d)
            rel[:, 0] = np.minimum(rel[:, 0], 86.0)
            self.target[sl] = pitch.to_rel(rel, d)
            if gk is not None:
                self.target[gk] = pitch.to_rel(np.array([104.6, 34.0]), d)
        elif r.kind == "goal_kick":
            opp = self._slots(1 - team)
            rel = pitch.to_rel(self.target[opp], d)
            rel[:, 0] = np.maximum(rel[:, 0], 18.0)
            self.target[opp] = pitch.to_rel(rel, d)
        if r.taker is not None and self.active[r.taker]:
            self.target[r.taker] = r.spot
            self.urg[r.taker] = 0.7

    # --------------------------------------------------------------- motion

    def _integrate(self) -> None:
        to = self.target - self.pos
        dist = np.hypot(to[:, 0], to[:, 1]) + 1e-9
        fatigue = 1.0 - 0.07 * (self.covered / 10_000.0) * (1.3 - self.stamina / 100.0)
        cap = self.vmax * fatigue * self.urg
        cap = np.where(self.stun_until > self.frame, cap * 0.35, cap)
        want = np.minimum(cap, 1.6 * dist)
        desired = to / dist[:, None] * want[:, None]
        dv = desired - self.vel
        dvn = np.hypot(dv[:, 0], dv[:, 1]) + 1e-9
        speed = np.hypot(self.vel[:, 0], self.vel[:, 1])
        amax = np.where(want < speed, 1.6 * self.accel, self.accel)
        self.vel += dv * np.minimum(1.0, amax * DT / dvn)[:, None]
        self.vel[~self.active] = 0.0
        self.pos += self.vel * DT
        self.covered += np.hypot(self.vel[:, 0], self.vel[:, 1]) * DT

    def _attach_ball(self) -> None:
        c = self.owner
        v = self.vel[c]
        speed = float(np.hypot(*v))
        dirn = v / speed if speed > 0.3 else np.array([float(self.direction[self.team_of[c]]), 0.0])
        self.ball[:2] = pitch.clip(self.pos[c] + dirn * 0.45, 0.05)
        self.ball[2] = 1.0 if self.slot_role[c] == "GK" and speed < 0.3 else 0.0

    def _record(self) -> None:
        i = self.n
        self.f_t[i] = self.match_time
        self.f_period[i] = self.period
        self.f_state[i] = STATE_CODE[self.state]
        self.f_owner[i] = -1 if self.owner is None else self.owner
        self.f_ball[i] = self.ball
        xy = self.pos.astype(np.float32)
        xy[~self.active] = np.nan
        self.f_xy[i] = xy
        self.f_ids[i] = self.slot_reg
        self.n += 1

    # ------------------------------------------------------------ on the ball

    def _hold_frames(self, s: int) -> int:
        style = self.style[self.team_of[s]]
        base = 2.2 if self.slot_role[s] == "GK" else 1.1 / style.tempo
        if self._nearest_opp(s)[1] < 4.0:
            base *= 0.55
        hold = base * math.exp(self.rng.normal(0.0, 0.45))
        return int(round(min(max(hold, 0.3), 4.5) * FPS))

    def _start_spell(self, s: int) -> None:
        self.spell = Spell(s, self.frame, self.ball[:2].copy(), self.frame + self._hold_frames(s), self.frame + 4)

    def _end_spell(self, s: int) -> None:
        sp = self.spell
        if sp is None or sp.slot != s:
            return
        self.spell = None
        end = self.ball[:2].copy()
        length = float(np.hypot(*(end - sp.start_pos)))
        if length >= 3.0 and self.frame > sp.start_frame:
            self._event("carry", s, location=sp.start_pos, end_location=end, frame=sp.start_frame,
                        length_m=round(length, 1), duration_s=round((self.frame - sp.start_frame) * DT, 1),
                        end_frame=self.frame)

    def _complete_pass(self, ev: dict, s: int) -> None:
        ev["outcome"] = "complete"
        ev["attributes"]["recipient"] = self._pid(s)
        self.last_pass = ev

    def _gain_control(self, s: int, cause: str) -> None:
        team = int(self.team_of[s])
        self.state = OWNED
        self.owner = s
        self.flight = None
        self.loose_v[:] = 0.0
        self.ball[2] = 0.0
        self.last_touch_team = team
        pp = self.pending
        if pp is not None:
            self.pending = None
            if pp.team == team and self.frame <= pp.deadline:
                self._complete_pass(pp.event, s)
            else:
                pp.event["outcome"] = "incomplete"
        if team != self.possession_team:
            self._set_possession(team, cause)
            if cause == "loose":
                self._event("ball_recovery", s, location=self.ball[:2])
        self._start_spell(s)

    def _make_loose(self, velocity) -> None:
        self.state = LOOSE
        self.owner = None
        self.flight = None
        self.spell = None
        self.loose_v = np.asarray(velocity, dtype=float)

    def _random_direction(self) -> np.ndarray:
        a = self.rng.uniform(0, 2 * math.pi)
        return np.array([math.cos(a), math.sin(a)])

    def _update_owned(self) -> None:
        c = self.owner
        sp = self.spell
        team = int(self.team_of[c])
        opp = self._slots(1 - team)
        vec = self.pos[opp] - self.pos[c]
        dist = np.hypot(vec[:, 0], vec[:, 1])
        for j, dd, v in zip(opp, dist, vec):
            j = int(j)
            if dd < 2.5 and j not in sp.pressers:
                closing = float((self.vel[j] - self.vel[c]) @ (-v / max(dd, 1e-6)))
                if closing > 2.0:
                    sp.pressers.add(j)
                    self._event("pressure", j, location=self.pos[j], opponent=self._pid(c))
        if self.frame >= sp.tackle_ready and self.slot_role[c] != "GK":
            for j, dd in zip(opp, dist):
                if dd < 1.1 and self.stun_until[j] <= self.frame:
                    rate = 0.15 + 0.2 * self._r(int(j), "defending") / 100.0
                    if self.rng.random() < rate * DT:
                        self._tackle(int(j), c)
                        return
        if self.frame >= sp.decision_frame:
            self._decide(c)

    def _decide(self, c: int, restart: str | None = None) -> None:
        team = int(self.team_of[c])
        d = self.direction[team]
        style = self.style[team]
        ball = self.ball[:2].copy()
        rel = pitch.to_rel(ball, d)
        role = self.slot_role[c]
        _, press_d = self._nearest_opp(c)

        if restart == "penalty":
            return self._shoot(c, set_piece="penalty")
        if restart == "corner":
            return self._kick_pass(c, self._plan_cross(c), set_piece="corner")
        if restart == "free_kick":
            dist, ang = pitch.shot_geometry(ball, d)
            if dist < 28.0 and ang > 0.22 and self.rng.random() < 0.45:
                return self._shoot(c, set_piece="free_kick")
        options: list[tuple[float, str, object]] = []
        if restart is None and role != "GK":
            dist, ang = pitch.shot_geometry(ball, d)
            if dist < 30.0 and ang > 0.12:
                xg = self._shot_xg(c)
                if xg >= 0.02:
                    tendency = 0.8 + 0.4 * (self._r(c, "finishing") - 50.0) / 45.0
                    options.append((0.5 + 2.6 * xg * tendency, "shoot", None))
        max_len = 25.0 if restart == "throw_in" else 75.0
        kind = "throw_in" if restart == "throw_in" else "pass"
        for m in (int(m) for m in self._slots(team) if m != c):
            lead = 0.6 if float(np.hypot(*self.vel[m])) > 2.5 else 0.15
            tp = pitch.clip(self.pos[m] + self.vel[m] * lead, 1.0)
            if np.hypot(*(tp - ball)) > max_len:
                continue
            plan = self._plan_pass(c, tp, m, kind=kind, lofted=False if restart == "throw_in" else None)
            u = self._pass_utility(c, plan)
            if self.slot_role[m] == "GK":
                u -= 0.08 if press_d < 3.0 else 0.3
            options.append((u, "pass", plan))
            if restart is None and self.run_until[m] > self.frame:
                tp2 = self._through_point(m, team)
                if tp2 is not None:
                    plan2 = self._plan_pass(c, tp2, m, kind="through")
                    options.append((self._pass_utility(c, plan2) + 0.22, "pass", plan2))
        if restart is None and role != "GK":
            if rel[0] > 70.0 and (rel[1] < 18.0 or rel[1] > 50.0):
                plan = self._plan_cross(c)
                options.append((self._pass_utility(c, plan) + 0.12 * style.crossing, "pass", plan))
            options.extend(self._carry_options(c))
            if rel[0] < 30.0 and press_d < 4.0 and ROLE_GROUP[role] in ("DEF", "MID"):
                options.append((0.35 + 0.2 * style.risk, "clear", None))
        if not options:
            return self._clear(c)

        u = np.array([o[0] for o in options])
        temperature = 0.07 + 0.08 * (1.0 - self._r(c, "passing") / 100.0)
        p = np.exp((u - u.max()) / temperature)
        k = int(self.rng.choice(len(options), p=p / p.sum()))
        _, action, payload = options[k]
        if action == "pass":
            self._kick_pass(c, payload, set_piece=restart)
        elif action == "shoot":
            self._shoot(c)
        elif action == "carry":
            self.spell.carries += 1
            self.spell.decision_frame = self.frame + int(self.rng.uniform(0.5, 1.2) * FPS)
        elif action == "dribble":
            self._dribble(c, payload)
        else:
            self._clear(c)

    def _carry_options(self, c: int) -> list[tuple[float, str, object]]:
        team = int(self.team_of[c])
        d = self.direction[team]
        style = self.style[team]
        p = self.pos[c]
        fwd = self._forward(c)
        j, dist = self._nearest_opp(c)
        here_risk = expected_threat(*pitch.to_rel(p, -d))
        risk_t = 0.3 if dist < 1.5 else 0.12 if dist < 3.0 else 0.02
        ahead = pitch.clip(p + fwd * 4.0, 1.0)
        here = expected_threat(*pitch.to_rel(p, d))
        gain = min(expected_threat(*pitch.to_rel(ahead, d)), here + 0.03)
        u = 0.9 * (1 - risk_t) + XT_WEIGHT * ((1 - risk_t) * gain - risk_t * 1.5 * style.risk * here_risk)
        if self._space_ahead(c, fwd):
            u += 0.04
        u -= 0.06 * (self.spell.carries if self.spell is not None else 0)
        out: list[tuple[float, str, object]] = [(u, "carry", None)]
        if j is not None and dist < 3.0 and float(pitch.unit(self.pos[j] - p) @ fwd) > 0.3:
            pd = min(max(0.5 + 0.007 * (self._r(c, "dribbling") - self._r(j, "defending")), 0.2), 0.8)
            beyond = pitch.clip(p + fwd * 5.0, 1.0)
            u2 = pd + XT_WEIGHT * (pd * expected_threat(*pitch.to_rel(beyond, d)) - (1 - pd) * 1.5 * style.risk * here_risk)
            out.append((u2 + (self._r(c, "dribbling") - 70.0) / 250.0, "dribble", j))
        return out

    # ---------------------------------------------------------------- passes

    def _plan_pass(self, c: int, target, receiver: int | None, *, kind: str = "pass",
                   lofted: bool | None = None) -> PassPlan:
        team = int(self.team_of[c])
        origin = self.ball[:2].copy()
        target = pitch.clip(target, 0.5)
        dist = max(float(np.hypot(*(target - origin))), 0.5)
        if lofted is None:
            lofted = dist > 32.0 or kind == "cross"
        v0, v_end, duration, apex = pass_kinematics(dist, lofted)
        _, press_d = self._nearest_opp(c)
        p_acc = pass_accuracy(self._r(c, "passing"), dist, lofted, press_d, kind)
        p_lane, j, tj, qj = self._lane(origin, target, dist, duration, v0, v_end, lofted, apex, team)
        p_ctrl = self._control(receiver, target, duration, team, lofted)
        return PassPlan(kind, receiver, target, dist, lofted, v0, v_end, duration, apex,
                        p_acc, p_lane, p_ctrl, j, tj, qj)

    def _lane(self, origin, target, dist, duration, v0, v_end, lofted, apex, team):
        """Probability that no opponent intercepts the pass, and the likeliest interceptor."""
        opp = self._slots(1 - team)
        n = max(int(dist / 1.5), 2)
        s = np.linspace(1.0, max(dist - 1.5, 1.2), n)
        if lofted:
            t = s / dist * duration
        else:
            k = (v0 - v_end) / duration
            t = (v0 - np.sqrt(np.maximum(v0 * v0 - 2.0 * k * s, 0.0))) / k if k > 1e-6 else s / v0
        u = (target - origin) / dist
        pts = origin + s[:, None] * u
        dd = np.linalg.norm(self.pos[opp][:, None, :] - pts[None, :, :], axis=2)
        reach = np.array([1.7 if self.slot_role[o] == "GK" else 1.0 for o in opp])
        td = 0.25 + np.maximum(dd - reach[:, None], 0.0) / (0.92 * self.vmax[opp])[:, None]
        margin = t[None, :] - td
        if lofted:
            frac = t / duration
            low = (4.0 * apex * frac * (1.0 - frac) < 2.3) & (frac < 0.3)
            margin = np.where(low[None, :], margin, -9.0)
        p = 1.0 / (1.0 + np.exp(-margin / 0.15))
        pj = p.max(axis=1)
        p_lane = float(np.prod(1.0 - 0.4 * pj))
        jj = int(np.argmax(pj))
        if pj[jj] < 0.05:
            return p_lane, None, 0.0, None
        hits = np.flatnonzero(p[jj] > 0.5)
        k = int(hits[0]) if len(hits) else int(np.argmax(p[jj]))
        return p_lane, int(opp[jj]), float(t[k]), pts[k].copy()

    def _control(self, receiver, target, duration, team, lofted) -> float:
        """Probability the intended side wins the ball where the pass arrives."""
        mates = self._slots(team)
        if receiver is None:
            times = self._reach_time(mates, target, 0.2, 0.5)
            receiver = int(mates[np.argmin(times)])
            tr = float(times.min())
        else:
            tr = float(self._reach_time([receiver], target, 0.2, 0.5)[0])
        to = float(self._reach_time(self._slots(1 - team), target, 0.25, 1.0).min())
        # A marker only wins the ball if he gets to it before the receiver does.
        margin = to - tr
        late = duration + (0.4 if lofted else 1.0) - tr
        touch = 0.955 + 0.035 * (self._r(receiver, "dribbling") - 50.0) / 45.0 - (0.06 if lofted else 0.0)
        return logistic((margin + 0.1) / 0.25) * logistic(late / 0.3) * min(touch, 0.995)

    def _pass_utility(self, c: int, plan: PassPlan) -> float:
        team = int(self.team_of[c])
        d = self.direction[team]
        style = self.style[team]
        rel_from = pitch.to_rel(self.ball[:2], d)
        rel_to = pitch.to_rel(plan.target, d)
        P = plan.p_success
        gain = expected_threat(*rel_to)
        risk = expected_threat(*pitch.to_rel(plan.loss_point, -d))
        progress = float(rel_to[0] - rel_from[0])
        u = P + XT_WEIGHT * (P * gain - (1 - P) * 1.5 * style.risk * risk)
        u += style.directness * 0.004 * max(progress, 0.0) * P
        u -= 0.004 * max(-progress, 0.0) * style.directness
        if plan.lofted:
            u -= 0.03 + 0.12 * (1 - style.directness)
        return u

    def _through_point(self, m: int, team: int):
        """Where to play a ball into the space a runner is attacking."""
        gap = self.run_target[m] - self.pos[m]
        if np.hypot(*gap) < 3.0:
            return None
        return self.pos[m] + gap * 0.8

    def _plan_cross(self, c: int) -> PassPlan:
        team = int(self.team_of[c])
        d = self.direction[team]
        att = np.array([s for s in self._slots(team) if s != c and self.slot_role[s] != "GK"])
        best = None
        rel = pitch.to_rel(self.pos[att], d)
        in_box = att[(rel[:, 0] > 84.0) & (np.abs(rel[:, 1] - 34.0) < 14.0)]
        candidates = [(int(m), pitch.to_rel(np.clip(pitch.to_rel(self.pos[m], d) + [1.5, 0.0], [86, 27], [101, 41]), d))
                      for m in in_box]
        for _ in range(max(0, 2 - len(candidates))):
            tp = pitch.to_rel(np.array([self.rng.uniform(93.0, 100.0), self.rng.uniform(28.0, 40.0)]), d)
            candidates.append((int(att[np.argmin(self._reach_time(att, tp, 0.2, 0.5))]), tp))
        for m, tp in candidates:
            plan = self._plan_pass(c, tp, m, kind="cross", lofted=True)
            if best is None or plan.p_success > best.p_success:
                best = plan
        return best

    def _kick_pass(self, c: int, plan: PassPlan, set_piece: str | None = None) -> None:
        team = int(self.team_of[c])
        d = self.direction[team]
        self._end_spell(c)
        origin = self.ball[:2].copy()
        target = plan.target.copy()
        accurate = bool(self.rng.random() < plan.p_accuracy)
        if not accurate:
            spread = 3.5 + 0.12 * plan.distance + (3.0 if plan.lofted else 0.0)
            target = target + self._random_direction() * spread * self.rng.uniform(0.6, 1.0)
        dist = max(float(np.hypot(*(target - origin))), 0.5)
        v0, v_end, duration, apex = pass_kinematics(dist, plan.lofted, self.rng.uniform(0.95, 1.05))
        if set_piece == "throw_in":
            v0, v_end, duration = min(v0, 13.0), min(v_end, 8.0), dist / min(0.75 * v0, 10.0)
        body = "hands" if set_piece == "throw_in" else (
            "left_foot" if (self.slot_player[c].foot == "L") == (self.rng.random() < 0.85) else "right_foot")
        receiver_id = self._pid(plan.receiver) if plan.receiver is not None else None
        ev = self._event("pass", c, location=origin, end_location=target, recipient=receiver_id,
                         length_m=round(dist, 1), height="high" if plan.lofted else "ground",
                         kind=plan.kind if plan.kind != "throw_in" else "pass", set_piece=set_piece,
                         body_part=body, speed_ms=round(v0, 1))
        self.truth.append({
            "event": ev["id"], "type": "pass", "p_success": plan.p_success, "p_accuracy": plan.p_accuracy,
            "p_lane": plan.p_lane, "p_control": plan.p_control, "accurate": accurate,
            "xt_start": expected_threat(*pitch.to_rel(origin, d)),
            "xt_end": expected_threat(*pitch.to_rel(plan.target, d)),
        })
        if plan.kind == "through" and plan.receiver is not None and self._is_offside(plan.receiver, team):
            ev["outcome"] = "offside"
            off = self._event("offside", plan.receiver, location=self.pos[plan.receiver], pass_event=ev["id"])
            ev["related"].append(off["id"])
            spot = pitch.clip(self.pos[plan.receiver], 1.0)
            self.state = DEAD
            return self._dead("free_kick", 1 - team, spot, self.rng.uniform(12, 26))
        if plan.kind == "cross" and set_piece is None:
            j, dj = self._nearest_opp(c)
            if j is not None and dj < 2.5 and self.rng.random() < 0.3:
                ev["outcome"] = "incomplete"
                bev = self._event("block", j, location=self.pos[j], pass_event=ev["id"])
                ev["related"].append(bev["id"])
                self.last_touch_team = 1 - team
                if self.rng.random() < 0.55:
                    return self._award_corner(team, origin[1])
                self.ball[:2] = self.pos[j]
                return self._make_loose(self._random_direction() * self.rng.uniform(3.0, 8.0))
        intercept = None
        if plan.interceptor is not None and self.rng.random() > plan.p_lane:
            intercept = (plan.interceptor, plan.intercept_time, plan.intercept_point)
        self.flight = Flight("pass", ev, team, c, origin, target, self.frame, duration, v0, v_end,
                             plan.lofted, apex=apex, receiver=plan.receiver, intercept=intercept)
        self.pending = PendingPass(ev, team, self.frame + int((duration + 0.6) * FPS))
        self.state = FLIGHT
        self.owner = None
        self.last_touch_team = team

    def _is_offside(self, s: int, team: int) -> bool:
        rel_x = pitch.to_rel(self.pos[s], self.direction[team])[0]
        margin = rel_x - self._offside_line(team)
        # assistant referees miss some very tight calls
        return margin > 0.8 or (margin > 0.2 and self.rng.random() < 0.5)

    def _clear(self, c: int) -> None:
        team = int(self.team_of[c])
        d = self.direction[team]
        self._end_spell(c)
        origin = self.ball[:2].copy()
        target = pitch.to_rel(np.array([self.rng.uniform(50.0, 82.0), self.rng.uniform(-6.0, 74.0)]), d)
        dist = float(np.hypot(*(target - origin)))
        v0, v_end, duration, apex = pass_kinematics(dist, True, self.rng.uniform(1.0, 1.15))
        ev = self._event("clearance", c, location=origin, end_location=target, body_part="foot")
        self.flight = Flight("clearance", ev, team, c, origin, target, self.frame, duration, v0, v_end, True, apex=apex)
        self.state = FLIGHT
        self.owner = None
        self.last_touch_team = team

    # ------------------------------------------------------------ duels

    def _tackle(self, j: int, c: int) -> None:
        foul_p = 0.2 + 0.1 * (1.0 - self._r(j, "defending") / 100.0)
        if pitch.in_penalty_area_rel(pitch.to_rel(self.ball[:2], self.direction[self.team_of[c]])):
            foul_p *= 0.12
        if self.rng.random() < foul_p:
            return self._foul(j, c)
        p_win = min(max(0.6 + 0.007 * (self._r(j, "defending") - self._r(c, "dribbling")), 0.25), 0.85)
        if self.rng.random() >= p_win:
            self._event("tackle", j, location=self.pos[j], outcome="lost", opponent=self._pid(c))
            self._event("dribble", c, location=self.ball[:2], outcome="complete", opponent=self._pid(j))
            self.stun_until[j] = self.frame + 10
            return
        self._end_spell(c)
        dteam = int(self.team_of[j])
        if self.rng.random() < 0.6:
            self._set_possession(dteam, "tackle")
            self._event("tackle", j, location=self.pos[j], outcome="won", opponent=self._pid(c))
            return self._gain_control(j, "tackle")
        self._event("tackle", j, location=self.pos[j], outcome="won", opponent=self._pid(c))
        self.last_touch_team = dteam
        self._make_loose(self._random_direction() * self.rng.uniform(3.0, 6.0))

    def _dribble(self, c: int, j: int) -> None:
        pd = min(max(0.5 + 0.007 * (self._r(c, "dribbling") - self._r(j, "defending")), 0.2), 0.8)
        if self.rng.random() < pd:
            self._event("dribble", c, location=self.ball[:2], outcome="complete", opponent=self._pid(j))
            self.stun_until[j] = self.frame + 12
            self.vel[c] += self._forward(c) * 1.5
            self.spell.decision_frame = self.frame + int(self.rng.uniform(0.6, 1.2) * FPS)
            return
        self._event("dribble", c, location=self.ball[:2], outcome="incomplete", opponent=self._pid(j))
        self._end_spell(c)
        dteam = int(self.team_of[j])
        if self.rng.random() < 0.6:
            self._set_possession(dteam, "tackle")
            self._event("tackle", j, location=self.pos[j], outcome="won", opponent=self._pid(c))
            return self._gain_control(j, "tackle")
        self._event("tackle", j, location=self.pos[j], outcome="won", opponent=self._pid(c))
        self.last_touch_team = dteam
        self._make_loose(self._random_direction() * self.rng.uniform(3.0, 6.0))

    def _foul(self, j: int, c: int) -> None:
        self._end_spell(c)
        victim_team = int(self.team_of[c])
        d = self.direction[victim_team]
        spot = self.ball[:2].copy()
        rel = pitch.to_rel(spot, d)
        in_box = bool(pitch.in_penalty_area_rel(rel))
        self._event("foul", j, location=spot, victim=self._pid(c), in_box=in_box)
        defenders_behind = sum(1 for s in self._slots(1 - victim_team)
                               if pitch.to_rel(self.pos[s], d)[0] > rel[0])
        self._maybe_card(j, tactical=rel[0] > 45.0 and defenders_behind < 5)
        if in_box:
            spot = pitch.to_rel(np.array([pitch.LENGTH - pitch.PENALTY_SPOT, pitch.WIDTH / 2]), d)
            return self._dead("penalty", victim_team, spot, self.rng.uniform(50, 90))
        self._dead("free_kick", victim_team, spot, self.rng.uniform(22, 44))

    def _maybe_card(self, j: int, tactical: bool) -> None:
        pid = self._pid(j)
        booked = self.cards.get(pid, 0) >= 1
        r = self.rng.random()
        p_yellow = (0.1 + (0.22 if tactical else 0.0)) * (0.35 if booked else 1.0)
        if r < 0.002:
            card = "red"
        elif r < p_yellow:
            card = "second_yellow" if booked else "yellow"
        else:
            return
        self.cards[pid] = self.cards.get(pid, 0) + 1
        self._event("card", j, card=card)
        if card != "yellow":
            self._send_off(j)

    def _send_off(self, j: int) -> None:
        self.active[j] = False
        self.pos[j] = (-5.0, -5.0)
        self.vel[j] = 0.0
        if self.slot_role[j] == "GK":
            team = int(self.team_of[j])
            stand_in = min((s for s in self._slots(team) if ROLE_GROUP[self.slot_role[s]] == "DEF"),
                           key=lambda s: -self._r(int(s), "aerial"), default=None)
            if stand_in is not None:
                self.slot_role[stand_in] = "GK"
                self._load_slot(int(stand_in))

    # ----------------------------------------------------------------- shots

    def _blockers(self, c: int) -> list[int]:
        team = int(self.team_of[c])
        b = self.ball[:2]
        g = pitch.goal_center(self.direction[team])
        axis = g - b
        length2 = float(axis @ axis)
        out = []
        for s in self._slots(1 - team):
            if self.slot_role[s] == "GK":
                continue
            q = self.pos[s]
            t = float((q - b) @ axis) / length2
            if 0.0 < t < 1.0 and np.hypot(*(q - b)) < 11.0:
                off_line = np.hypot(*(q - (b + t * axis)))
                if off_line < t * pitch.GOAL_WIDTH / 2 + 0.9:
                    out.append(int(s))
        return out

    def _shot_xg(self, c: int, header: bool = False, set_piece: str | None = None) -> float:
        dist, ang = pitch.shot_geometry(self.ball[:2], self.direction[self.team_of[c]])
        blockers = [] if set_piece == "penalty" else self._blockers(c)
        opp = [s for s in self._slots(1 - self.team_of[c]) if self.slot_role[s] != "GK"]
        crowd = int(np.sum(np.hypot(*(self.pos[opp] - self.pos[c]).T) < 3.0)) if opp else 0
        return expected_goals(dist, ang, header=header, blockers=min(len(blockers), 3),
                              pressured=self._nearest_opp(c)[1] < 2.5, crowd=crowd, set_piece=set_piece)

    def _shoot(self, c: int, set_piece: str | None = None, header: bool = False) -> None:
        team = int(self.team_of[c])
        d = self.direction[team]
        ball = self.ball[:2].copy()
        dist, ang = pitch.shot_geometry(ball, d)
        blockers = [] if set_piece == "penalty" else self._blockers(c)
        xg = self._shot_xg(c, header=header, set_piece=set_piece)
        self._end_spell(c)
        gk = self._gk(1 - team)
        keep = self._r(gk, "keeping") if gk is not None else 30.0
        fin = self._r(c, "finishing")
        p_goal = min(max(xg * (0.85 + 0.3 * (fin - 50) / 45) * (1.15 - 0.3 * (keep - 50) / 45), 0.003), 0.95)
        p_block = 1.0 - 0.65 ** len(blockers)
        blocker = min(blockers, key=lambda s: np.hypot(*(self.pos[s] - ball))) if blockers else None
        r = self.rng.random()
        if r < p_goal:
            outcome = "goal"
        elif r < p_goal + (1 - p_goal) * p_block:
            outcome = "blocked"
        else:
            r2 = self.rng.random()
            outcome = "saved" if r2 < 0.34 else "post" if r2 < 0.39 else "off_target"
        if header:
            speed = self.rng.uniform(9.0, 15.0)
        elif set_piece == "penalty":
            speed = float(np.clip(self.rng.normal(24.0, 3.0), 17.0, 32.0))
        else:
            speed = float(np.clip(self.rng.normal(24.0 + 0.12 * (dist - 16.0) + 0.08 * (fin - 70.0), 3.2), 14.0, 36.0))
        goal_x = pitch.LENGTH if d > 0 else 0.0
        z = 0.0
        if outcome == "goal":
            end = np.array([goal_x, 34.0 + self.rng.uniform(-3.3, 3.3)])
            z = self.rng.uniform(0.1, 2.3)
        elif outcome == "saved":
            gy = self.pos[gk][1] if gk is not None else 34.0
            end = np.array([goal_x - d * 0.6, 34.0 + float(np.clip(gy - 34.0 + self.rng.normal(0, 1.2), -3.3, 3.3))])
            z = self.rng.uniform(0.2, 2.0)
        elif outcome == "post":
            end = np.array([goal_x, 34.0 + self.rng.choice([-1.0, 1.0]) * pitch.GOAL_WIDTH / 2])
            z = self.rng.uniform(0.3, 2.3)
        elif outcome == "off_target":
            if self.rng.random() < 0.35:
                end = np.array([goal_x + d * 1.5, 34.0 + self.rng.uniform(-3.5, 3.5)])
                z = self.rng.uniform(2.6, 4.5)
            else:
                end = np.array([goal_x + d * 1.5, 34.0 + self.rng.choice([-1.0, 1.0]) * self.rng.uniform(4.0, 9.0)])
                z = self.rng.uniform(0.1, 2.0)
        else:
            end = self.pos[blocker].copy()
            z = 0.4
        foot = self.slot_player[c].foot
        body = "head" if header else ("left_foot" if (foot == "L") == (self.rng.random() < 0.8) else "right_foot")
        ev = self._event("shot", c, location=ball, end_location=end, xg=round(xg, 3), body_part=body,
                         speed_ms=round(speed, 1), distance_m=round(dist, 1), angle_deg=round(math.degrees(ang), 1),
                         set_piece=set_piece, goalkeeper=self._pid(gk) if gk is not None else None)
        lp = self.last_pass
        if lp is not None and lp["attributes"].get("recipient") == self._pid(c) and lp["possession"] == self.possession_id:
            lp["attributes"]["shot_assist"] = True
        self.truth.append({"event": ev["id"], "type": "shot", "xg": xg, "p_goal": p_goal, "p_block": p_block,
                           "planned_outcome": outcome})
        length = float(np.hypot(*(end - ball)))
        self.flight = Flight("shot", ev, team, c, ball, end, self.frame, max(length / speed, 0.15), speed, speed,
                             False, z_end=z, shot_outcome=outcome, blocker=blocker)
        self.state = FLIGHT
        self.owner = None
        self.last_touch_team = team

    def _resolve_shot(self, f: Flight) -> None:
        ev = f.event
        team = f.team
        out = f.shot_outcome
        ev["outcome"] = out
        if out == "goal":
            return self._goal(team, f.passer, ev)
        d = self.direction[team]
        away_from_goal = pitch.unit(f.origin - pitch.goal_center(d))
        if out == "blocked":
            bev = self._event("block", f.blocker, location=self.ball[:2], shot_event=ev["id"])
            ev["related"].append(bev["id"])
            self.last_touch_team = 1 - team
            if self.rng.random() < 0.4:
                return self._award_corner(team, self.ball[1])
            return self._make_loose(pitch.unit(away_from_goal + 0.8 * self._random_direction()) * self.rng.uniform(4, 10))
        gk = self._gk(1 - team)
        if out == "saved" and gk is not None:
            caught = self.rng.random() < (0.5 if f.v0 < 24.0 else 0.3)
            gev = self._event("goalkeeper", gk, location=self.ball[:2], outcome="caught" if caught else "parried",
                              action="save", shot_event=ev["id"])
            ev["related"].append(gev["id"])
            self.pos[gk] = self.ball[:2] - away_from_goal * 0.3
            if caught:
                self._set_possession(1 - team, "save")
                return self._gain_control(gk, "save")
            self.last_touch_team = 1 - team
            if self.rng.random() < 0.45:
                return self._award_corner(team, self.ball[1])
            return self._make_loose(pitch.unit(away_from_goal + 0.9 * self._random_direction()) * self.rng.uniform(5, 11))
        if out == "post":
            self.last_touch_team = team
            self.ball[:2] = pitch.clip(self.ball[:2], 0.3)
            return self._make_loose(pitch.unit(away_from_goal + 0.6 * self._random_direction()) * self.rng.uniform(6, 12))
        self._ball_out(self.ball[:2].copy(), team)

    def _goal(self, team: int, scorer: int, shot_ev: dict) -> None:
        self.score[team] += 1
        self.goals.append((int(self.match_time // 60), team))
        assist = None
        lp = self.last_pass
        if lp is not None and lp["attributes"].get("recipient") == self._pid(scorer) and lp["possession"] == self.possession_id:
            lp["attributes"]["goal_assist"] = True
            assist = lp["player"]
        gev = self._event("goal", scorer, location=self.ball[:2], scorer=self._pid(scorer), assist=assist,
                          score_home=self.score[0], score_away=self.score[1], shot_event=shot_ev["id"], own_goal=False)
        shot_ev["related"].append(gev["id"])
        self._dead("kick_off", 1 - team, pitch.CENTER.copy(), self.rng.uniform(60, 90))

    # ------------------------------------------------------- ball in flight

    def _update_flight(self) -> None:
        f = self.flight
        tau = (self.frame - f.start_frame + 1) * DT
        if f.intercept is not None and tau >= f.intercept[1]:
            j, _, q = f.intercept
            if self.active[j] and np.hypot(*(self.pos[j] - q)) < 2.5:
                self.ball[:] = (q[0], q[1], 0.0)
                return self._intercepted(f, j)
            f.intercept = None
        self.ball[:] = f.ball_at(tau)
        if not pitch.inside(self.ball[:2]):
            if f.kind == "shot":
                return self._resolve_shot(f)
            if self.pending is not None and self.pending.event is f.event:
                f.event["outcome"] = "out"
                self.pending = None
            return self._ball_out(self.ball[:2].copy(), f.team)
        if tau >= f.duration:
            if f.kind == "shot":
                return self._resolve_shot(f)
            return self._resolve_arrival(f)

    def _intercepted(self, f: Flight, j: int) -> None:
        f.event["outcome"] = "incomplete"
        self.pending = None
        self._set_possession(int(self.team_of[j]), "interception")
        ev = self._event("interception", j, location=self.ball[:2], pass_event=f.event["id"])
        f.event["related"].append(ev["id"])
        self._gain_control(j, "interception")

    def _resolve_arrival(self, f: Flight) -> None:
        pt = self.ball[:2].copy()
        team = f.team
        mates = self._slots(team)
        opps = self._slots(1 - team)
        dm = np.hypot(*(self.pos[mates] - pt).T)
        do = np.hypot(*(self.pos[opps] - pt).T)
        r = None
        if f.receiver is not None and self.active[f.receiver] and np.hypot(*(self.pos[f.receiver] - pt)) <= 2.5:
            r = f.receiver
        elif dm.min() <= 2.0:
            r = int(mates[dm.argmin()])
        o = int(opps[do.argmin()]) if do.min() <= 2.0 else None
        if f.lofted:
            return self._aerial(f, r, o, pt)
        if r is None and o is None:
            return self._make_loose(pitch.unit(f.target - f.origin) * f.v_end * 0.9)
        if o is None:
            return self._gain_control(r, "pass")
        if r is None:
            return self._gain_control(o, "loose")
        dr = float(np.hypot(*(self.pos[r] - pt)))
        dO = float(np.hypot(*(self.pos[o] - pt)))
        in_their_box = pitch.in_penalty_area_rel(pitch.to_rel(pt, self.direction[team]))
        p = 0.65 - (0.15 if in_their_box else 0.0)
        p = min(max(p + 0.004 * (self._r(r, "dribbling") - self._r(o, "defending")) + 0.25 * (dO - dr), 0.15), 0.94)
        if self.rng.random() < p:
            self._gain_control(r, "pass")
            in_box = pitch.in_penalty_area_rel(pitch.to_rel(pt, self.direction[team]))
            if self.rng.random() < (0.015 if in_box else 0.18):
                self._foul(o, r)
            return
        if self.rng.random() < 0.55:
            self._set_possession(1 - team, "tackle")
            self._event("tackle", o, location=self.pos[o], outcome="won", opponent=self._pid(r))
            return self._gain_control(o, "tackle")
        self._event("tackle", o, location=self.pos[o], outcome="won", opponent=self._pid(r))
        self.last_touch_team = 1 - team
        self._make_loose(self._random_direction() * self.rng.uniform(3.0, 6.0))

    def _aerial(self, f: Flight, r: int | None, o: int | None, pt) -> None:
        team = f.team
        d = self.direction[team]
        rel_pt = pitch.to_rel(pt, d)
        gk = self._gk(1 - team)
        is_cross = f.event["attributes"].get("kind") == "cross"
        if (gk is not None and f.kind == "pass" and pitch.in_penalty_area_rel(rel_pt)
                and np.hypot(*(self.pos[gk] - pt)) < 3.5 and self.rng.random() < 0.6):
            self._set_possession(1 - team, "claim")
            self._event("goalkeeper", gk, location=pt, outcome="caught", action="claim")
            return self._gain_control(gk, "claim")
        if r is None and o is None:
            return self._make_loose(pitch.unit(f.target - f.origin) * f.v_end * 0.35)
        if o is None:
            winner = r
        elif r is None:
            winner = o
        else:
            dr = float(np.hypot(*(self.pos[r] - pt)))
            dO = float(np.hypot(*(self.pos[o] - pt)))
            p = logistic((self._r(r, "aerial") - self._r(o, "aerial")) / 12.0 + 0.8 * (dO - dr))
            if is_cross:
                p *= 0.85
            winner = r if self.rng.random() < p else o
            loser = o if winner == r else r
            if self.rng.random() < 0.05:
                if self.pending is not None:
                    self.pending.event["outcome"] = "incomplete"
                    self.pending = None
                self.ball[:2] = pt
                self.spell = None
                fouler, victim = (r, o) if pitch.in_penalty_area_rel(rel_pt) and self.rng.random() < 0.95 else (winner, loser)
                return self._foul(fouler, victim)
            self._event("aerial_duel", winner, location=pt, outcome="won", opponent=self._pid(loser))
        if winner == r:
            dist_goal, ang = pitch.shot_geometry(pt, d)
            if is_cross and dist_goal < 15.0 and ang > 0.25 and self.rng.random() < 0.65:
                self._gain_control(r, "pass")
                return self._shoot(r, header=True)
            if self.rng.random() < 0.6:
                return self._gain_control(r, "pass")
            self.last_touch_team = team
            return self._make_loose(self._random_direction() * self.rng.uniform(2.0, 5.0))
        dteam = 1 - team
        if self.pending is not None:
            self.pending.event["outcome"] = "incomplete"
            self.pending = None
        if pitch.to_rel(pt, self.direction[dteam])[0] < 35.0 or self.rng.random() < 0.6:
            self._event("clearance", winner, location=pt, body_part="head")
            self.last_touch_team = dteam
            if is_cross and self.rng.random() < 0.15:
                return self._award_corner(team, pt[1])
            away = pitch.unit(pt - pitch.own_goal_center(self.direction[dteam]))
            return self._make_loose(pitch.unit(away + 0.7 * self._random_direction()) * self.rng.uniform(8.0, 14.0))
        return self._gain_control(winner, "loose")

    # ---------------------------------------------------------- loose ball

    def _update_loose(self) -> None:
        speed = float(np.hypot(*self.loose_v))
        if speed > 0:
            self.loose_v *= max(speed - 2.2 * DT, 0.0) / speed
        self.ball[:2] += self.loose_v * DT
        self.ball[2] = max(self.ball[2] - 0.4, 0.0)
        if not pitch.inside(self.ball[:2]):
            if self.pending is not None:
                self.pending.event["outcome"] = "out"
                self.pending = None
            return self._ball_out(self.ball[:2].copy(), self.last_touch_team)
        ok = self.active & (self.stun_until <= self.frame)
        dist = np.hypot(*(self.pos - self.ball[:2]).T)
        dist[~ok] = np.inf
        cand = np.flatnonzero(dist < 1.0)
        if len(cand) == 0:
            return
        if len(cand) > 1:
            w = 1.0 / (dist[cand] + 0.2)
            s = int(self.rng.choice(cand, p=w / w.sum()))
        else:
            s = int(cand[0])
        self._gain_control(s, "loose")

    # ------------------------------------------------------------ restarts

    def _own_goal_team(self, x: float) -> int:
        """The team defending the goal line at this end of the pitch."""
        left = x < pitch.LENGTH / 2
        return 0 if (self.direction[0] > 0) == left else 1

    def _award_corner(self, team: int, side_y: float) -> None:
        defending = 1 - team
        x = 0.0 if self.direction[team] < 0 else pitch.LENGTH
        y = 0.5 if side_y < pitch.WIDTH / 2 else pitch.WIDTH - 0.5
        self._event("ball_out", team=defending, location=(x, y), restart="corner",
                    last_touch_team=self.teams[defending].id)
        self._dead("corner", team, np.array([x, y]), self.rng.uniform(25, 45))

    def _ball_out(self, point, last_team: int) -> None:
        x, y = float(point[0]), float(point[1])
        if y < 0.0 or y > pitch.WIDTH:
            kind, team = "throw_in", 1 - last_team
            spot = np.array([min(max(x, 1.0), pitch.LENGTH - 1.0), 0.2 if y < 0 else pitch.WIDTH - 0.2])
        else:
            defending = self._own_goal_team(x)
            line_x = 0.0 if x < pitch.LENGTH / 2 else pitch.LENGTH
            if last_team == defending:
                kind, team = "corner", 1 - defending
                spot = np.array([line_x, 0.5 if y < pitch.WIDTH / 2 else pitch.WIDTH - 0.5])
            else:
                kind, team = "goal_kick", defending
                inward = 5.5 if line_x == 0.0 else pitch.LENGTH - 5.5
                spot = np.array([inward, pitch.WIDTH / 2 + (-9.16 if y < pitch.WIDTH / 2 else 9.16)])
        self._event("ball_out", team=last_team, location=(x, y), restart=kind,
                    last_touch_team=self.teams[last_team].id)
        delay = {"throw_in": (12, 30), "corner": (25, 45), "goal_kick": (20, 38)}[kind]
        self._dead(kind, team, spot, self.rng.uniform(*delay))

    def _pick_taker(self, kind: str, team: int, spot) -> int:
        slots = [int(s) for s in self._slots(team)]
        outfield = [s for s in slots if self.slot_role[s] != "GK"]
        near = sorted(outfield, key=lambda s: np.hypot(*(self.pos[s] - spot)))
        gk = self._gk(team)
        if kind == "goal_kick" and gk is not None:
            return gk
        if kind == "kick_off":
            forwards = [s for s in outfield if ROLE_GROUP[self.slot_role[s]] == "ATT"]
            return min(forwards or outfield, key=lambda s: np.hypot(*(self.pos[s] - spot)))
        if kind == "penalty":
            return max(outfield, key=lambda s: self._r(s, "finishing"))
        if kind == "corner":
            takers = [s for s in outfield if self.slot_role[s] in CORNER_TAKERS] or outfield
            return max(takers, key=lambda s: self._r(s, "passing"))
        if kind == "free_kick":
            if gk is not None and pitch.to_rel(spot, self.direction[team])[0] < 20.0:
                return gk
            return max(near[:3], key=lambda s: self._r(s, "passing"))
        return near[0]

    def _dead(self, kind: str, team: int, spot, delay: float, taker: int | None = None) -> None:
        self.state = DEAD
        self.owner = None
        self.flight = None
        self.spell = None
        self.loose_v[:] = 0.0
        if self.pending is not None:
            self.pending.event["outcome"] = self.pending.event["outcome"] or "incomplete"
            self.pending = None
        spot = np.asarray(spot, dtype=float)
        self.ball[:] = (spot[0], spot[1], 0.0)
        self._set_possession(team, kind)
        if taker is None:
            taker = self._pick_taker(kind, team, spot)
        self.restart = Restart(kind, team, spot, taker, self.frame + int(delay * FPS))
        self.run_until[:] = np.minimum(self.run_until, self.frame)
        self._maybe_substitute()

    def _update_dead(self) -> None:
        r = self.restart
        if self.frame < r.resume_frame:
            return
        if r.taker is None or not self.active[r.taker]:
            r.taker = self._pick_taker(r.kind, r.team, r.spot)
        t = r.taker
        if np.hypot(*(self.pos[t] - r.spot)) > 1.5 and self.frame < r.resume_frame + 60:
            return
        self.pos[t] = r.spot
        self.vel[t] = 0.0
        self.restart = None
        self.state = OWNED
        self.owner = t
        self.spell = Spell(t, self.frame, r.spot.copy(), self.frame, self.frame + 10_000)
        self._decide(t, restart=r.kind)

    def _maybe_substitute(self) -> None:
        if self.period != 2:
            return
        minute = self.match_time / 60.0
        for team in (0, 1):
            windows = self.sub_windows[team]
            while windows and windows[0][0] <= minute and self.subs_made[team] < 5:
                _, count = windows.pop(0)
                for _ in range(count):
                    self._substitute(team)

    def _substitute(self, team: int) -> None:
        bench = self.bench[team]
        tired = sorted((int(s) for s in self._slots(team) if self.slot_role[s] != "GK"),
                       key=lambda s: -self.covered[s] * (1.3 - self.stamina[s] / 100.0))
        for s in tired:
            options = [p for p in bench if p.group == ROLE_GROUP[self.slot_role[s]]]
            if options:
                break
        else:
            return
        on = options[0]
        off = self.slot_player[s]
        bench.remove(on)
        self.slot_player[s] = on
        self.slot_reg[s] = self.reg_index[on.id]
        self._load_slot(s)
        self.covered[s] = 0.0
        self.subs_made[team] += 1
        self._event("substitution", team=team, player_off=off.id, player_on=on.id)
        if self.restart is not None:
            self.restart.resume_frame += 20 * FPS

    # --------------------------------------------------------------- output

    def _result(self) -> SimulationResult:
        self._event("match_end", frame=self.n - 1, score_home=self.score[0], score_away=self.score[1])
        order = sorted(range(len(self.events)), key=lambda i: (self.events[i]["frame"], i))
        events = [self.events[i] for i in order]
        for k, ev in enumerate(events):
            ev["index"] = k
        n = self.n
        tracking = {
            "t": self.f_t[:n].copy(),
            "period": self.f_period[:n].copy(),
            "state": self.f_state[:n].copy(),
            "owner": self.f_owner[:n].copy(),
            "ball": self.f_ball[:n].copy(),
            "xy": self.f_xy[:n].copy(),
            "player_index": self.f_ids[:n].copy(),
        }
        return SimulationResult(self._metadata(), events, self.truth, tracking)

    def _metadata(self) -> dict:
        def team_meta(idx: int) -> dict:
            t = self.teams[idx]
            return {
                "id": t.id, "name": t.name, "short_name": t.short_name, "colors": t.colors,
                "formation": t.formation, "stadium": t.stadium,
                "style": {k: getattr(t.style, k) for k in t.style.__dataclass_fields__},
                "lineup": [{"slot": idx * 11 + k, "player": p.id, "role": r}
                           for k, (p, r) in enumerate(zip(t.starters, t.roles))],
                "bench": [p.id for p in t.bench],
                "attacks_first_half": "right" if idx == 0 else "left",
            }

        players = []
        for i, p in enumerate(self.registry):
            team_idx = 0 if i < len(self.teams[0].all_players()) else 1
            players.append({
                "index": i, "id": p.id, "name": p.name, "short_name": p.short_name, "number": p.number,
                "position": p.position, "team": self.teams[team_idx].id, "nationality": p.nationality, "foot": p.foot,
            })
        return {
            "schema_version": "1.0",
            "generator": f"matchsense-sim {__version__}",
            "seed": self.seed,
            "competition": "Synthetic Premier Division (fictional clubs and players)",
            "kickoff": self.kickoff,
            "venue": self.teams[0].stadium,
            "referee": self.referee,
            "pitch": {"length": pitch.LENGTH, "width": pitch.WIDTH, "origin": "bottom-left"},
            "fps": FPS,
            "frames": self.n,
            "teams": {"home": team_meta(0), "away": team_meta(1)},
            "players": players,
            "score": {"home": self.score[0], "away": self.score[1]},
            "half_time_score": {"home": self.half_time_score[0], "away": self.half_time_score[1]},
            "storyline": self.storyline.to_dict(),
            "tracking_layout": "xy[frame, slot]; slots 0-10 home, 11-21 away; players[player_index[frame, slot]]",
        }


def simulate(home: TeamSpec, away: TeamSpec, *, seed: int = 0, storyline: str | Storyline = "balanced",
             **kwargs) -> SimulationResult:
    story = STORYLINES[storyline] if isinstance(storyline, str) else storyline
    return MatchSimulator(home, away, seed=seed, storyline=story, **kwargs).run()
