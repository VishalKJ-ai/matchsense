"""The producer: turns signals into timed, machine-readable overlay cues for one viewer profile.

A broadcast graphics operator decides what goes on screen, when, for how long,
and what gets bumped. This does the same with explicit rules, so the output can
be handed to a rendering partner as-is:

- each cue has a layer, start and end frame, priority, anchor and template data;
- each layer has a budget (one lower third at a time; a few player tags);
- a much more important cue pre-empts a lesser one, and a cue that must wait
  is dropped if it goes stale.

The cue format is documented in CUE_SCHEMA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..analytics.context import MatchContext
from .profiles import Profile
from .templates import Templates

FPS = 10
LAYER = {
    "goal": "lower_third", "tactical_shift": "lower_third", "big_chance": "lower_third", "save": "lower_third",
    "momentum_swing": "lower_third", "chaos_spell": "lower_third", "control_spell": "lower_third",
    "card": "lower_third", "substitution": "lower_third", "period_summary": "lower_third",
    "added_time": "lower_third", "key_pass": "lower_third",
    "hard_pass": "tag", "speed_burst": "tag", "fastest_sprint": "tag", "shot_speed": "tag",
    "distance_milestone": "tag", "pass_milestone": "tag",
}
DURATION_S = {
    "goal": 10, "tactical_shift": 11, "big_chance": 7, "save": 6, "momentum_swing": 7, "chaos_spell": 7,
    "control_spell": 7, "card": 5, "substitution": 6, "period_summary": 14, "added_time": 5, "key_pass": 5,
    "hard_pass": 4, "speed_burst": 3, "fastest_sprint": 4, "shot_speed": 4, "distance_milestone": 4,
    "pass_milestone": 4,
}
# Minimum minutes between two cues of the same kind, so the screen does not repeat itself.
KIND_COOLDOWN_MIN = {"momentum_swing": 12, "chaos_spell": 15, "control_spell": 15, "hard_pass": 6}
PREEMPT_MARGIN = 25.0
MAX_WAIT_S = 8.0

CUE_SCHEMA = {
    "id": "unique cue id",
    "kind": "signal kind that produced the cue",
    "layer": "lower_third | tag",
    "template": "graphics template name (same as kind)",
    "start_frame": "tracking frame (10 Hz) to show the cue",
    "end_frame": "tracking frame to remove it",
    "moment_frame": "frame the underlying moment happened, for replays that can pre-roll",
    "priority": "0-140; higher wins the layer",
    "anchor": "{type: screen, region} | {type: player, player} | {type: pitch, x, y}",
    "title": "short headline", "body": "one line of detail",
    "data": "template values: team, players and the signal's facts",
    "signal": "signal id", "evidence": "event ids the numbers come from",
}


@dataclass
class Cue:
    id: str
    kind: str
    layer: str
    template: str
    start_frame: int
    end_frame: int
    moment_frame: int
    priority: float
    anchor: dict
    title: str
    body: str
    clock: str
    team: str | None
    players: list[str]
    data: dict
    signal: str
    evidence: list[str] = field(default_factory=list)
    language: str = "en"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = round(self.priority, 1)
        return d


class Producer:
    def __init__(self, ctx: MatchContext, profile: Profile):
        self.ctx = ctx
        self.profile = profile
        self.templates = Templates(ctx)
        self.active: dict[str, list[Cue]] = {"lower_third": [], "tag": []}
        self.waiting: list[tuple[Cue, int]] = []  # (cue, frame it started waiting)
        self._last_kind: dict[str, int] = {}
        self._n = 0

    def _anchor(self, s) -> dict:
        f = s.facts
        if s.kind == "hard_pass" and f.get("end_xy"):
            return {"type": "pitch", "x": f["end_xy"][0], "y": f["end_xy"][1]}
        if s.kind == "shot_speed" and f.get("shot_xy"):
            return {"type": "pitch", "x": f["shot_xy"][0], "y": f["shot_xy"][1]}
        if LAYER[s.kind] == "tag" and s.players:
            return {"type": "player", "player": s.players[0]}
        return {"type": "screen", "region": "bottom"}

    def build(self, s, now: int) -> Cue | None:
        if s.kind not in LAYER:
            return None
        wanted, priority = self.profile.wants(s)
        if not wanted:
            return None
        if s.kind == "key_pass" and s.facts.get("shot_xg", 0) < 0.08:
            return None
        cooldown = KIND_COOLDOWN_MIN.get(s.kind) if self.profile.mode != "analyst" else None
        if cooldown and now - self._last_kind.get(s.kind, -10**9) < cooldown * 60 * FPS:
            return None
        self._last_kind[s.kind] = now
        text = self.templates.render(s, self.profile.style, perspective=self.profile.team if self.profile.mode == "club" else None)
        if text is None:
            return None
        self._n += 1
        duration = int(DURATION_S[s.kind] * FPS)
        return Cue(
            id=f"cue{self._n:04d}", kind=s.kind, layer=LAYER[s.kind], template=s.kind,
            start_frame=now, end_frame=now + duration, moment_frame=s.frame, priority=priority,
            anchor=self._anchor(s), title=text["title"], body=text["body"], clock=s.clock, team=s.team,
            players=list(s.players), data={"facts": s.facts}, signal=s.id, evidence=list(s.evidence),
            language=self.profile.language,
        )

    def _expire(self, now: int) -> None:
        for layer in self.active:
            self.active[layer] = [c for c in self.active[layer] if c.end_frame > now]

    def _place(self, cue: Cue, now: int, messages: list[dict]) -> bool:
        layer = self.active[cue.layer]
        budget = self.profile.budgets.get(cue.layer, 0)
        if budget == 0:
            return False
        if len(layer) < budget:
            layer.append(cue)
            messages.append({"k": "cue", **cue.to_dict()})
            return True
        weakest = min(layer, key=lambda c: c.priority)
        if cue.priority >= weakest.priority + PREEMPT_MARGIN:
            layer.remove(weakest)
            messages.append({"k": "cue_end", "id": weakest.id, "frame": now})
            layer.append(cue)
            messages.append({"k": "cue", **cue.to_dict()})
            return True
        return False

    def on_signal(self, s, now: int) -> list[dict]:
        self._expire(now)
        cue = self.build(s, now)
        messages: list[dict] = []
        if cue is not None and not self._place(cue, now, messages) and cue.priority >= 40:
            self.waiting.append((cue, now))
        return messages

    def tick(self, now: int) -> list[dict]:
        self._expire(now)
        messages: list[dict] = []
        still = []
        for cue, since in sorted(self.waiting, key=lambda cw: -cw[0].priority):
            if now - since > MAX_WAIT_S * FPS:
                continue
            duration = cue.end_frame - cue.start_frame
            cue.start_frame, cue.end_frame = now, now + duration
            if not self._place(cue, now, messages):
                still.append((cue, since))
        self.waiting = still
        return messages
