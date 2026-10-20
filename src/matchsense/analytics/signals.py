"""Signals: the facts the analytics layer hands to the explanation layer.

A signal is one noteworthy, verifiable thing: a goal, a hard pass completed,
a sprint over 32 km/h, a change in how a team presses. Every number a
narrator may quote is in `facts`, and `evidence` lists the event ids it came
from. Agents write words; signals supply the numbers, and a fact-checker can
confirm that every number in the words appears here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

KINDS = (
    "goal", "big_chance", "save", "shot_speed", "hard_pass", "key_pass",
    "speed_burst", "fastest_sprint", "distance_milestone", "pass_milestone",
    "tactical_shift", "momentum_swing", "chaos_spell", "control_spell",
    "card", "substitution", "added_time", "period_summary",
)


@dataclass
class Signal:
    id: str
    kind: str
    frame: int
    t: float
    period: int
    clock: str
    team: str | None
    players: list[str]
    facts: dict
    evidence: list[str] = field(default_factory=list)
    significance: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class SignalFactory:
    def __init__(self) -> None:
        self._n = 0

    def make(self, kind: str, *, frame: int, t: float, period: int, clock: str, team: str | None = None,
             players: list[str] | None = None, facts: dict | None = None, evidence: list[str] | None = None,
             significance: float = 0.5) -> Signal:
        assert kind in KINDS, kind
        self._n += 1
        return Signal(f"sg{self._n:04d}", kind, frame, round(t, 1), period, clock, team, players or [],
                      facts or {}, evidence or [], round(min(max(significance, 0.0), 1.0), 3))
