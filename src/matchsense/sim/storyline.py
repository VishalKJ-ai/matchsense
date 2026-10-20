"""Storylines: scripted changes in how teams play, plus a test of the final result.

A storyline never forces an event. It changes team styles at set minutes (a
coach switching to a high press, a side tiring, a confidence boost after a
goal) and the simulator plays on with its normal physics. `predicate` lets a
seed search keep only the matches whose result fits the story, such as a
comeback.

The phases are written to the ground-truth file, so the analytics layer can
be checked on whether it spots each tactical shift from the data alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Phase:
    minute: float
    team: int
    overrides: dict[str, float]
    note: str


@dataclass(frozen=True)
class MatchOutline:
    """The parts of a finished match a storyline predicate can test."""

    final_score: tuple[int, int]
    half_time_score: tuple[int, int]
    goals: tuple[tuple[int, int], ...]  # (minute, team)


@dataclass(frozen=True)
class Storyline:
    name: str
    description: str
    phases: tuple[Phase, ...] = ()
    predicate: Callable[[MatchOutline], bool] | None = field(default=None, compare=False)

    def styles_at(self, minute: float, team: int) -> dict[str, float]:
        merged: dict[str, float] = {}
        for phase in self.phases:
            if phase.team == team and phase.minute <= minute:
                merged.update(phase.overrides)
        return merged

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "phases": [
                {"minute": p.minute, "team": ("home", "away")[p.team], "overrides": p.overrides, "note": p.note}
                for p in self.phases
            ],
        }


def _comeback(o: MatchOutline) -> bool:
    ht_home, ht_away = o.half_time_score
    ft_home, ft_away = o.final_score
    return ht_home > ht_away and ft_away > ft_home and sum(o.final_score) >= 3


def _late_drama(o: MatchOutline) -> bool:
    return any(minute >= 85 for minute, _ in o.goals) and sum(o.final_score) >= 3


def _siege(o: MatchOutline) -> bool:
    home, away = o.final_score
    return away >= home and away >= 1


STORYLINES: dict[str, Storyline] = {
    "balanced": Storyline(
        name="balanced",
        description="No scripted changes: two sides playing their natural styles.",
    ),
    "comeback": Storyline(
        name="comeback",
        description=(
            "Ironvale dominate the first half with a high press. At the hour Westmarsh push "
            "their line up and press higher while Ironvale tire and drop off, and the game turns."
        ),
        phases=(
            Phase(0, HOME, {"strength": 4, "tempo": 1.15, "press_line": 88, "press_intensity": 0.8},
                  "Ironvale start fast and press high"),
            Phase(0, AWAY, {"strength": -2, "press_line": 48, "line_height": -6},
                  "Westmarsh sit in a deep block"),
            Phase(58, AWAY, {"strength": 5, "press_line": 82, "press_intensity": 0.85, "line_height": 4,
                             "tempo": 1.15, "directness": 0.55},
                  "Westmarsh push up and press high"),
            Phase(62, HOME, {"strength": -3, "press_line": 55, "press_intensity": 0.4, "tempo": 0.95,
                             "line_height": -4},
                  "Ironvale tire and drop into a mid block"),
        ),
        predicate=_comeback,
    ),
    "siege": Storyline(
        name="siege",
        description=(
            "Ironvale camp in the Westmarsh half for long spells, while Westmarsh defend deep and "
            "break quickly through their wide forwards."
        ),
        phases=(
            Phase(0, HOME, {"strength": 3, "press_line": 90, "press_intensity": 0.85, "width": 1.12, "crossing": 0.7}, "Ironvale press relentlessly"),
            Phase(0, AWAY, {"press_line": 42, "line_height": -9, "directness": 0.85, "tempo": 1.05}, "Westmarsh defend deep and counter"),
        ),
        predicate=_siege,
    ),
    "late_drama": Storyline(
        name="late_drama",
        description="An open game where both sides tire and take more risks in the last 15 minutes.",
        phases=(
            Phase(75, HOME, {"risk": 0.7, "directness": 0.7, "line_height": 5, "tempo": 1.15}, "Ironvale go for it"),
            Phase(75, AWAY, {"risk": 0.7, "directness": 0.75, "line_height": 4, "tempo": 1.15}, "Westmarsh go for it"),
        ),
        predicate=_late_drama,
    ),
}
