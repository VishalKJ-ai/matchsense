"""Viewer profiles: the same match intelligence, shaped for different fans."""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("casual", "analyst", "player", "club", "audio")

CASUAL_KINDS = frozenset({
    "goal", "big_chance", "save", "tactical_shift", "momentum_swing", "chaos_spell", "control_spell",
    "card", "period_summary", "fastest_sprint", "hard_pass", "added_time",
})


@dataclass(frozen=True)
class Profile:
    """mode: casual | analyst | player (follow one player) | club (a favourite team) | audio (audio description)."""

    mode: str = "casual"
    team: str | None = None
    player: str | None = None
    language: str = "en"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}")

    @property
    def style(self) -> str:
        return "analyst" if self.mode == "analyst" else "casual"

    @property
    def budgets(self) -> dict[str, int]:
        """How many cues each screen layer may show at once."""
        if self.mode == "analyst":
            return {"lower_third": 1, "tag": 3}
        if self.mode == "player":
            return {"lower_third": 1, "tag": 2}
        if self.mode == "audio":
            return {"lower_third": 1, "tag": 0}
        return {"lower_third": 1, "tag": 1}

    def wants(self, signal) -> tuple[bool, float]:
        """Whether this viewer should see the signal, and its on-screen priority (0-140)."""
        priority = signal.significance * 100.0
        if self.mode == "analyst":
            return signal.significance >= 0.2, priority
        involved = self.player is not None and self.player in signal.players
        if self.mode == "player" and involved:
            return True, priority + 40.0
        if signal.kind not in CASUAL_KINDS:
            return False, priority
        f = signal.facts
        if signal.kind == "tactical_shift" and f.get("metric") not in ("press", "possession"):
            return False, priority
        if signal.kind == "hard_pass" and f.get("difficulty", 0) < 80:
            return False, priority
        threshold = 0.45
        if self.mode == "club" and self.team:
            if signal.team == self.team:
                threshold, priority = 0.35, priority + 15.0
            elif signal.team is not None:
                threshold = 0.5
        return signal.significance >= threshold, priority

    @classmethod
    def from_params(cls, params: dict) -> "Profile":
        return cls(mode=params.get("mode") or "casual", team=params.get("team") or None,
                   player=params.get("player") or None, language=params.get("lang") or "en")
