"""Static facts about a match that every analytics component needs."""

from __future__ import annotations

import numpy as np

from ..sim import pitch


def clock_label(period: int, t: float) -> str:
    """Broadcast clock, for example 23' or 45+2'."""
    minute = int(t // 60)
    if period == 1 and minute >= 45:
        return f"45+{minute - 44}'"
    if period == 2 and minute >= 90:
        return f"90+{minute - 89}'"
    return f"{minute + 1}'"


class MatchContext:
    def __init__(self, metadata: dict):
        self.meta = metadata
        teams = metadata["teams"]
        self.home = teams["home"]["id"]
        self.away = teams["away"]["id"]
        self.team_ids = (self.home, self.away)
        self.team_info = {teams["home"]["id"]: teams["home"], teams["away"]["id"]: teams["away"]}
        self.players = {p["id"]: p for p in metadata["players"]}
        self.by_index = {p["index"]: p for p in metadata["players"]}
        self._home_first = 1 if teams["home"].get("attacks_first_half", "right") == "right" else -1

    def team_index(self, team_id: str) -> int:
        return 0 if team_id == self.home else 1

    def other(self, team_id: str) -> str:
        return self.away if team_id == self.home else self.home

    def team_name(self, team_id: str, short: bool = True) -> str:
        info = self.team_info[team_id]
        return info["short_name"] if short else info["name"]

    def direction(self, team_id: str, period: int) -> int:
        """+1 if the team attacks towards x = 105 in this period."""
        d = self._home_first if team_id == self.home else -self._home_first
        return d if period == 1 else -d

    def to_rel(self, xy, team_id: str, period: int) -> np.ndarray:
        return pitch.to_rel(xy, self.direction(team_id, period))

    def slots(self, team_id: str) -> slice:
        return slice(0, 11) if team_id == self.home else slice(11, 22)

    def player_name(self, player_id: str | None, short: bool = True) -> str:
        if player_id is None:
            return ""
        p = self.players[player_id]
        return p["short_name"] if short else p["name"]

    def player_team(self, player_id: str) -> str:
        return self.players[player_id]["team"]
