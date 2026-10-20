"""The match event contract shared by every stage of the pipeline.

Events follow the conventions of commercial event feeds: one record per
on-ball action or match incident, ordered by time, with pitch locations in
metres (origin bottom-left, 105 x 68). `frame` is the tracking frame (10 Hz)
the event starts on, and is the key used to synchronise overlays with tracking.
Type-specific fields live in `attributes`; EVENT_ATTRIBUTES documents them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "period_start", "period_end", "added_time", "match_end",
    "pass", "carry", "dribble", "pressure", "tackle", "interception", "ball_recovery",
    "clearance", "block", "aerial_duel", "shot", "goalkeeper", "goal",
    "foul", "card", "offside", "ball_out", "substitution", "possession_change",
]

EVENT_ATTRIBUTES: dict[str, str] = {
    "pass": "recipient, length_m, height (ground|high), kind (pass|through|cross|clearance), set_piece, "
            "body_part, speed_ms (initial ball speed), shot_assist, goal_assist",
    "carry": "length_m, duration_s, end_frame",
    "dribble": "opponent",
    "pressure": "opponent (player being pressed)",
    "tackle": "opponent; outcome won|lost",
    "interception": "pass_event",
    "shot": "xg, body_part, speed_ms, distance_m, angle_deg, set_piece, goalkeeper; "
            "outcome goal|saved|blocked|off_target|post",
    "goalkeeper": "action save|claim, shot_event; outcome caught|parried",
    "goal": "scorer, assist, score_home, score_away, shot_event, own_goal",
    "foul": "victim, in_box, advantage",
    "card": "card yellow|second_yellow|red",
    "offside": "pass_event",
    "ball_out": "restart (throw_in|corner|goal_kick), last_touch_team",
    "substitution": "player_off, player_on",
    "possession_change": "from_team, cause",
    "added_time": "minutes",
    "period_end": "score_home, score_away",
}


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: float
    y: float


class MatchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    index: int = Field(description="Position in the time-ordered event stream")
    frame: int = Field(description="Tracking frame (10 Hz) the event starts on")
    period: int
    t: float = Field(description="Match clock in seconds; the second half starts at 2700")
    minute: int
    second: int
    type: EventType
    team: str | None = None
    player: str | None = None
    possession: int
    location: Point | None = None
    end_location: Point | None = None
    outcome: str | None = None
    under_pressure: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)
    related: list[str] = Field(default_factory=list)

    def clock_label(self) -> str:
        """Broadcast-style clock, for example 23' or 45+2'."""
        if self.period == 1 and self.minute >= 45:
            return f"45+{self.minute - 44}'"
        if self.period == 2 and self.minute >= 90:
            return f"90+{self.minute - 89}'"
        return f"{self.minute + 1}'"
