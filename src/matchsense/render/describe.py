"""Live audio description for blind and partially sighted fans.

Ordinary commentary assumes you can see the pitch. Audio-descriptive commentary
keeps telling you who has the ball, where it is and which way play is going,
and calls every incident. This builds that track live:

- tracking (instant): who has the ball, where on the pitch, possession changes;
- events (about 2 s later): shots and their outcome, goals, set pieces, fouls,
  cards, substitutions;
- signals: occasional tactical context ("Westmarsh are pressing high now").

Each line has a priority. 0 is routine (who and where), 1 is context or a
restart, 2 is an incident, and 3 is a goal. Routine lines are spaced out so
the listener is not flooded, incidents always get through, and goals
interrupt whatever is being said. Text is plain English for now; the narrator
agents and Azure Speech add other languages and natural voices later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..analytics.context import MatchContext
from ..sim import pitch

FPS = 10
MIN_GAP = {0: 4.0, 1: 2.5}  # seconds of match time since the previous line


@dataclass
class Line:
    text: str
    priority: int
    frame: int
    interrupt: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def clock_words(period: int, t: float) -> str:
    minute = int(t // 60)
    if period == 1 and minute >= 45:
        return f"45 plus {minute - 44}"
    if period == 2 and minute >= 90:
        return f"90 plus {minute - 89}"
    return f"{minute + 1} minutes"


class Describer:
    def __init__(self, ctx: MatchContext):
        self.ctx = ctx
        self.last_frame = -10**9
        self.owner = -1
        self.owner_team: str | None = None
        self.owner_since = 0
        self.score = {ctx.home: 0, ctx.away: 0}
        self.next_reminder = 5 * 60.0
        self.period = 0
        self.frame_index = 0

    # ------------------------------------------------------------- helpers

    def zone(self, team: str, period: int, xy) -> str:
        """Where the ball is, from the point of view of the team that has it."""
        x, y = self.ctx.to_rel(xy, team, period)
        opp = self.ctx.team_name(self.ctx.other(team))
        if pitch.in_penalty_area_rel((x, y)):
            return f"in the {opp} box"
        side = " on the right" if y < 20 else " on the left" if y > 48 else ""
        if x > 70:
            return f"in the attacking third{side}"
        if x > 52.5:
            return f"in the {opp} half{side}"
        if x > 35:
            return f"in midfield{side}"
        return f"deep in their own half{side}"

    def _score_words(self) -> str:
        c = self.ctx
        return f"{c.team_name(c.home)} {self.score[c.home]}, {c.team_name(c.away)} {self.score[c.away]}"

    def _emit(self, text: str, priority: int, frame: int, interrupt: bool = False) -> list[Line]:
        if priority < 2 and (frame - self.last_frame) < MIN_GAP[priority] * FPS:
            return []
        self.last_frame = frame
        return [Line(text, priority, frame, interrupt)]

    def _name(self, pid: str | None) -> str:
        return self.ctx.player_name(pid) if pid else "a player"

    # --------------------------------------------------------------- inputs

    def on_frame(self, frame) -> list[Line]:
        self.frame_index = frame.index
        if frame.period != self.period:
            self.period = frame.period
            self.next_reminder = (frame.period - 1) * 2700 + 5 * 60.0
        out: list[Line] = []
        if frame.state == 3 and frame.t >= self.next_reminder:
            self.next_reminder = (frame.t // 300 + 1) * 300
            out += self._emit(f"{clock_words(frame.period, frame.t)}. {self._score_words()}.", 1, frame.index)
            return out
        if frame.state != 0 or frame.owner < 0:
            return out
        pid = self.ctx.by_index[int(frame.player_index[frame.owner])]["id"]
        team = self.ctx.player_team(pid)
        where = self.zone(team, frame.period, frame.ball[:2])
        if frame.owner != self.owner:
            changed_side = self.owner_team is not None and team != self.owner_team
            self.owner, self.owner_team, self.owner_since = frame.owner, team, frame.index
            if changed_side:
                out += self._emit(f"{self.ctx.team_name(team)} win it. {self._name(pid)}, {where}.", 1, frame.index)
            else:
                out += self._emit(f"{self._name(pid)}, {where}.", 0, frame.index)
        elif frame.index - self.last_frame > 7 * FPS:
            out += self._emit(f"{self._name(pid)} still on the ball, {where}.", 0, frame.index)
        return out

    def on_event(self, ev: dict) -> list[Line]:
        etype = ev["type"]
        f = self.frame_index
        team = ev.get("team")
        a = ev.get("attributes", {})
        if etype == "shot":
            shooter = self._name(ev["player"])
            keeper = self._name(a.get("goalkeeper"))
            outcome = {
                "saved": f"saved by {keeper}", "blocked": "blocked", "off_target": "wide of the target",
                "post": "off the post!", "goal": "",
            }[ev["outcome"]]
            if ev["outcome"] == "goal":
                return []  # the goal line follows straight away
            dist = a.get("distance_m")
            return self._emit(f"{shooter} shoots from {dist:.0f} metres... {outcome}.", 2, f, interrupt=True)
        if etype == "goal":
            self.score = {self.ctx.home: a["score_home"], self.ctx.away: a["score_away"]}
            scorer = self.ctx.player_name(a["scorer"], short=False)
            assist = f", set up by {self._name(a['assist'])}" if a.get("assist") else ""
            return self._emit(f"Goal! {self.ctx.team_name(team)} score! {scorer}{assist}. {self._score_words()}.",
                              3, f, interrupt=True)
        if etype == "ball_out":
            restart = a.get("restart")
            # the event's team touched the ball last; the other side takes every restart
            name = self.ctx.team_name(self.ctx.other(team))
            text = {"throw_in": f"Out for a throw-in to {name}.", "corner": f"Corner to {name}.",
                    "goal_kick": f"Goal kick to {name}."}.get(restart)
            return self._emit(text, 1, f) if text else []
        if etype == "foul":
            victim_team = self.ctx.other(team)
            if a.get("in_box"):
                return self._emit(f"Penalty to {self.ctx.team_name(victim_team)}! {self._name(ev['player'])} brings down "
                                  f"{self._name(a.get('victim'))}.", 2, f, interrupt=True)
            return self._emit(f"Foul by {self._name(ev['player'])}. Free kick to {self.ctx.team_name(victim_team)}.", 1, f)
        if etype == "card":
            word = {"yellow": "Yellow card", "second_yellow": "Second yellow card, he's sent off", "red": "Red card, he's sent off"}
            return self._emit(f"{word[a['card']]}: {self.ctx.player_name(ev['player'], short=False)} of "
                              f"{self.ctx.team_name(team)}.", 2, f)
        if etype == "offside":
            return self._emit(f"Flag up, {self._name(ev['player'])} is offside.", 1, f)
        if etype == "substitution":
            return self._emit(f"Substitution for {self.ctx.team_name(team)}: {self._name(a['player_on'])} comes on "
                              f"for {self._name(a['player_off'])}.", 1, f)
        if etype == "added_time":
            return self._emit(f"{a['minutes']} minutes of added time.", 1, f)
        if etype == "period_start":
            return self._emit("We're under way." if ev["period"] == 1 else "The second half is under way.", 2, f)
        if etype == "period_end":
            word = "Half-time" if ev["period"] == 1 else "Full-time"
            return self._emit(f"{word}. {self._score_words()}.", 2, f, interrupt=True)
        return []

    def on_signal(self, s) -> list[Line]:
        f = self.frame_index
        team = self.ctx.team_name(s.team) if s.team else ""
        if s.kind == "tactical_shift" and s.facts.get("metric") == "press":
            text = (f"{team} have pushed right up and are pressing high now."
                    if s.facts["pressing_harder"] else f"{team} have dropped deeper and are sitting back.")
            return self._emit(text, 1, f)
        if s.kind == "momentum_swing":
            return self._emit(f"{team} are on top at the moment.", 1, f)
        if s.kind == "save" and s.facts.get("xg_faced", 0) >= 0.25:
            return self._emit(f"What a save from {self._name(s.players[0])}!", 2, f)
        return []
