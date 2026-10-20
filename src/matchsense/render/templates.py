"""On-screen text for each signal, in two registers.

casual: plain words, with probabilities turned into everyday odds.
analyst: compact numbers with the method in brackets.

Templates only use values from the signal's facts, so every number on screen
can be traced back to the data. They are the fast, always-available path; the
narrator agents can rewrite and translate them when there is time.
"""

from __future__ import annotations

from ..analytics.context import MatchContext


def odds(p: float | None) -> str:
    """0.32 -> 'about 1 in 3'."""
    if not p:
        return "a rare chance"
    if p >= 0.5:
        return "scored more often than not"
    return f"scored about 1 in {max(2, round(1 / p))} times"


MEASURE_LABEL = {
    "high_line": ("line height", "m"),
    "press_gap": ("press distance", "m"),
    "possession": ("possession", "%"),
    "tempo": ("passes per minute on the ball", ""),
    "directness": ("metres gained per pass", "m"),
}


class Templates:
    def __init__(self, ctx: MatchContext):
        self.ctx = ctx

    def name(self, pid: str | None, full: bool = False) -> str:
        return self.ctx.player_name(pid, short=not full) if pid else ""

    def team(self, tid: str | None) -> str:
        return self.ctx.team_name(tid) if tid else ""

    def score_line(self, score: dict) -> str:
        c = self.ctx
        return f"{c.team_name(c.home)} {score[c.home]}-{score[c.away]} {c.team_name(c.away)}"

    def render(self, s, style: str, perspective: str | None = None) -> dict | None:
        fn = getattr(self, f"_{s.kind}", None)
        if fn is None:
            return None
        title, body = fn(s, s.facts, style == "analyst", perspective)
        return {"title": title, "body": body}

    # ----------------------------------------------------------- big moments

    def _goal(self, s, f, analyst, perspective):
        team = self.team(s.team)
        scorer = self.name(f["scorer"], full=True)
        if analyst:
            assist = f" · assist {self.name(f['assist'])}" if f.get("assist") else ""
            move = f" · {f['passes_in_move']} passes" + (f" in {f['move_duration_s']:.0f} s" if f.get("move_duration_s") else "")
            return (f"GOAL · {team} · {s.clock}",
                    f"{scorer}{assist} · xG {f['xg']:.2f} · {f['shot_speed_kmh']:.0f} km/h · {f['distance_m']:.0f} m{move}")
        situation = {"equaliser": f"{team} are level", "takes the lead": f"{team} lead",
                     "pulls one back": f"{team} pull one back", "extends the lead": f"{team} extend the lead"}[f["situation"]]
        extra = " Against the run of play." if f.get("against_run_of_play") else ""
        extra += " A late goal!" if f.get("late") else ""
        if perspective and perspective != s.team:
            return ("Goal", f"{scorer} scores for {team}. {self.score_line(f['score'])}.")
        title = "GOAL!" if perspective != s.team else f"GOAL! {team.upper()}!"
        return (title, f"{scorer}, {s.clock}. {situation}: {self.score_line(f['score'])}.{extra}")

    def _tactical_shift(self, s, f, analyst, perspective):
        team = self.team(s.team)
        label, unit = MEASURE_LABEL.get(f["measure"], (f["measure"], ""))
        c = f["context"]
        if analyst:
            opp = self.team(self.ctx.other(s.team))
            return (f"TACTICAL SHIFT · {team}",
                    f"{f['label']}: {label} {f['before']}{unit} → {f['after']}{unit} "
                    f"(last {f['recent_minutes']}' vs prior {f['baseline_minutes']}') · possession "
                    f"{c['possession_before']:.0f}% → {c['possession_after']:.0f}% · {opp} pass completion "
                    f"{c['opponent_pass_completion_before']:.0f}% → {c['opponent_pass_completion_after']:.0f}%")
        if f["metric"] == "press":
            if f["pressing_harder"]:
                detail = (f"Their back line is {abs(f['change']):.0f} m further up the pitch than earlier."
                          if f["measure"] == "high_line" else
                          f"Their nearest player is now {f['after']:.1f} m from the ball, down from {f['before']:.1f} m.")
                return (f"{team} turn up the pressure", f"{team} have pushed up and are pressing much higher. {detail}")
            return (f"{team} sit deeper", f"{team} have dropped back and stopped pressing high. "
                    f"Possession {c['possession_before']:.0f}% → {c['possession_after']:.0f}%.")
        more = f["change"] > 0
        return (f"{team} {'take over' if more else 'lose their grip'}",
                f"{team} have had {f['after']:.0f}% of the ball recently, {'up' if more else 'down'} from {f['before']:.0f}%.")

    def _big_chance(self, s, f, analyst, perspective):
        shooter = self.name(s.players[0], full=True)
        if analyst:
            return ("BIG CHANCE", f"{self.name(s.players[0])} · xG {f['xg']:.2f} · {f['distance_m']:.0f} m · "
                                  f"{(f.get('body_part') or '').replace('_', ' ')} · {f['outcome'].replace('_', ' ')}")
        how = {"saved": "denied by the keeper", "post": "off the post", "off_target": "misses the target",
               "blocked": "blocked on the line"}.get(f["outcome"], f["outcome"])
        return ("Big chance!", f"{shooter} {how} from {f['distance_m']:.0f} m: a chance {odds(f['xg'])}.")

    def _save(self, s, f, analyst, perspective):
        keeper, shooter = self.name(s.players[0], full=True), self.name(s.players[1])
        if analyst:
            return ("SAVE", f"{self.name(s.players[0])} · xG faced {f['xg_faced']:.2f} · {f['shot_speed_kmh']:.0f} km/h · "
                            f"{'held' if f['held'] else 'parried'}")
        return (f"Great save, {keeper}!", f"{shooter}'s shot from {f['distance_m']:.0f} m was a chance {odds(f['xg_faced'])}.")

    def _period_summary(self, s, f, analyst, perspective):
        c = self.ctx
        title = "Half-time" if f["period"] == 1 else "Full-time"
        scope = "this half" if f["period"] == 1 else "over the match"
        teams = f.get("teams", {}) if f["period"] == 1 else f.get("match_teams") or f.get("teams", {})
        if not teams:
            return (title, self.score_line(f["score"]))
        h, a = teams[c.home], teams[c.away]
        if analyst:
            return (f"{title.upper()} · {self.score_line(f['score'])}",
                    f"xG {h['xg']:.2f}-{a['xg']:.2f} · shots {h['shots']}-{a['shots']} · possession "
                    f"{h['possession']:.0f}-{a['possession']:.0f}% · pass completion {h['pass_completion']:.0f}-{a['pass_completion']:.0f}% "
                    f"({scope})")
        leader = c.home if h["possession"] >= a["possession"] else c.away
        return (title, f"{self.score_line(f['score'])}. {self.team(leader)} had {teams[leader]['possession']:.0f}% "
                       f"of the ball {scope}, with xG {h['xg']:.1f} to {a['xg']:.1f}.")

    # ------------------------------------------------------ flow of the game

    def _momentum_swing(self, s, f, analyst, perspective):
        if analyst:
            return ("MOMENTUM SWING", f"{self.team(s.team)} · {f['momentum_before']:+d} → {f['momentum_now']:+d} "
                                      f"(balance of threat created, 3-minute half-life)")
        return ("Momentum shift", f"The game has swung towards {self.team(s.team)}: they're creating the danger now.")

    def _chaos_spell(self, s, f, analyst, perspective):
        if analyst:
            return ("CHAOS", f"index {f['chaos_index']} · {f['turnovers_per_min']:.1f} turnovers/min · "
                             f"{f['duels_per_min']:.1f} duels/min · {f['possession_s']:.0f} s per possession")
        return ("End to end", "It's frantic: the ball keeps changing hands and nobody can settle.")

    def _control_spell(self, s, f, analyst, perspective):
        team = self.team(s.team)
        if analyst:
            return ("CONTROL", f"index {f['chaos_index']} · {team} possession {f['possession']:.0f}% · "
                               f"{f['possession_s']:.0f} s per possession")
        return (f"{team} in control", f"{team} are keeping the ball for long spells: {f['possession']:.0f}% of it in the last 5 minutes.")

    # ------------------------------------------------------- tags and cards

    def _hard_pass(self, s, f, analyst, perspective):
        passer, receiver = self.name(s.players[0]), self.name(s.players[1])
        reasons = ", ".join(f["reasons"]) if f.get("reasons") else "tight window"
        if analyst:
            return (f"PASS DIFFICULTY {f['difficulty']}", f"{passer} → {receiver} · {reasons} · {f['length_m']:.0f} m")
        return ("What a pass!", f"{passer} to {receiver}: only {f['completion_chance_pct']}% of passes like that arrive.")

    def _key_pass(self, s, f, analyst, perspective):
        if analyst:
            return ("KEY PASS", f"{self.name(s.players[0])} → {self.name(s.players[1])} · shot xG {f['shot_xg']:.2f}")
        return ("Chance created", f"{self.name(s.players[0])} sets up {self.name(s.players[1])}: "
                                  f"a chance {odds(f['shot_xg'])}.")

    def _speed_burst(self, s, f, analyst, perspective):
        return (f"{f['speed_kmh']:.1f} km/h", self.name(s.players[0]))

    def _fastest_sprint(self, s, f, analyst, perspective):
        return (f"{f['speed_kmh']:.1f} km/h", f"{self.name(s.players[0])}: fastest sprint of the match")

    def _shot_speed(self, s, f, analyst, perspective):
        return (f"{f['speed_kmh']:.0f} km/h", f"shot by {self.name(s.players[0])}")

    def _distance_milestone(self, s, f, analyst, perspective):
        return (f"{f['distance_km']:.0f} km", f"{self.name(s.players[0])} has covered {f['distance_km']:.0f} km")

    def _pass_milestone(self, s, f, analyst, perspective):
        return (f"{f['completed_passes']} passes", f"{self.name(s.players[0])}: {f['accuracy_pct']:.0f}% accurate")

    def _card(self, s, f, analyst, perspective):
        word = {"yellow": "Yellow card", "second_yellow": "Second yellow: sent off", "red": "Red card"}[f["card"]]
        return (word, f"{self.name(s.players[0], full=True)} ({self.team(s.team)})")

    def _substitution(self, s, f, analyst, perspective):
        body = f"On: {self.name(f['player_on'])} · Off: {self.name(f['player_off'])}"
        if analyst:
            body += f" ({f['off_distance_km']:.1f} km covered)"
        return (f"Substitution · {self.team(s.team)}", body)

    def _added_time(self, s, f, analyst, perspective):
        return (f"+{f['minutes']}", f"{f['minutes']} minutes added")
