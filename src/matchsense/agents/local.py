"""Deterministic explanations and recaps: the crew's local fallback.

These rules read the same fact pack the language model would, and say why a
moment matters with cause-and-effect links the numbers support. They run
instantly with no network, so every viewer gets an explanation even when
Foundry is unavailable. With Foundry configured they are replaced by richer
text, but they never go away.
"""

from __future__ import annotations

from ..render.templates import odds


def _team(pack: dict, which: str = "team") -> str:
    return pack.get(which) or ""


def explain(pack: dict) -> str | None:
    kind = pack["kind"]
    f = pack["facts"]
    team, opp = _team(pack), _team(pack, "opponent")
    ctx = pack.get("context", {})
    poss = ctx.get("possession_last_10", {})
    if kind == "goal":
        if f.get("against_run_of_play"):
            return f"It comes against the run of play: {opp} had been creating more of the danger."
        if f.get("situation") == "equaliser" and f.get("late"):
            return f"A late equaliser: {opp}'s lead is gone with little time left."
        if f.get("situation") == "takes the lead" and f.get("late"):
            return f"A late lead: {opp} now have only minutes to respond."
        if f.get("passes_in_move", 0) >= 6:
            return f"It ends a patient move of {f['passes_in_move']} passes."
        if team in poss:
            return f"{team} have had {poss[team]:.0f}% of the ball over the last 10 minutes."
        return None
    if kind == "tactical_shift":
        c = f.get("context", {})
        if f.get("metric") == "press":
            if f.get("pressing_harder"):
                return (f"Pressing higher wins the ball nearer goal: {opp}'s pass completion is "
                        f"{c.get('opponent_pass_completion_after', 0):.0f}%, from "
                        f"{c.get('opponent_pass_completion_before', 0):.0f}%.")
            return (f"Sitting deeper hands over the ball: {team} have had {c.get('possession_after', 0):.0f}% of it, "
                    f"from {c.get('possession_before', 0):.0f}%.")
        if f.get("metric") == "possession":
            return f"The balance of the game has shifted towards {team if f.get('change', 0) > 0 else opp}."
        return None
    if kind == "big_chance":
        return f"Chances like that are {odds(f.get('xg'))}, so {team} will feel they should be ahead."
    if kind == "save":
        score = f.get("score")
        if isinstance(score, dict) and len(score) == 2:
            (h, hs), (a, as_) = score.items()
            return f"It keeps the score at {h} {hs}-{as_} {a}."
        return None
    if kind == "momentum_swing":
        return f"{team} are creating more threat now than at any point in the last few minutes."
    if kind == "chaos_spell":
        return f"Neither side can keep the ball: possessions are lasting {f.get('possession_s', 0):.0f} seconds on average."
    if kind == "control_spell":
        return f"{team} are setting the tempo, with possessions averaging {f.get('possession_s', 0):.0f} seconds."
    if kind == "card" and f.get("card") in ("red", "second_yellow"):
        return f"{team} must play the rest of the match with ten men."
    return None


def recap(timeline: list[dict], stats: dict, teams: dict[str, str], score_line: str, period: int) -> dict:
    """A short, factual recap from the key moments and statistics."""
    home, away = list(teams)
    h, a = stats.get(home, {}), stats.get(away, {})
    goals = [m for m in timeline if m["kind"] == "goal"]
    shifts = [m for m in timeline if m["kind"] == "tactical_shift" and m["facts"].get("metric") == "press"]
    reds = [m for m in timeline if m["kind"] == "card" and m["facts"].get("card") in ("red", "second_yellow")]
    controller = home if h.get("possession", 50) >= a.get("possession", 50) else away
    other = away if controller == home else home
    parts = [f"{teams[controller]} had {stats[controller]['possession']:.0f}% of the ball, with xG "
             f"{h.get('xg', 0):.1f} to {a.get('xg', 0):.1f}."] if stats else []
    if goals:
        scorers = ", ".join(f"{g['players'][0]} ({g['clock']})" for g in goals if g["players"])
        parts.append(f"Goals: {scorers}.")
    else:
        parts.append("There have been no goals.")
    for r in reds:
        parts.append(f"{r['players'][0] if r['players'] else 'A player'} was sent off on {r['clock']}.")
    for s in shifts:
        verb = "pushed up and pressed higher" if s["facts"].get("pressing_harder") else "dropped deeper"
        parts.append(f"{teams.get(s['team'], s['team'])} {verb} around {s['clock']}.")
    word = "Half-time" if period == 1 else "Full-time"
    return {"headline": f"{word}: {score_line}", "recap": " ".join(parts[:5])}
