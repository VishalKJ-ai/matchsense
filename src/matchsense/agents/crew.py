"""The narration crew: specialised agents that share one blackboard.

    signal ──► Tactician ──► Narrator ──► Fact-checker ──► on-air text
               (why it       (one per      (every number      │ ungrounded: one repair
                matters)      audience)     from evidence)    └ still wrong: template
    period end ──► Recapper ──► Fact-checker ──► recap card

- Tactician: explains why a moment matters. Its analysis goes on the blackboard
  keyed by signal, so it runs once per moment however many viewers there are.
- Narrator: writes the graphic for one audience (style, club, focus player,
  language), and avoids repeating that audience's recent lines.
- Fact-checker: deterministic. Every number in the text must appear in the
  fact pack. It hands the narrator one repair request, then falls back.
- Recapper: half-time and full-time stories from the blackboard timeline.

Each step fails soft. With no Foundry configured, the local rules explain and
the templates write. On a timeout, error or rate limit, the local path is used.
If text is still ungrounded after the repair, the template is used. The live
broadcast never waits for a model.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

from ..analytics.context import MatchContext
from . import local
from .factcheck import check_text
from .models import ChatModel, ModelError
from .prompts import (CAPTION_LIMITS, CAPTION_SCHEMA, LANGUAGES, NARRATOR, RECAP_LIMITS, RECAP_SCHEMA,
                      RECAPPER, REPAIR, ROLES, STYLES, TACTICIAN, WHY_SCHEMA)

EXPLAIN_KINDS = {"goal", "tactical_shift", "big_chance", "save", "momentum_swing", "chaos_spell", "control_spell", "card"}
RECAP_KINDS = {"goal", "tactical_shift", "big_chance", "save", "card", "momentum_swing", "substitution"}


def _momentum_word(strength: int) -> str:
    """The 0-100 momentum strength as a word."""
    if strength < 20:
        return "level"
    if strength < 45:
        return "slight"
    if strength < 70:
        return "clear"
    return "strong"


@dataclass
class Narration:
    title: str
    body: str
    why: str | None
    angle: str | None
    language: str
    source: str  # "foundry" or "local"
    attempts: int
    latency_ms: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RateLimiter:
    """At most `per_minute` model calls in any rolling minute."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def take(self) -> bool:
        now = time.monotonic()
        with self.lock:
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.per_minute:
                return False
            self.calls.append(now)
            return True


class Blackboard:
    """State shared by the agents, and by every viewer of the same match."""

    def __init__(self):
        self.analysis: dict[str, dict] = {}  # signal id -> tactician output
        self.captions: dict[tuple, Narration] = {}  # (signal id, audience key) -> narrator output
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=6))  # audience key -> recent lines
        self.lock = threading.Lock()


class NarrationCrew:
    def __init__(self, ctx: MatchContext, model: ChatModel | None = None, blackboard: Blackboard | None = None,
                 calls_per_minute: int = 40):
        self.ctx = ctx
        self.model = model
        self.board = blackboard or Blackboard()
        self.limiter = RateLimiter(calls_per_minute)

    @property
    def remote(self) -> bool:
        return self.model is not None

    # -------------------------------------------------------------- packs

    def _named(self, value):
        """Replace player and team ids with names, so the model sees what viewers see."""
        if isinstance(value, str):
            if value in self.ctx.players:
                return self.ctx.player_name(value, short=False)
            if value in self.ctx.team_info:
                return self.ctx.team_name(value)
            return value
        if isinstance(value, dict):
            return {(self.ctx.team_name(k) if k in self.ctx.team_info else k): self._named(v)
                    for k, v in value.items() if not k.endswith("_xy")}
        if isinstance(value, list):
            return [self._named(v) for v in value]
        return value

    def fact_pack(self, s, snapshot: dict, profile) -> dict:
        c = self.ctx
        recent = snapshot.get("recent", {}).get("teams", {})
        score = snapshot.get("score", {})
        momentum = snapshot.get("momentum", 0)
        pack = {
            "kind": s.kind,
            "clock": s.clock,
            "team": c.team_name(s.team) if s.team else None,
            "opponent": c.team_name(c.other(s.team)) if s.team else None,
            "players": [c.player_name(p, short=False) for p in s.players],
            "facts": self._named(s.facts),
            "context": {
                "score_line": f"{c.team_name(c.home)} {score.get(c.home, 0)}-{score.get(c.away, 0)} {c.team_name(c.away)}",
                "possession_last_10": {c.team_name(t): v["possession"] for t, v in recent.items()},
                # In words: as a 0-100 number, models read it as a probability ("7 in 10").
                "momentum": f"{_momentum_word(abs(momentum))}, towards "
                            f"{c.team_name(c.home if momentum >= 0 else c.away)}",
            },
        }
        if profile is not None:
            pack["audience"] = {
                "style": profile.style,
                "style_guide": STYLES[profile.style],
                "language": LANGUAGES.get(profile.language, "English"),
                "club": c.team_name(profile.team) if profile.mode == "club" and profile.team else None,
                "focus_player": c.player_name(profile.player, short=False) if profile.mode == "player" and profile.player else None,
            }
        return pack

    @staticmethod
    def audience_key(profile) -> str:
        club = profile.team if profile.mode == "club" else ""
        player = profile.player if profile.mode == "player" else ""
        return f"{profile.style}|{club}|{player}|{profile.language}"

    # ------------------------------------------------------------ tactician

    def explain_now(self, s, snapshot: dict) -> str | None:
        """The local explanation: instant and deterministic, available to every viewer."""
        if s.kind not in EXPLAIN_KINDS:
            return None
        return local.explain(self.fact_pack(s, snapshot, None))

    def _analysis(self, s, pack: dict) -> dict:
        with self.board.lock:
            cached = self.board.analysis.get(s.id)
        if cached is not None:
            return cached
        result = {"why": local.explain(pack), "angle": None, "source": "local"}
        if self.remote and s.kind in EXPLAIN_KINDS and self.limiter.take():
            try:
                out = self.model.complete("tactician", TACTICIAN, {k: v for k, v in pack.items() if k != "audience"},
                                          WHY_SCHEMA)
                if check_text({"why": out.get("why", "")}, pack, {"why": 260}).ok:
                    result = {"why": out["why"], "angle": out.get("angle"), "source": "foundry"}
            except (ModelError, Exception):  # noqa: BLE001 - any failure falls back to the local rule
                pass
        with self.board.lock:
            self.board.analysis[s.id] = result
        return result

    # -------------------------------------------------------------- narrator

    def narrate(self, s, snapshot: dict, profile, template: dict) -> Narration:
        start = time.perf_counter()
        key = (s.id, self.audience_key(profile))
        with self.board.lock:
            cached = self.board.captions.get(key)
        if cached is not None:
            return cached
        pack = self.fact_pack(s, snapshot, profile)
        analysis = self._analysis(s, pack)

        def fallback(note: str) -> Narration:
            return Narration(template["title"], template["body"], analysis["why"], analysis.get("angle"), "en",
                             "local", 0, round((time.perf_counter() - start) * 1000, 1), note)

        if not self.remote:
            return fallback("no remote model configured")
        if not self.limiter.take():
            return fallback("rate limited")
        audience = self.audience_key(profile)
        payload = {"fact_pack": pack, "tactician": {"why": analysis["why"], "angle": analysis.get("angle")},
                   "template": template, "recent_lines": list(self.board.recent[audience])}
        evidence = {**pack, "tactician": analysis.get("why") or ""}
        attempts = 0
        try:
            out = self.model.complete("narrator", NARRATOR, payload, CAPTION_SCHEMA)
            attempts = 1
            check = check_text({"title": out.get("title", ""), "body": out.get("body", "")}, evidence, CAPTION_LIMITS)
            if not check.ok and self.limiter.take():
                repair = REPAIR.format(numbers=", ".join(check.ungrounded) or "none",
                                       problems="; ".join(check.problems))
                out = self.model.complete("narrator", NARRATOR, {**payload, "previous": out, "feedback": repair},
                                          CAPTION_SCHEMA)
                attempts = 2
                check = check_text({"title": out.get("title", ""), "body": out.get("body", "")}, evidence,
                                   CAPTION_LIMITS)
            if not check.ok:
                return fallback(f"fact-check failed: {', '.join(check.ungrounded + check.problems)}")
        except (ModelError, Exception) as exc:  # noqa: BLE001
            return fallback(f"model unavailable: {type(exc).__name__}")
        result = Narration(out["title"].strip(), out["body"].strip(), analysis["why"], analysis.get("angle"),
                           profile.language, "foundry", attempts, round((time.perf_counter() - start) * 1000, 1))
        with self.board.lock:
            self.board.captions[key] = result
            self.board.recent[audience].append(result.body)
        return result

    # -------------------------------------------------------------- recapper

    def recap(self, signals: list, snapshot: dict, profile, period: int, local_only: bool = False) -> dict:
        c = self.ctx
        timeline = [{"kind": s.kind, "clock": s.clock, "team": s.team,
                     "players": [c.player_name(p, short=False) for p in s.players], "facts": self._named(s.facts)}
                    for s in signals if s.kind in RECAP_KINDS and s.significance >= 0.4]
        stats = snapshot.get("match_so_far", {}).get("teams", {})
        teams = {c.home: c.team_name(c.home), c.away: c.team_name(c.away)}
        score = snapshot.get("score", {})
        score_line = f"{teams[c.home]} {score.get(c.home, 0)}-{score.get(c.away, 0)} {teams[c.away]}"
        fallback = {**local.recap(timeline, stats, teams, score_line, period), "source": "local"}
        if local_only or not self.remote or not self.limiter.take():
            return fallback
        payload = {
            "period": "half-time" if period == 1 else "full-time",
            "score": score_line,
            "timeline": [{**m, "team": teams.get(m["team"], m["team"])} for m in timeline],
            "statistics": {teams[t]: {k: v for k, v in st.items() if v is not None} for t, st in stats.items()},
            "audience": self.fact_pack(signals[-1], snapshot, profile)["audience"] if signals else {},
        }
        try:
            out = self.model.complete("recapper", RECAPPER, payload, RECAP_SCHEMA)
        except (ModelError, Exception):  # noqa: BLE001
            return fallback
        if not check_text({"headline": out.get("headline", ""), "recap": out.get("recap", "")}, payload, RECAP_LIMITS).ok:
            return fallback
        return {"headline": out["headline"], "recap": out["recap"], "source": "foundry"}


AGENT_ROLES = ROLES  # role -> (instructions, output schema), registered as Foundry prompt agents
