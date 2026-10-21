"""One viewer's live session: the whole pipeline, paced in real time, streamed over a websocket.

ingest (feed replay) -> interpret (LiveAnalytics) -> render (Producer, Describer) -> browser

Each connected viewer gets their own producer and describer, so the same match
intelligence reaches every profile differently. Message types sent to the client:

    init    teams, players, colours, frame count            (once)
    f       tracking frame: ball, 22 player positions, state
    roster  which player is in each of the 22 slots          (on change)
    state   clock, score, momentum, possession, chaos, ball carrier   (every second)
    ev      ticker events: goals, shots, cards, substitutions
    sig     analytics signals (for the analyst log)
    cue     an overlay cue to show (see render.producer.CUE_SCHEMA), with a "why it matters" line
    cue_text  improved text for a cue from the narration crew on Microsoft Foundry
    cue_end an earlier cue removed early (pre-empted)
    recap   a half-time or full-time recap
    say     an audio-description line
    end     the match is over
"""

from __future__ import annotations

import asyncio
import json
import math

from ..agents.crew import NarrationCrew
from ..agents.prompts import LANGUAGES
from ..analytics.context import clock_label
from ..analytics.engine import LiveAnalytics
from ..live.feed import replay
from ..render.describe import Describer
from ..render.producer import LAYER, Producer
from ..render.profiles import Profile
from ..sim.io import StoredMatch

FPS = 10
END_WAIT_S = 15.0  # how long full time waits for the crew's last captions and recap
CATCH_UP = asyncio.Semaphore(1)  # viewers joining mid-match catch up one at a time
TICKER = {"goal", "shot", "card", "substitution", "period_start", "period_end", "added_time", "offside"}


def timeline_pct(period: int, t: float) -> float:
    """Position on a 0-100 match timeline where each half gets 50 units (45 minutes plus up to 5 added)."""
    in_half = t - 2700 if period == 2 else t
    return round((period - 1) * 50 + min(in_half / 60.0, 50.0), 2)


def _r(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 1)


class LiveSession:
    def __init__(self, match: StoredMatch, profile: Profile, audio: bool = False, speed: float = 8.0,
                 crew: NarrationCrew | None = None):
        self.match = match
        self.analytics = LiveAnalytics(match.metadata)
        self.ctx = self.analytics.ctx
        self.crew = crew or NarrationCrew(self.ctx)
        self._by_id: dict = {}
        self._jobs: list[tuple] = []
        self._tasks: set = set()
        self.set_profile(profile, audio)
        self.speed = speed
        self.paused = False
        self._roster = None
        self._resync = True

    def set_profile(self, profile: Profile, audio: bool) -> None:
        self.profile = profile
        self.producer = Producer(self.ctx, profile)
        self.describer = Describer(self.ctx) if (audio or profile.mode == "audio") else None

    # --------------------------------------------------------------- messages

    def init_message(self) -> dict:
        meta = self.match.metadata
        return {
            "k": "init",
            "match": {"venue": meta["venue"], "kickoff": meta["kickoff"], "competition": meta["competition"],
                      "referee": meta["referee"], "frames": meta["frames"], "fps": meta["fps"]},
            "teams": {side: {k: meta["teams"][side][k] for k in ("id", "name", "short_name", "colors", "formation")}
                      for side in ("home", "away")},
            "players": [{k: p[k] for k in ("index", "id", "name", "short_name", "number", "position", "team")}
                        for p in meta["players"]],
            "profile": {"mode": self.profile.mode, "team": self.profile.team, "player": self.profile.player,
                        "language": self.profile.language},
            "narration": {"remote": self.crew.remote,
                          "languages": list(LANGUAGES) if self.crew.remote else ["en"]},
            "history": self.history(),
        }

    def history(self) -> dict:
        """Momentum and timeline markers so far, for viewers who join or jump mid-match."""
        buckets = self.analytics.tactical.buckets
        momentum = [[timeline_pct(buckets[i].period, buckets[i].t), v]
                    for i, v in self.analytics.tactical.momentum_series]
        markers = []
        for s in self.analytics.emitted:
            x = timeline_pct(s.period, s.t)
            if s.kind == "goal":
                markers.append({"kind": "goal", "x": x, "team": s.team,
                                "side": "home" if s.team == self.ctx.home else "away"})
            elif s.kind == "card":
                markers.append({"kind": "booking", "x": x, "red": s.facts["card"] != "yellow"})
            elif s.kind == "tactical_shift" and s.facts.get("metric") == "press":
                markers.append({"kind": "shift", "x": x, "team": s.team})
        moments, signals = [], []
        perspective = self.profile.team if self.profile.mode == "club" else None
        for s in self.analytics.emitted:
            if s.significance >= 0.3:
                signals.append(s.to_dict())
            wanted, _ = self.profile.wants(s)
            if not wanted or LAYER.get(s.kind) is None:
                continue
            if LAYER[s.kind] == "tag" and self.profile.mode != "player":
                continue
            text = self.producer.templates.render(s, self.profile.style, perspective=perspective)
            if text:
                moments.append({"clock": s.clock, **text})
        return {"momentum": momentum, "markers": markers, "moments": moments, "signals": signals}

    def frame_messages(self, frame, send_frame: bool) -> list[dict]:
        out: list[dict] = []
        signals = self.analytics.on_frame(frame)
        roster = frame.player_index.tolist()
        if roster != self._roster:
            self._roster = roster
            out.append({"k": "roster", "slots": roster})
        if send_frame:
            xy = [_r(v) for v in frame.xy.reshape(-1).tolist()]
            out.append({"k": "f", "f": frame.index, "t": round(frame.t, 1), "p": frame.period, "s": frame.state,
                        "o": frame.owner, "b": [_r(v) for v in frame.ball.tolist()], "xy": xy})
        out += self._signals(signals, frame.index)
        out += [self._decorate(m) for m in self.producer.tick(frame.index)]
        if self.describer:
            out += self._say(self.describer.on_frame(frame))
        if frame.index % FPS == 0:
            out.append(self.state_message())
        return out

    def event_messages(self, ev: dict) -> list[dict]:
        out: list[dict] = []
        signals = self.analytics.on_event(ev)
        if ev["type"] in TICKER:
            out.append({"k": "ev", "id": ev["id"], "type": ev["type"], "clock": self._clock(ev), "team": ev["team"],
                        "p": ev["period"], "t": ev["t"],
                        "player": ev["player"], "outcome": ev["outcome"],
                        "attributes": {k: v for k, v in ev["attributes"].items()
                                       if k in ("xg", "card", "player_on", "player_off", "minutes", "score_home",
                                                "score_away", "assist", "speed_ms")}})
        now = self.analytics.frame.index if self.analytics.frame else ev["frame"]
        out += self._signals(signals, now)
        if self.describer:
            out += self._say(self.describer.on_event(ev))
        return out

    def _signals(self, signals, now: int) -> list[dict]:
        out = []
        for s in signals:
            self._by_id[s.id] = s
            if s.significance >= 0.3:
                out.append({"k": "sig", **s.to_dict()})
            out += [self._decorate(m) for m in self.producer.on_signal(s, now)]
            if self.describer:
                out += self._say(self.describer.on_signal(s))
            if s.kind == "period_summary":
                out.append(self._recap(s))
        return out

    def _decorate(self, msg: dict) -> dict:
        """Add the instant "why it matters" line to a lower third, and queue the crew's improved text."""
        if msg.get("k") != "cue" or msg.get("layer") != "lower_third":
            return msg
        s = self._by_id.get(msg["signal"])
        if s is None:
            return msg
        snapshot = self.analytics.snapshot(window_minutes=10)
        msg["why"] = self.crew.explain_now(s, snapshot)
        msg["source"] = "local"
        if self.crew.remote:
            self._jobs.append(("narrate", msg["id"], s, snapshot, {"title": msg["title"], "body": msg["body"]},
                               self.profile))
        return msg

    def _recap(self, s) -> dict:
        period = s.facts.get("period", 1)
        snapshot = self.analytics.snapshot(window_minutes=10)
        signals = list(self.analytics.emitted)
        local = self.crew.recap(signals, snapshot, self.profile, period, local_only=True)
        if self.crew.remote:
            self._jobs.append(("recap", period, signals, snapshot, self.profile))
        return {"k": "recap", "period": period, "clock": s.clock, **local}

    async def _run_job(self, send, job: tuple) -> None:
        loop = asyncio.get_running_loop()
        try:
            if job[0] == "narrate":
                _, cue_id, s, snapshot, template, profile = job
                n = await loop.run_in_executor(None, self.crew.narrate, s, snapshot, profile, template)
                if n.source == "foundry":
                    await send({"k": "cue_text", "id": cue_id, "title": n.title, "body": n.body, "why": n.why,
                                "language": n.language, "source": n.source, "attempts": n.attempts,
                                "latency_ms": n.latency_ms})
            else:
                _, period, signals, snapshot, profile = job
                r = await loop.run_in_executor(None, self.crew.recap, signals, snapshot, profile, period)
                if r.get("source") == "foundry":
                    await send({"k": "recap", "period": period, **r})
        except Exception:  # noqa: BLE001 - narration is an enhancement; the broadcast carries on without it
            pass

    def _start_jobs(self, send) -> None:
        for job in self._jobs:
            task = asyncio.create_task(self._run_job(send, job))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._jobs.clear()

    def _say(self, lines) -> list[dict]:
        f = self.analytics.frame
        clock = clock_label(f.period, f.t) if f else ""
        return [{"k": "say", "clock": clock, **line.to_dict()} for line in lines]

    def _clock(self, ev: dict) -> str:
        return clock_label(ev["period"], ev["t"])

    def state_message(self) -> dict:
        snap = self.analytics.snapshot(window_minutes=10)
        recent, so_far = snap["recent"], snap["match_so_far"]
        def team_view(w):
            return {tid: {k: v.get(k) for k in ("possession", "xg", "shots", "pass_completion", "ppda", "line_height",
                                                  "press_gap", "passes")}
                    for tid, v in w.get("teams", {}).items()}
        return {
            "k": "state", "clock": snap["clock"], "period": snap["period"], "score": snap["score"],
            "momentum": snap["momentum"], "carrier": snap["ball_carrier"],
            "recent": {"teams": team_view(recent), "chaos": recent.get("match", {}).get("chaos"),
                       "rhythm": recent.get("match", {}).get("rhythm")},
            "match": {"teams": team_view(so_far)},
        }

    def focus_message(self) -> dict | None:
        pid = self.profile.player
        if not pid:
            return None
        table = self.analytics.player_table()
        return {"k": "focus", "player": pid, "stats": table.get(pid)}

    # ------------------------------------------------------------------ loop

    def fast_forward(self, feed, start_frame: int) -> None:
        """Run the analytics up to the start point without producing output."""
        if start_frame <= 0:
            return
        self._jobs.clear()
        for kind, item in feed:
            if kind == "frame":
                self.analytics.on_frame(item)
                if item.index >= start_frame:
                    return
            else:
                self.analytics.on_event(item)

    async def run(self, send, start_frame: int = 0) -> None:
        loop = asyncio.get_running_loop()
        feed = replay(self.match)
        # Catching up replays every minute before the start point: CPU-bound, so it runs in a worker
        # thread, one at a time. On the event loop, a few viewers joining mid-match at once stalled health
        # checks long enough for Container Apps to restart the replica; in parallel threads they slowed
        # each other down through the GIL.
        async with CATCH_UP:
            await loop.run_in_executor(None, self.fast_forward, feed, start_frame)
        await send(self.init_message())
        ref_wall, ref_frame = loop.time(), start_frame
        last_sent = -10**9
        for kind, item in feed:
            if kind == "frame":
                while self.paused:
                    await asyncio.sleep(0.1)
                    self._resync = True
                if self._resync:
                    ref_wall, ref_frame, self._resync = loop.time(), item.index, False
                target = ref_wall + (item.index - ref_frame) / (FPS * self.speed)
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -1.0:
                    self._resync = True
                step = max(1, round(self.speed / 4))
                send_frame = item.index - last_sent >= step
                if send_frame:
                    last_sent = item.index
                messages = self.frame_messages(item, send_frame)
                if item.index % (5 * FPS) == 0:
                    focus = self.focus_message()
                    if focus:
                        messages.append(focus)
            else:
                messages = self.event_messages(item)
            for m in messages:
                await send(m)
            if self._jobs:
                self._start_jobs(send)
        await send(self.state_message())
        if self._tasks:  # the full-time recap and last captions are still being written; the socket closes on "end"
            await asyncio.wait(list(self._tasks), timeout=END_WAIT_S)
        await send({"k": "end"})

    def control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "speed":
            self.speed = float(min(max(msg.get("value", 1.0), 0.25), 64.0))
            self._resync = True
        elif kind == "pause":
            self.paused = True
        elif kind == "resume":
            self.paused = False
        elif kind == "profile":
            self.set_profile(Profile.from_params(msg), bool(msg.get("audio")))

    def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()


def dumps(message: dict) -> str:
    return json.dumps(message, ensure_ascii=False, default=float, separators=(",", ":"))
