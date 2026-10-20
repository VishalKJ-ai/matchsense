"""MatchSense web service: match listings, cue schema, and the live websocket stream.

    matchsense-web --port 8000      then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..agents import AGENT_INSTRUCTIONS, Blackboard, NarrationCrew, model_from_env
from ..analytics.context import MatchContext
from ..render.producer import CUE_SCHEMA
from ..render.profiles import MODES, Profile
from ..sim.io import StoredMatch, load_match
from .session import FPS, LiveSession, dumps

DATA_ROOT = Path(os.environ.get("MATCHSENSE_DATA", Path(__file__).resolve().parents[3] / "data" / "matches"))
STATIC = Path(__file__).with_name("static")

log = logging.getLogger("matchsense.web")
MODEL = model_from_env()
CREWS: dict[str, NarrationCrew] = {}


def _configure_telemetry() -> None:
    """Send traces to Application Insights when it is configured (keyless, with the app's identity)."""
    if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        return
    try:
        from azure.identity import DefaultAzureCredential
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(credential=DefaultAzureCredential())
    except Exception as exc:  # noqa: BLE001 - telemetry must never stop the app
        log.warning("Azure Monitor not configured: %s", exc)


def _register_agents() -> None:
    try:
        agents = MODEL.register_agents(AGENT_INSTRUCTIONS)
        if agents:
            log.info("Foundry agents ready: %s", ", ".join(agents.values()))
    except Exception as exc:  # noqa: BLE001
        log.warning("Foundry agent registration skipped: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_telemetry()
    if MODEL is not None:
        threading.Thread(target=_register_agents, daemon=True).start()
    yield


app = FastAPI(title="MatchSense", description="Explainable, personalised match intelligence", lifespan=lifespan)


def crew_for(match_id: str, match: StoredMatch) -> NarrationCrew:
    """One crew per match: every viewer shares its analysis cache and rate limit."""
    if match_id not in CREWS:
        CREWS[match_id] = NarrationCrew(MatchContext(match.metadata), model=MODEL, blackboard=Blackboard(),
                                        calls_per_minute=int(os.environ.get("NARRATION_CALLS_PER_MINUTE", "40")))
    return CREWS[match_id]


@lru_cache(maxsize=8)
def get_match(match_id: str) -> StoredMatch:
    path = DATA_ROOT / match_id
    if not (path / "match.json").exists() or ".." in match_id:
        raise HTTPException(404, f"unknown match {match_id!r}")
    return load_match(path)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/matches")
def list_matches() -> list[dict]:
    out = []
    for path in sorted(DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []:
        if (path / "match.json").exists():
            meta = get_match(path.name).metadata
            out.append({
                "id": path.name,
                "home": meta["teams"]["home"]["name"], "away": meta["teams"]["away"]["name"],
                "venue": meta["venue"], "kickoff": meta["kickoff"], "competition": meta["competition"],
                "storyline": meta["storyline"]["name"], "frames": meta["frames"],
            })
    return out


@app.get("/api/narration")
def narration_status() -> dict:
    """Which narration backend is live, and what it has cost so far."""
    if MODEL is None:
        return {"backend": "local", "detail": "deterministic explanations and templates; set FOUNDRY_* to use Foundry"}
    usage = MODEL.usage
    return {"backend": "foundry", "model": MODEL.settings.model, "agents": sorted(MODEL._agents.values()),
            "calls": usage.calls, "failures": usage.failures, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "by_role": usage.by_role,
            "avg_latency_ms": round(usage.latency_ms / usage.calls, 1) if usage.calls else None}


@app.get("/api/schema/cues")
def cue_schema() -> dict:
    return {"fps": FPS, "fields": CUE_SCHEMA, "modes": list(MODES)}


@app.websocket("/ws/matches/{match_id}")
async def stream(ws: WebSocket, match_id: str, mode: str = "casual", team: str = "", player: str = "",
                 speed: float = 8.0, start: float = 0.0, audio: int = 0, lang: str = "en"):
    await ws.accept()
    try:
        match = get_match(match_id)
        profile = Profile.from_params({"mode": mode, "team": team, "player": player, "lang": lang})
    except (HTTPException, ValueError) as exc:
        await ws.send_text(dumps({"k": "error", "message": str(getattr(exc, "detail", exc))}))
        await ws.close()
        return
    session = LiveSession(match, profile, audio=bool(audio), speed=speed, crew=crew_for(match_id, match))
    start_frame = _minute_to_frame(match, start)

    async def send(message: dict) -> None:
        await ws.send_text(dumps(message))

    async def receive() -> None:
        while True:
            session.control(await ws.receive_json())

    receiver = asyncio.create_task(receive())
    try:
        await session.run(send, start_frame=start_frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
        session.close()


def _minute_to_frame(match: StoredMatch, minute: float) -> int:
    """First frame at or after a match minute; minutes after 45 count from the second half."""
    if minute <= 0:
        return 0
    t, period = match.tracking["t"], match.tracking["period"]
    want_period = 1 if minute < 45 else 2
    target = minute * 60.0
    for i in range(len(t)):
        if period[i] == want_period and t[i] >= target:
            return i
    return len(t) - 1


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="matchsense-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
