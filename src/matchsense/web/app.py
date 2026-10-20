"""MatchSense web service: match listings, cue schema, and the live websocket stream.

    matchsense-web --port 8000      then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..render.producer import CUE_SCHEMA
from ..render.profiles import MODES, Profile
from ..sim.io import StoredMatch, load_match
from .session import FPS, LiveSession, dumps

DATA_ROOT = Path(os.environ.get("MATCHSENSE_DATA", Path(__file__).resolve().parents[3] / "data" / "matches"))
STATIC = Path(__file__).with_name("static")

app = FastAPI(title="MatchSense", description="Explainable, personalised match intelligence")


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
    session = LiveSession(match, profile, audio=bool(audio), speed=speed)
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
