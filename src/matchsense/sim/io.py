"""Reading and writing simulated matches.

A match is a folder with four files:

    match.json     metadata: teams, line-ups, players, venue, score, storyline
    events.jsonl   one event per line, time-ordered (see matchsense.schema)
    tracking.npz   10 Hz positions of all 22 players and the ball
    truth.jsonl    the simulator's internal probabilities, for evaluation only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schema import MatchEvent
from .engine import SimulationResult


@dataclass
class StoredMatch:
    metadata: dict
    events: list[dict]
    tracking: dict[str, np.ndarray]
    truth: list[dict]


def _jsonable(row: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in row.items()}


def write_match(result: SimulationResult, out_dir: str | Path, *, validate: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if validate:
        for ev in result.events:
            MatchEvent.model_validate(ev)
    (out / "match.json").write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        for ev in result.events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    with (out / "truth.jsonl").open("w", encoding="utf-8") as f:
        for row in result.truth:
            f.write(json.dumps(_jsonable(row)) + "\n")
    np.savez_compressed(out / "tracking.npz", **result.tracking)
    return out


def load_match(path: str | Path) -> StoredMatch:
    root = Path(path)
    metadata = json.loads((root / "match.json").read_text(encoding="utf-8"))
    with (root / "events.jsonl").open(encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    truth_path = root / "truth.jsonl"
    truth = []
    if truth_path.exists():
        with truth_path.open(encoding="utf-8") as f:
            truth = [json.loads(line) for line in f if line.strip()]
    with np.load(root / "tracking.npz") as npz:
        tracking = {k: npz[k] for k in npz.files}
    return StoredMatch(metadata, events, tracking, truth)
