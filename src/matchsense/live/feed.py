"""Replay a stored match as a live data feed.

A real broadcast feed delivers tracking frames almost instantly, while event
data arrives a couple of seconds after the action because an operator or model
has to confirm it. The replay copies that: frames come in order at 10 Hz, and
each event is released `latency` seconds after it finishes. Carries finish at
their `end_frame`; everything else finishes on its own frame.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..sim.io import StoredMatch

FPS = 10


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    period: int
    state: int  # 0 owned, 1 in flight, 2 loose, 3 dead
    owner: int  # slot holding the ball, or -1
    ball: np.ndarray  # (3,)
    xy: np.ndarray  # (22, 2); NaN for players sent off
    player_index: np.ndarray  # (22,) index into metadata["players"]


def release_frame(ev: dict, latency_frames: int) -> int:
    end = ev["attributes"].get("end_frame", ev["frame"]) if ev["type"] == "carry" else ev["frame"]
    return int(end) + latency_frames


def replay(match: StoredMatch, latency: float = 2.0) -> Iterator[tuple[str, object]]:
    """Yield ("frame", Frame) and ("event", dict) items in the order a live consumer would see them."""
    tr = match.tracking
    n = len(tr["t"])
    lat = int(round(latency * FPS))
    queue = [(release_frame(ev, lat), ev["index"], ev) for ev in match.events]
    heapq.heapify(queue)
    for i in range(n):
        frame = Frame(i, float(tr["t"][i]), int(tr["period"][i]), int(tr["state"][i]), int(tr["owner"][i]),
                      tr["ball"][i], tr["xy"][i], tr["player_index"][i])
        yield "frame", frame
        while queue and queue[0][0] <= i:
            yield "event", heapq.heappop(queue)[2]
    while queue:
        yield "event", heapq.heappop(queue)[2]
