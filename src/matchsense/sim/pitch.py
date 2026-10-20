"""Pitch geometry.

Coordinates are metres on a 105 x 68 pitch with the origin at the bottom-left
corner. Absolute coordinates never change during a match. Each team also has a
*relative* frame in which it always attacks towards x = 105; that frame is used
for everything tactical (shape, threat, shooting) so the same logic works for
both teams and both halves.
"""

from __future__ import annotations

import math

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
HALFWAY = LENGTH / 2
CENTER = np.array([LENGTH / 2, WIDTH / 2])

GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
POST_Y = (WIDTH / 2 - GOAL_WIDTH / 2, WIDTH / 2 + GOAL_WIDTH / 2)
PENALTY_DEPTH = 16.5
PENALTY_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
PENALTY_SPOT = 11.0

_FLIP = np.array([LENGTH, WIDTH])


def to_rel(xy, direction: int) -> np.ndarray:
    """Convert absolute coordinates to the frame of a team attacking in `direction`.

    `direction` is +1 for a team attacking towards x = 105 and -1 otherwise.
    Works on a single point or an (n, 2) array. The transform is its own inverse.
    """
    xy = np.asarray(xy, dtype=float)
    return xy.copy() if direction > 0 else _FLIP - xy


from_rel = to_rel


def goal_center(direction: int) -> np.ndarray:
    """Centre of the goal that a team attacking in `direction` is shooting at."""
    return np.array([LENGTH if direction > 0 else 0.0, WIDTH / 2])


def own_goal_center(direction: int) -> np.ndarray:
    return goal_center(-direction)


def shot_geometry_rel(rel_x: float, rel_y: float) -> tuple[float, float]:
    """Distance to goal centre (m) and the angle subtended by the posts (radians)."""
    dx = max(LENGTH - rel_x, 0.05)
    dy = rel_y - WIDTH / 2
    dist = math.hypot(LENGTH - rel_x, dy)
    half = GOAL_WIDTH / 2
    angle = abs(math.atan2(half - dy, dx) - math.atan2(-half - dy, dx))
    return dist, angle


def shot_geometry(xy, direction: int) -> tuple[float, float]:
    rel = to_rel(xy, direction)
    return shot_geometry_rel(float(rel[0]), float(rel[1]))


def in_penalty_area_rel(rel_xy) -> bool:
    """True when a relative point is inside the penalty area the team attacks."""
    return rel_xy[0] >= LENGTH - PENALTY_DEPTH and abs(rel_xy[1] - WIDTH / 2) <= PENALTY_HALF_WIDTH


def inside(xy, margin: float = 0.0) -> bool:
    return -margin <= xy[0] <= LENGTH + margin and -margin <= xy[1] <= WIDTH + margin


def clip(xy, margin: float = 0.5) -> np.ndarray:
    return np.array([min(max(xy[0], margin), LENGTH - margin), min(max(xy[1], margin), WIDTH - margin)])


def unit(v) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return np.zeros(2) if n < 1e-9 else np.asarray(v, dtype=float) / n


def distance_to_segment(p, a, b) -> float:
    """Shortest distance from point p to segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.hypot(*(p - a)))
    t = min(max(float((p - a) @ ab) / denom, 0.0), 1.0)
    return float(np.hypot(*(p - (a + t * ab))))
