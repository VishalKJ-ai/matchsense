"""Probability and physics models the simulated players use to make decisions.

These are deliberately simple, documented parametric models. They drive the
simulation, and their outputs are written to a separate ground-truth file so
the analytics layer can be checked against them. The analytics layer never
reads them during a match: it has to work from tracking and events, like a
real broadcast data feed.
"""

from __future__ import annotations

import math

from . import pitch

GRAVITY = 9.81


def logistic(z: float) -> float:
    if z < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def expected_goals(
    distance: float,
    angle: float,
    *,
    header: bool = False,
    blockers: int = 0,
    pressured: bool = False,
    crowd: int = 0,
    set_piece: str | None = None,
) -> float:
    """Probability that a shot is scored.

    Logistic in distance and goal-mouth angle, fitted by hand to three anchor
    points that match public open-play xG models: about 0.50 from six metres
    out centrally, 0.25 from the penalty spot and 0.04 from 25 m.
    """
    if set_piece == "penalty":
        return 0.78
    z = -0.44 - 0.122 * distance + 1.066 * angle
    if header:
        z -= 0.9
    if pressured:
        z -= 0.35
    if set_piece == "free_kick":
        z -= 0.35
    z -= 0.45 * blockers
    # several defenders around the shooter close the angle and hurry the shot
    z -= 0.3 * max(crowd - 1, 0)
    return logistic(z)


def expected_threat(rel_x: float, rel_y: float) -> float:
    """Smooth expected-threat surface: the chance a possession here ends in a goal.

    Combines slow growth up the pitch with the shooting value near goal. Its
    magnitudes follow published xT grids: under 0.01 in your own half, about
    0.04 at the edge of the area and about 0.3 inside the six-yard box.
    """
    dist, angle = pitch.shot_geometry_rel(rel_x, rel_y)
    progression = 0.005 + 0.045 * max(rel_x / pitch.LENGTH, 0.0) ** 2.5
    shooting = 0.55 * expected_goals(dist, angle) * logistic((rel_x - 80.0) / 4.0)
    return progression + shooting


def pass_kinematics(distance: float, lofted: bool, power: float = 1.0) -> tuple[float, float, float, float]:
    """Initial speed, arrival speed, flight time and apex height of a pass.

    Ground passes are struck harder over longer distances and decelerate
    linearly through rolling friction. Lofted passes keep roughly constant
    horizontal speed, and their apex comes from projectile motion.
    """
    distance = max(distance, 0.5)
    if lofted:
        v0 = min(max(11.0 + 0.38 * distance, 14.0), 30.0) * power
        duration = distance / (0.75 * v0)
        apex = GRAVITY * duration**2 / 8.0
        return v0, 0.75 * v0, duration, apex
    v0 = min(max(6.5 + 0.42 * distance, 7.0), 24.0) * power
    v_end = max(3.0, 0.5 * v0)
    duration = 2.0 * distance / (v0 + v_end)
    return v0, v_end, duration, 0.0


def pass_accuracy(passing: float, distance: float, lofted: bool, pressure_dist: float, kind: str) -> float:
    """Chance the passer puts the ball where intended, before interceptions and control."""
    base = 0.86 + 0.12 * (passing - 50.0) / 45.0
    penalty = 0.0035 * max(0.0, distance - 12.0)
    if lofted:
        penalty += 0.05
    if kind == "cross":
        penalty += 0.08
    elif kind == "through":
        penalty += 0.05
    elif kind == "throw_in":
        penalty -= 0.04
    if pressure_dist < 1.5:
        penalty += 0.14
    elif pressure_dist < 3.0:
        penalty += 0.07
    elif pressure_dist < 5.0:
        penalty += 0.02
    return min(max(base - penalty, 0.35), 0.99)


def rating_to_top_speed(pace: float) -> float:
    """Top sprint speed in m/s: pace 50 is about 26.6 km/h and pace 95 about 33.5 km/h."""
    return 7.4 + (pace - 50.0) / 45.0 * 1.9


def rating_to_acceleration(accel: float) -> float:
    return 3.0 + (accel - 50.0) / 45.0 * 1.5
