import math

import numpy as np
import pytest

from matchsense.sim import pitch
from matchsense.sim.models import expected_goals, expected_threat, pass_accuracy, pass_kinematics
from matchsense.sim.storyline import STORYLINES


def test_relative_frame_is_its_own_inverse():
    xy = np.array([12.3, 45.6])
    assert np.allclose(pitch.to_rel(pitch.to_rel(xy, -1), -1), xy)
    assert np.allclose(pitch.to_rel(xy, 1), xy)
    assert np.allclose(pitch.to_rel(np.array([0.0, 0.0]), -1), [pitch.LENGTH, pitch.WIDTH])


def test_xg_anchor_points():
    def central(dist):
        return expected_goals(*pitch.shot_geometry_rel(pitch.LENGTH - dist, 34.0))

    assert central(6.0) == pytest.approx(0.50, abs=0.05)
    assert central(11.0) == pytest.approx(0.25, abs=0.04)
    assert central(25.0) == pytest.approx(0.04, abs=0.015)
    assert expected_goals(11.0, 0.6, set_piece="penalty") == pytest.approx(0.78)


def test_xg_falls_with_pressure_and_headers():
    dist, angle = pitch.shot_geometry_rel(93.0, 34.0)
    base = expected_goals(dist, angle)
    assert expected_goals(dist, angle, header=True) < base
    assert expected_goals(dist, angle, blockers=2) < expected_goals(dist, angle, blockers=1) < base
    assert expected_goals(dist, angle, crowd=3) < base


def test_expected_threat_grows_towards_goal():
    values = [expected_threat(x, 34.0) for x in (10, 40, 70, 90, 100)]
    assert values == sorted(values)
    assert values[0] < 0.01 < values[-1]


@pytest.mark.parametrize("lofted", [False, True])
def test_pass_kinematics_cover_the_distance(lofted):
    for dist in (5.0, 15.0, 30.0, 50.0):
        v0, v_end, duration, apex = pass_kinematics(dist, lofted)
        assert v0 >= v_end > 0
        covered = (v0 + v_end) / 2 * duration if not lofted else 0.75 * v0 * duration
        assert covered == pytest.approx(dist, rel=1e-6)
        assert (apex > 0) == lofted


def test_pass_accuracy_drops_with_distance_and_pressure():
    free = pass_accuracy(80, 15, False, 10.0, "pass")
    assert pass_accuracy(80, 45, False, 10.0, "pass") < free
    assert pass_accuracy(80, 15, False, 1.0, "pass") < free
    assert pass_accuracy(95, 15, False, 10.0, "pass") > pass_accuracy(55, 15, False, 10.0, "pass")


def test_storyline_phases_accumulate():
    story = STORYLINES["comeback"]
    early = story.styles_at(10, 1)
    late = story.styles_at(70, 1)
    assert early["press_line"] < late["press_line"]
    assert story.styles_at(70, 0)["press_line"] < story.styles_at(10, 0)["press_line"]


def test_shot_angle_is_widest_in_front_of_goal():
    _, central = pitch.shot_geometry_rel(95.0, 34.0)
    _, wide = pitch.shot_geometry_rel(95.0, 10.0)
    assert central > wide > 0
    assert central < math.pi
