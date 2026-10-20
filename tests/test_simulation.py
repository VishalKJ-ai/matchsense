"""Invariants of a full simulated match. One 90-minute match is shared by all tests."""

from collections import Counter

import numpy as np
import pytest

from conftest import SEED

from matchsense.schema import MatchEvent
from matchsense.sim import load_default_teams, load_match, simulate, write_match
from matchsense.sim.report import realism_checks


def test_events_match_the_schema_and_are_time_ordered(match):
    frames = []
    for i, ev in enumerate(match.events):
        model = MatchEvent.model_validate(ev)
        assert model.index == i
        frames.append(model.frame)
    assert frames == sorted(frames)
    assert len({ev["id"] for ev in match.events}) == len(match.events)


def test_score_matches_goal_events(match):
    meta = match.metadata
    goals = Counter(ev["team"] for ev in match.events if ev["type"] == "goal")
    assert goals[meta["teams"]["home"]["id"]] == meta["score"]["home"]
    assert goals[meta["teams"]["away"]["id"]] == meta["score"]["away"]
    shots = {ev["id"]: ev for ev in match.events if ev["type"] == "shot"}
    for ev in match.events:
        if ev["type"] == "goal":
            assert shots[ev["attributes"]["shot_event"]]["outcome"] == "goal"


def test_every_pass_and_shot_has_an_outcome(match):
    passes = [ev for ev in match.events if ev["type"] == "pass"]
    assert all(ev["outcome"] in {"complete", "incomplete", "out", "offside"} for ev in passes)
    for ev in passes:
        if ev["outcome"] == "complete":
            assert ev["attributes"]["recipient"]
    shots = [ev for ev in match.events if ev["type"] == "shot"]
    assert all(ev["outcome"] in {"goal", "saved", "blocked", "off_target", "post"} for ev in shots)


def test_possession_ids_only_increase(match):
    ids = [ev["possession"] for ev in match.events]
    assert ids == sorted(ids)
    changes = [ev for ev in match.events if ev["type"] == "possession_change"]
    assert len(changes) == ids[-1]


def test_players_move_within_physical_limits(match):
    xy = match.tracking["xy"].astype(np.float64)
    ids = match.tracking["player_index"]
    period = match.tracking["period"]
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1))
    same = (ids[1:] == ids[:-1]) & (period[1:] == period[:-1])[:, None] & np.isfinite(step)
    speeds = step[same] * 10.0
    # 10.5 m/s is about 38 km/h, above any real sprint; restart placements are excluded
    assert np.quantile(speeds, 0.9999) < 10.5


def test_ball_stays_near_the_pitch(match):
    ball = match.tracking["ball"]
    assert ball[:, 0].min() > -3 and ball[:, 0].max() < 108
    assert ball[:, 1].min() > -3 and ball[:, 1].max() < 71
    assert ball[:, 2].min() >= 0


def test_match_is_realistic(match):
    checks, _ = realism_checks(match.metadata, match.events, match.tracking)
    failed = [c for c in checks if not c.ok]
    assert len(failed) <= 6, [f"{c.name}={c.value}" for c in failed]


def test_round_trip_through_files(match, tmp_path):
    write_match(match, tmp_path / "m")
    loaded = load_match(tmp_path / "m")
    assert loaded.metadata["score"] == match.metadata["score"]
    assert len(loaded.events) == len(match.events)
    assert loaded.tracking["xy"].shape == match.tracking["xy"].shape


@pytest.mark.slow
def test_same_seed_gives_the_same_match(match):
    home, away = load_default_teams()
    again = simulate(home, away, seed=SEED)
    assert again.metadata["score"] == match.metadata["score"]
    assert [e["type"] for e in again.events] == [e["type"] for e in match.events]
