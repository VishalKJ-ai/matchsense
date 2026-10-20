"""The live analytics layer, run over the shared simulated match."""

import json

import numpy as np
import pytest

from matchsense.analytics.context import MatchContext, clock_label
from matchsense.analytics.engine import LiveAnalytics, analyse
from matchsense.analytics.passing import FEATURES, PassDifficultyModel, pass_features
from matchsense.live.feed import replay
from matchsense.sim import load_default_teams, simulate
from matchsense.sim.io import StoredMatch
from matchsense.sim.report import player_physical


def stored(result):
    return StoredMatch(result.metadata, result.events, result.tracking, result.truth)


@pytest.fixture(scope="module")
def analytics(match):
    return analyse(stored(match))


def test_feed_releases_every_event_once_after_it_happens(match):
    seen, last_frame = [], -1
    for kind, item in replay(stored(match), latency=2.0):
        if kind == "frame":
            assert item.index == last_frame + 1
            last_frame = item.index
        else:
            assert item["frame"] <= last_frame
            seen.append(item["id"])
    assert sorted(seen) == sorted(ev["id"] for ev in match.events)
    assert last_frame == len(match.tracking["t"]) - 1


def test_clock_labels():
    assert clock_label(1, 0) == "1'"
    assert clock_label(1, 2700 + 75) == "45+2'"
    assert clock_label(2, 2700) == "46'"
    assert clock_label(2, 5400 + 30) == "90+1'"


def test_pass_model_is_shipped_and_sensible(match):
    model = PassDifficultyModel.load()
    assert len(model.coef) == len(FEATURES)
    assert model.metrics["test_auc"] > 0.75
    ctx = MatchContext(match.metadata)
    diffs = []
    for ev in match.events:
        if ev["type"] == "pass" and ev["outcome"] != "offside":
            diffs.append(model.difficulty(pass_features(ev, match.tracking["xy"][ev["frame"]], ctx)))
    diffs = np.array(diffs)
    assert diffs.min() >= 0 and diffs.max() <= 100
    assert 5 < np.median(diffs) < 40


def test_goals_and_score_follow_the_feed(match, analytics):
    meta = match.metadata
    goals = [s for s in analytics.emitted if s.kind == "goal"]
    assert len(goals) == meta["score"]["home"] + meta["score"]["away"]
    ctx = analytics.ctx
    assert analytics.score == {ctx.home: meta["score"]["home"], ctx.away: meta["score"]["away"]}


def test_signals_are_serialisable_and_cite_real_events(match, analytics):
    ids = {ev["id"] for ev in match.events}
    for s in analytics.emitted:
        json.dumps(s.to_dict())
        assert set(s.evidence) <= ids
        assert 0.0 <= s.significance <= 1.0


def test_distances_agree_with_the_offline_report(match, analytics):
    offline = player_physical(match.tracking)
    by_index = {p["index"]: p["id"] for p in match.metadata["players"]}
    for idx, st in offline.items():
        if st["minutes"] < 30:
            continue
        live = analytics.physical.stats[by_index[idx]]["distance_m"] / 1000
        assert live == pytest.approx(st["distance_km"], rel=0.02)


def test_windows_are_consistent(analytics):
    w = analytics.tactical.window(len(analytics.tactical.buckets))
    poss = sum(t["possession"] for t in w["teams"].values())
    assert poss == pytest.approx(100.0, abs=0.2)
    assert 0 <= w["match"]["chaos"] <= 100
    assert w["match"]["rhythm"] in {"patient", "flowing", "frantic", "stop-start"}


def test_snapshot_and_player_table(analytics):
    snap = analytics.snapshot()
    json.dumps(snap, default=float)
    table = analytics.player_table()
    assert len(table) >= 22
    assert all(row["distance_km"] >= 0 for row in table.values())


@pytest.mark.slow
def test_detects_the_scripted_pressing_change():
    """Comeback storyline: Westmarsh press higher from 58', Ironvale drop off from 62'."""
    home, away = load_default_teams()
    result = simulate(home, away, seed=7001, storyline="comeback")
    signals = analyse(stored(result)).emitted
    shifts = [(s.team, s.t / 60, s.facts["pressing_harder"]) for s in signals
              if s.kind == "tactical_shift" and s.facts["metric"] == "press"]
    assert any(t == "WMA" and harder and 58 <= m <= 78 for t, m, harder in shifts), shifts
    assert any(t == "IRV" and not harder and 62 <= m <= 82 for t, m, harder in shifts), shifts
