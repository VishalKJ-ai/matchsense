"""Overlay cues, the producer's scheduling rules, audio description and the live websocket."""

import json

import pytest
from fastapi.testclient import TestClient

from matchsense.analytics.context import MatchContext
from matchsense.analytics.engine import analyse
from matchsense.analytics.signals import SignalFactory
from matchsense.render import Producer, Profile
from matchsense.render.describe import Describer
from matchsense.render.templates import Templates, odds
from matchsense.sim.io import StoredMatch


@pytest.fixture(scope="module")
def analysed(match):
    return analyse(StoredMatch(match.metadata, match.events, match.tracking, match.truth))


def test_every_signal_renders_in_both_registers(analysed):
    templates = Templates(analysed.ctx)
    rendered = 0
    for s in analysed.emitted:
        for style in ("casual", "analyst"):
            text = templates.render(s, style, perspective=analysed.ctx.home)
            if text is not None:
                assert text["title"] and isinstance(text["body"], str)
                rendered += 1
    assert rendered >= len(analysed.emitted)


def test_odds_are_plain_english():
    assert odds(0.32) == "scored about 1 in 3 times"
    assert odds(0.6) == "scored more often than not"
    assert odds(0.04) == "scored about 1 in 25 times"


def _signal(factory, kind, significance, team="IRV", players=None, facts=None, t=600.0):
    return factory.make(kind, frame=int(t * 10), t=t, period=1, clock="11'", team=team, players=players or [],
                        facts=facts or {}, significance=significance)


@pytest.fixture()
def ctx(match):
    return MatchContext(match.metadata)


def test_layer_budget_and_preemption(ctx):
    f = SignalFactory()
    producer = Producer(ctx, Profile(mode="analyst"))
    swing = _signal(f, "momentum_swing", 0.5, facts={"momentum_before": 50, "momentum_now": -60})
    msgs = producer.on_signal(swing, now=1000)
    assert [m["k"] for m in msgs] == ["cue"]
    goal = _signal(f, "goal", 1.0, players=["IRV-09"], facts={
        "scorer": "IRV-09", "assist": None, "score": {"IRV": 1, "WMA": 0}, "situation": "takes the lead",
        "xg": 0.2, "shot_speed_kmh": 90.0, "distance_m": 12.0, "passes_in_move": 3, "move_duration_s": 9.0,
        "against_run_of_play": False, "late": False})
    msgs = producer.on_signal(goal, now=1010)
    assert [m["k"] for m in msgs] == ["cue_end", "cue"]
    assert msgs[1]["kind"] == "goal" and msgs[1]["layer"] == "lower_third"
    # a weaker cue waits for the layer instead of bumping the goal, then is dropped once stale
    card = _signal(f, "card", 0.6, players=["WMA-04"], facts={"card": "yellow", "score": {"IRV": 1, "WMA": 0}})
    assert producer.on_signal(card, now=1020) == []
    assert producer.tick(now=1020 + 9 * 10) == []
    assert producer.tick(now=1020 + 200) == []


def test_profiles_shape_what_is_shown(ctx):
    f = SignalFactory()
    sprint = _signal(f, "speed_burst", 0.25, team="WMA", players=["WMA-07"], facts={"speed_kmh": 32.0})
    assert not Profile(mode="casual").wants(sprint)[0]
    assert Profile(mode="analyst").wants(sprint)[0]
    focused, priority = Profile(mode="player", player="WMA-07").wants(sprint)
    assert focused and priority > 60
    own = _signal(f, "big_chance", 0.4, team="WMA", players=["WMA-07"], facts={"xg": 0.3})
    assert Profile(mode="club", team="WMA").wants(own)[0]
    assert not Profile(mode="club", team="IRV").wants(own)[0]


def test_audio_description_spaces_routine_lines_and_calls_goals(ctx):
    d = Describer(ctx)
    goal = {"type": "goal", "team": "WMA", "player": "WMA-07", "period": 2, "t": 5100.0,
            "attributes": {"scorer": "WMA-07", "assist": None, "score_home": 1, "score_away": 1}}
    lines = d.on_event(goal)
    assert len(lines) == 1 and lines[0].priority == 3 and lines[0].interrupt
    assert "Goal!" in lines[0].text and "1" in lines[0].text
    assert d._emit("routine", 0, d.last_frame + 10) == []
    assert d._emit("routine", 0, d.last_frame + 50)
    assert "box" in d.zone("IRV", 1, (100.0, 34.0))


def test_websocket_streams_the_live_pipeline():
    from matchsense.web.app import app

    client = TestClient(app)
    matches = client.get("/api/matches").json()
    assert any(m["id"] == "comeback" for m in matches)
    assert "fields" in client.get("/api/schema/cues").json()
    kinds = set()
    with client.websocket_connect("/ws/matches/comeback?mode=analyst&speed=64&start=95&audio=1") as ws:
        first = json.loads(ws.receive_text())
        assert first["k"] == "init" and len(first["players"]) >= 36 and "history" in first
        while True:
            msg = json.loads(ws.receive_text())
            kinds.add(msg["k"])
            if msg["k"] == "f":
                assert len(msg["xy"]) == 44 and len(msg["b"]) == 3
            if msg["k"] == "end":
                break
    assert {"f", "state", "end"} <= kinds
