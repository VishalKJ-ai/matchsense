"""The narration crew: fact-checking, fallbacks, caching and the Foundry request shape.

Foundry is replaced by a fake OpenAI-compatible client, so these tests run offline.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from matchsense.agents import Blackboard, FoundryModel, FoundrySettings, NarrationCrew, model_from_env
from matchsense.agents.factcheck import allowed_numbers, check_text
from matchsense.analytics.engine import analyse
from matchsense.render import Profile
from matchsense.render.templates import Templates
from matchsense.sim.io import StoredMatch
from matchsense.web.session import LiveSession

SETTINGS = FoundrySettings("https://aif-x.services.ai.azure.com/api/projects/proj-matchsense", "gpt-5-mini")


class FakeResponses:
    """Answers like the Responses API. `script` maps role -> list of outputs (dict or Exception), used in turn."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "text" in kwargs:
            role = kwargs["text"]["format"]["name"]
        else:  # an agent reference: the agent definition holds the schema
            role = kwargs["extra_body"]["agent_reference"]["name"].removeprefix("matchsense-")
        queue = self.script.get(role) or [{"why": "x", "angle": "context"}]
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(output_text=json.dumps(out), usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def fake_model(script):
    responses = FakeResponses(script)
    return FoundryModel(SETTINGS, client=SimpleNamespace(responses=responses)), responses


@pytest.fixture(scope="module")
def analysed(match):
    return analyse(StoredMatch(match.metadata, match.events, match.tracking, match.truth))


@pytest.fixture()
def goal(analysed):
    return next(s for s in analysed.emitted if s.kind == "goal")


def test_fact_check_accepts_reformatted_numbers_and_rejects_invented_ones():
    pack = {"facts": {"xg": 0.32, "before": 27.9, "after": 43.3, "speed_ms": 25.0}, "clock": "86'"}
    allowed = allowed_numbers(pack)
    assert {32.0, 3.0, 15.0, 43.0, 86.0, 90.0} <= allowed  # 32%, 1 in 3, change, rounding, clock, km/h
    assert check_text({"body": "A 1 in 3 chance at 86'; the line moved 15 m up to 43.3 m"}, pack).ok
    bad = check_text({"body": "He ran 77 metres"}, pack)
    assert not bad.ok and bad.ungrounded == ["77"]
    assert not check_text({"title": "x" * 80}, pack, {"title": 60}).ok


def test_local_crew_explains_and_uses_templates(analysed, goal):
    crew = NarrationCrew(analysed.ctx)
    snap = analysed.snapshot()
    assert not crew.remote
    assert crew.explain_now(goal, snap)
    template = Templates(analysed.ctx).render(goal, "casual")
    n = crew.narrate(goal, snap, Profile(), template)
    assert n.source == "local" and n.title == template["title"] and n.why


def test_narrator_is_repaired_once_then_falls_back(analysed, goal):
    snap = analysed.snapshot()
    template = Templates(analysed.ctx).render(goal, "casual")
    model, responses = fake_model({
        "tactician": [{"why": "It changes the game.", "angle": "turning_point"}],
        "narrator": [{"title": "Goal", "body": "A 99 metre run"}, {"title": "Goal!", "body": "What a finish."}],
    })
    n = NarrationCrew(analysed.ctx, model=model).narrate(goal, snap, Profile(), template)
    assert n.source == "foundry" and n.attempts == 2 and n.body == "What a finish."
    assert [c["text"]["format"]["name"] for c in responses.calls] == ["tactician", "narrator", "narrator"]
    assert "99" in responses.calls[2]["input"][0]["content"]  # the repair request names the bad number

    model, _ = fake_model({"narrator": [{"title": "Goal", "body": "A 99 metre run"}]})
    n = NarrationCrew(analysed.ctx, model=model).narrate(goal, snap, Profile(), template)
    assert n.source == "local" and "fact-check" in n.note and n.title == template["title"]

    model, _ = fake_model({"narrator": [TimeoutError("slow")], "tactician": [TimeoutError("slow")]})
    n = NarrationCrew(analysed.ctx, model=model).narrate(goal, snap, Profile(), template)
    assert n.source == "local" and "unavailable" in n.note and n.why  # local explanation survives


def test_analysis_is_shared_and_captions_cached(analysed, goal):
    snap = analysed.snapshot()
    template = Templates(analysed.ctx).render(goal, "casual")
    model, responses = fake_model({"narrator": [{"title": "Goal!", "body": "Scored."}]})
    crew = NarrationCrew(analysed.ctx, model=model, blackboard=Blackboard())
    crew.narrate(goal, snap, Profile(mode="casual"), template)
    crew.narrate(goal, snap, Profile(mode="casual"), template)
    crew.narrate(goal, snap, Profile(mode="analyst"), template)
    roles = [c["text"]["format"]["name"] for c in responses.calls]
    assert roles.count("tactician") == 1  # one analysis per moment, whoever is watching
    assert roles.count("narrator") == 2  # one caption per audience


def test_foundry_request_shape_and_agent_fallback(analysed, goal):
    model, responses = fake_model({"narrator": [{"title": "T", "body": "B"}]})
    model.complete("narrator", "instructions", {"a": 1}, {"type": "object"})
    call = responses.calls[-1]
    assert call["model"] == "gpt-5-mini" and call["instructions"] == "instructions"
    assert call["store"] is False and call["reasoning"] == {"effort": "minimal"}
    assert call["text"]["format"] == {"type": "json_schema", "name": "narrator", "schema": {"type": "object"}, "strict": True}

    model._agents["narrator"] = "matchsense-narrator"
    responses.script["narrator"] = [RuntimeError("agent endpoint unavailable"), {"title": "T", "body": "B"}]
    assert model.complete("narrator", "instructions", {"a": 1}, {"type": "object"}) == {"title": "T", "body": "B"}
    agent_call, direct_call = responses.calls[-2], responses.calls[-1]
    assert agent_call["extra_body"] == {"agent_reference": {"name": "matchsense-narrator", "type": "agent_reference"}}
    # Foundry rejects model, schema or reasoning overrides on an agent call; the agent definition holds them.
    assert not {"model", "instructions", "text", "reasoning"} & agent_call.keys()
    assert direct_call["model"] == "gpt-5-mini" and "text" in direct_call


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    assert model_from_env() is None
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", SETTINGS.project_endpoint)
    monkeypatch.setenv("FOUNDRY_MODEL_DEPLOYMENT", "gpt-6-sol")
    monkeypatch.setenv("FOUNDRY_USE_AGENTS", "1")
    model = model_from_env()
    assert isinstance(model, FoundryModel) and model.settings.model == "gpt-6-sol" and model.settings.use_agents
    monkeypatch.setenv("MATCHSENSE_NARRATION", "local")
    assert model_from_env() is None


def test_live_session_streams_crew_text(match):
    stored = StoredMatch(match.metadata, match.events, match.tracking, match.truth)
    model, _ = fake_model({
        "tactician": [{"why": "It matters.", "angle": "context"}],
        "narrator": [{"title": "On air", "body": "Written by the crew."}],
        "recapper": [{"headline": "Full-time", "recap": "A match was played."}],
    })
    session = LiveSession(stored, Profile(mode="analyst"), speed=2000,
                          crew=NarrationCrew(_ctx(match), model=model))
    sent = []

    async def send(m):
        sent.append(m)

    async def go():
        await session.run(send, start_frame=len(match.tracking["t"]) - 9000)
        await asyncio.sleep(0.5)

    asyncio.run(go())
    kinds = {m["k"] for m in sent}
    assert "cue" in kinds and "cue_text" in kinds
    assert any(m["k"] == "recap" and m.get("source") == "foundry" for m in sent)
    lower = [m for m in sent if m["k"] == "cue" and m["layer"] == "lower_third"]
    assert lower and all("why" in m for m in lower)


def _ctx(match):
    from matchsense.analytics.context import MatchContext
    return MatchContext(match.metadata)
