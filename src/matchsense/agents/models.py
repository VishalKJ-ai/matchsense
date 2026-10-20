"""Model backends for the narration crew.

FoundryModel calls a model deployed in Microsoft Foundry through the project's
OpenAI-compatible Responses API, signed in with Microsoft Entra ID (a managed
identity on Azure Container Apps, `az login` on a laptop). There are no API keys.
Every call asks for strict JSON matching the role's schema and stores nothing
server-side.

With FOUNDRY_USE_AGENTS=1 each crew role is also registered as a versioned
prompt agent in the Foundry project (matchsense-tactician, -narrator,
-recapper), so instructions are managed and traced in the portal, and calls
reference those agents. If that preview path fails, the model falls back to
direct calls.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("matchsense.agents")

try:  # tracing is optional; spans go to Azure Monitor when it is configured
    from opentelemetry import trace

    tracer = trace.get_tracer("matchsense.agents")
except ImportError:  # pragma: no cover
    tracer = None


class ModelError(RuntimeError):
    pass


class ChatModel(Protocol):
    name: str

    def complete(self, role: str, instructions: str, payload: dict, schema: dict) -> dict: ...


@dataclass(frozen=True)
class FoundrySettings:
    project_endpoint: str
    model: str
    use_agents: bool = False
    timeout_s: float = 8.0
    reasoning_effort: str | None = "low"
    max_output_tokens: int = 600

    @classmethod
    def from_env(cls) -> "FoundrySettings | None":
        endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
        model = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT", "").strip()
        if not endpoint or not model:
            return None
        effort = os.environ.get("FOUNDRY_REASONING_EFFORT", "low").strip() or None
        return cls(
            project_endpoint=endpoint,
            model=model,
            use_agents=os.environ.get("FOUNDRY_USE_AGENTS", "0") == "1",
            timeout_s=float(os.environ.get("FOUNDRY_TIMEOUT_S", "8")),
            reasoning_effort=None if effort == "none" else effort,
            max_output_tokens=int(os.environ.get("FOUNDRY_MAX_OUTPUT_TOKENS", "600")),
        )


@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    by_role: dict[str, int] = field(default_factory=dict)


class FoundryModel:
    name = "foundry"

    def __init__(self, settings: FoundrySettings, client: Any = None, project: Any = None):
        self.settings = settings
        self._client = client
        self._project = project
        self._agents: dict[str, str] = {}
        self.usage = Usage()

    def _connect(self) -> Any:
        if self._client is None:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential

            self._project = AIProjectClient(endpoint=self.settings.project_endpoint, credential=DefaultAzureCredential())
            self._client = self._project.get_openai_client().with_options(
                timeout=self.settings.timeout_s, max_retries=1)
        return self._client

    def register_agents(self, roles: dict[str, str]) -> dict[str, str]:
        """Create or update a versioned Foundry prompt agent for each role. Returns role -> agent name."""
        if not self.settings.use_agents:
            return {}
        self._connect()
        from azure.ai.projects.models import PromptAgentDefinition

        for role, instructions in roles.items():
            name = f"matchsense-{role}"
            try:
                self._project.agents.create_version(
                    agent_name=name,
                    definition=PromptAgentDefinition(model=self.settings.model, instructions=instructions),
                    description=f"MatchSense {role}: part of the live narration crew",
                    metadata={"app": "matchsense", "role": role},
                )
                self._agents[role] = name
            except Exception as exc:  # the agents API is optional; direct calls still work
                log.warning("could not register Foundry agent %s: %s", name, exc)
        return dict(self._agents)

    def complete(self, role: str, instructions: str, payload: dict, schema: dict) -> dict:
        client = self._connect()
        kwargs: dict[str, Any] = {
            "input": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "text": {"format": {"type": "json_schema", "name": role, "schema": schema, "strict": True}},
            "max_output_tokens": self.settings.max_output_tokens,
            "store": False,
        }
        agent = self._agents.get(role)
        if agent:
            kwargs["extra_body"] = {"agent": {"name": agent, "type": "agent_reference"}}
        else:
            kwargs.update(model=self.settings.model, instructions=instructions)
        if self.settings.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.settings.reasoning_effort}
        start = time.perf_counter()
        span = tracer.start_as_current_span(f"chat {role}") if tracer else None
        try:
            if span:
                span.__enter__()
                current = trace.get_current_span()
                current.set_attribute("gen_ai.operation.name", "chat")
                current.set_attribute("gen_ai.request.model", self.settings.model)
                current.set_attribute("gen_ai.agent.name", agent or f"matchsense-{role}")
            try:
                response = client.responses.create(**kwargs)
            except Exception as exc:
                if agent:  # fall back from the agent reference to a direct call, once
                    log.warning("agent call for %s failed (%s); using a direct call", role, exc)
                    self._agents.pop(role, None)
                    return self.complete(role, instructions, payload, schema)
                self.usage.failures += 1
                raise ModelError(f"{role}: {type(exc).__name__}: {exc}") from exc
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                if span:
                    current.set_attribute("gen_ai.usage.input_tokens", int(getattr(usage, "input_tokens", 0) or 0))
                    current.set_attribute("gen_ai.usage.output_tokens", int(getattr(usage, "output_tokens", 0) or 0))
        finally:
            if span:
                span.__exit__(None, None, None)
        self.usage.calls += 1
        self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
        self.usage.latency_ms += (time.perf_counter() - start) * 1000
        text = getattr(response, "output_text", "") or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            self.usage.failures += 1
            raise ModelError(f"{role}: response was not JSON") from exc
        if not isinstance(parsed, dict):
            raise ModelError(f"{role}: response was not an object")
        return parsed


def model_from_env() -> ChatModel | None:
    """The configured remote model, or None to use the deterministic local narration."""
    mode = os.environ.get("MATCHSENSE_NARRATION", "auto").lower()
    if mode == "local":
        return None
    settings = FoundrySettings.from_env()
    if settings is None:
        if mode == "foundry":
            log.warning("MATCHSENSE_NARRATION=foundry but FOUNDRY_PROJECT_ENDPOINT/FOUNDRY_MODEL_DEPLOYMENT are not set")
        return None
    return FoundryModel(settings)
