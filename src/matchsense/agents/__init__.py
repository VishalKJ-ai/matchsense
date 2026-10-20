"""Explain: the narration crew (tactician, narrator, fact-checker, recapper) on Microsoft Foundry, with a local fallback."""

from .crew import AGENT_INSTRUCTIONS, Blackboard, Narration, NarrationCrew
from .models import FoundryModel, FoundrySettings, ModelError, model_from_env

__all__ = ["AGENT_INSTRUCTIONS", "Blackboard", "Narration", "NarrationCrew", "FoundryModel", "FoundrySettings",
           "ModelError", "model_from_env"]
