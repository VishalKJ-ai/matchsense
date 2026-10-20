"""Tracking-grade synthetic football match simulator."""

from .engine import MatchSimulator, SimulationResult, simulate
from .io import StoredMatch, load_match, write_match
from .squads import PlayerSpec, TeamSpec, TeamStyle, load_default_teams, load_team
from .storyline import STORYLINES, Storyline

__all__ = [
    "MatchSimulator", "SimulationResult", "simulate",
    "StoredMatch", "load_match", "write_match",
    "PlayerSpec", "TeamSpec", "TeamStyle", "load_default_teams", "load_team",
    "STORYLINES", "Storyline",
]
