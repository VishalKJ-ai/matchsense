"""Teams, players, playing styles and formations.

Squads are loaded from JSON files under data/squads. All clubs and players are
fictional.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

ROLE_GROUP = {
    "GK": "GK",
    "RB": "DEF", "RCB": "DEF", "LCB": "DEF", "LB": "DEF", "CB": "DEF", "FB": "DEF",
    "CDM": "MID", "RDM": "MID", "LDM": "MID", "RCM": "MID", "LCM": "MID", "CM": "MID", "DM": "MID",
    "CAM": "ATT", "AM": "ATT", "RAM": "ATT", "LAM": "ATT", "RW": "ATT", "LW": "ATT", "W": "ATT", "ST": "ATT",
}

# Anchor positions in the team's relative frame (attacking towards x = 105, y = 0
# on the right flank). Each role has an out-of-possession and an in-possession
# anchor, both for the ball on the centre spot; the engine shifts the whole
# shape with the ball.
FORMATIONS: dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]] = {
    "4-3-3": {
        "GK": ((5, 34), (8, 34)),
        "RB": ((27, 12), (47, 6)),
        "RCB": ((24, 26), (31, 22)),
        "LCB": ((24, 42), (31, 46)),
        "LB": ((27, 56), (47, 62)),
        "CDM": ((35, 34), (45, 34)),
        "RCM": ((43, 23), (59, 24)),
        "LCM": ((43, 45), (59, 44)),
        "RW": ((55, 10), (77, 7)),
        "ST": ((59, 34), (82, 34)),
        "LW": ((55, 58), (77, 61)),
    },
    "4-2-3-1": {
        "GK": ((5, 34), (8, 34)),
        "RB": ((26, 12), (45, 6)),
        "RCB": ((23, 26), (30, 22)),
        "LCB": ((23, 42), (30, 46)),
        "LB": ((26, 56), (45, 62)),
        "RDM": ((36, 27), (46, 28)),
        "LDM": ((36, 41), (46, 40)),
        "RAM": ((50, 12), (73, 8)),
        "CAM": ((52, 34), (69, 34)),
        "LAM": ((50, 56), (73, 60)),
        "ST": ((59, 34), (83, 34)),
    },
}

STYLE_FIELDS = (
    "tempo", "directness", "risk", "press_line", "press_intensity",
    "line_height", "width", "crossing", "strength",
)


@dataclass(frozen=True)
class TeamStyle:
    """How a team plays. Storylines change these values during a match.

    tempo: how quickly players move the ball on (1.0 = league average).
    directness: preference for forward and long passes (0 to 1).
    risk: how heavily losing the ball is weighed (higher = more careful).
    press_line: how far up the pitch the team presses, as the x position in its
        own frame where pressing starts (45 = deep block, 95 = all-out press).
    press_intensity: how aggressively a second player joins the press (0 to 1).
    line_height: metres the defensive block sits above (+) or below (-) normal.
    width: lateral spread of the in-possession shape (1.0 = normal).
    crossing: appetite for crossing from wide areas (0 to 1).
    strength: rating points added to every player's technical skills, used by
        storylines to model momentum and confidence.
    """

    tempo: float = 1.0
    directness: float = 0.5
    risk: float = 1.0
    press_line: float = 60.0
    press_intensity: float = 0.5
    line_height: float = 0.0
    width: float = 1.0
    crossing: float = 0.5
    strength: float = 0.0

    def with_overrides(self, overrides: dict[str, float]) -> "TeamStyle":
        unknown = set(overrides) - set(STYLE_FIELDS)
        if unknown:
            raise ValueError(f"unknown style fields: {sorted(unknown)}")
        return dataclasses.replace(self, **overrides)


@dataclass(frozen=True)
class Ratings:
    pace: int
    accel: int
    passing: int
    dribbling: int
    finishing: int
    defending: int
    stamina: int
    aerial: int
    keeping: int = 20


@dataclass(frozen=True)
class PlayerSpec:
    id: str
    name: str
    short_name: str
    number: int
    position: str
    foot: str
    nationality: str
    ratings: Ratings

    @property
    def group(self) -> str:
        return ROLE_GROUP[self.position]


@dataclass(frozen=True)
class TeamSpec:
    id: str
    name: str
    short_name: str
    colors: dict[str, str]
    stadium: str
    formation: str
    style: TeamStyle
    starters: tuple[PlayerSpec, ...]
    roles: tuple[str, ...]
    bench: tuple[PlayerSpec, ...]

    def all_players(self) -> tuple[PlayerSpec, ...]:
        return self.starters + self.bench


def _player(raw: dict, position: str) -> PlayerSpec:
    return PlayerSpec(
        id=raw["id"],
        name=raw["name"],
        short_name=raw.get("short_name", raw["name"].split()[-1]),
        number=int(raw["number"]),
        position=position,
        foot=raw.get("foot", "R"),
        nationality=raw.get("nationality", ""),
        ratings=Ratings(**raw["ratings"]),
    )


def load_team(path: str | Path) -> TeamSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    formation = raw["formation"]
    roles = FORMATIONS[formation]
    starters = tuple(_player(p, p["role"]) for p in raw["starters"])
    starter_roles = tuple(p["role"] for p in raw["starters"])
    if sorted(starter_roles) != sorted(roles):
        raise ValueError(f"{raw['id']}: starters must fill exactly the {formation} roles {sorted(roles)}")
    bench = tuple(_player(p, p["position"]) for p in raw["bench"])
    return TeamSpec(
        id=raw["id"],
        name=raw["name"],
        short_name=raw["short_name"],
        colors=raw["colors"],
        stadium=raw.get("stadium", ""),
        formation=formation,
        style=TeamStyle(**raw.get("style", {})),
        starters=starters,
        roles=starter_roles,
        bench=bench,
    )


DEFAULT_SQUAD_DIR = Path(__file__).resolve().parents[3] / "data" / "squads"


def load_default_teams() -> tuple[TeamSpec, TeamSpec]:
    return load_team(DEFAULT_SQUAD_DIR / "ironvale.json"), load_team(DEFAULT_SQUAD_DIR / "westmarsh.json")
