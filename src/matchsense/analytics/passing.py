"""Pass difficulty: how likely a pass was to be completed, given what was on the pitch when it was played.

The features come only from the event and the tracking frame at the moment of
the pass, so the same model would run on a real broadcast feed. Coefficients
are learned by logistic regression on simulated matches (see
scripts/train_pass_model.py) and shipped as JSON next to this file.

Difficulty is reported 0-100: 100 * (1 - p_complete). The coefficients make
the reasons explainable: "hard because two defenders were in the lane and the
receiver was tightly marked".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .context import MatchContext

FEATURES = ("bias", "distance", "lofted", "pressure", "lane", "marking", "forward", "cross", "through", "restart")
FEATURE_LABELS = {
    "distance": "pass length",
    "lofted": "ball in the air",
    "pressure": "pressure on the passer",
    "lane": "defenders in the passing lane",
    "marking": "tight marking on the receiver",
    "forward": "forward progress",
    "cross": "cross into the box",
    "through": "ball in behind",
    "restart": "set-piece restart",
}
MODEL_PATH = Path(__file__).with_name("models") / "pass_difficulty.json"
RESTARTS = {"throw_in", "goal_kick", "kick_off", "free_kick"}


def pass_features(ev: dict, xy: np.ndarray, ctx: MatchContext) -> np.ndarray:
    """Feature vector for one pass event, using the tracking frame `xy` (22 x 2) at the moment of the pass."""
    start = np.array([ev["location"]["x"], ev["location"]["y"]])
    end = np.array([ev["end_location"]["x"], ev["end_location"]["y"]])
    attrs = ev["attributes"]
    lofted = attrs.get("height") == "high"
    team = ev["team"]
    opp = xy[ctx.slots(ctx.other(team))]
    opp = opp[np.isfinite(opp[:, 0])]
    seg = end - start
    length = float(np.hypot(*seg))
    pressure = math.exp(-float(np.hypot(*(opp - start).T).min()) / 2.0) if len(opp) else 0.0
    marking = math.exp(-float(np.hypot(*(opp - end).T).min()) / 2.5) if len(opp) else 0.0
    lane = 0.0
    if length > 0.5 and len(opp):
        t = ((opp - start) @ seg) / (length * length)
        perp = np.hypot(*(opp - (start + t[:, None] * seg)).T)
        mask = (t > 0.08) & (t < 0.95)
        lane = float(np.exp(-(perp[mask] ** 2) / (2 * 1.5**2)).sum()) * (0.4 if lofted else 1.0)
    rel_s = ctx.to_rel(start, team, ev["period"])
    rel_e = ctx.to_rel(end, team, ev["period"])
    return np.array([
        1.0,
        length / 30.0,
        float(lofted),
        pressure,
        lane,
        marking,
        (rel_e[0] - rel_s[0]) / 30.0,
        float(attrs.get("kind") == "cross"),
        float(attrs.get("kind") == "through"),
        float(attrs.get("set_piece") in RESTARTS),
    ])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 40) -> np.ndarray:
    """L2-regularised logistic regression by iteratively reweighted least squares (no bias penalty)."""
    w = np.zeros(X.shape[1])
    penalty = np.full(X.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w)
        weight = p * (1 - p)
        hessian = X.T @ (X * weight[:, None]) + np.diag(penalty)
        grad = X.T @ (p - y) + penalty * w
        step = np.linalg.solve(hessian, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class PassDifficultyModel:
    coef: np.ndarray
    metrics: dict

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "PassDifficultyModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        assert tuple(raw["features"]) == FEATURES, "model features do not match this code"
        return cls(np.array(raw["coef"], dtype=float), raw.get("metrics", {}))

    def save(self, path: str | Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": list(FEATURES), "coef": [round(float(c), 5) for c in self.coef], "metrics": self.metrics}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def p_complete(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(features @ self.coef)

    def difficulty(self, features: np.ndarray) -> int:
        return int(round(100.0 * (1.0 - float(self.p_complete(features)))))

    def reasons(self, features: np.ndarray, top: int = 2) -> list[str]:
        """The features that made this pass harder than an average one, largest first."""
        contrib = -(features * self.coef)[1:]
        names = FEATURES[1:]
        ranked = sorted(((c, n) for c, n in zip(contrib, names) if c > 0.15), reverse=True)
        return [FEATURE_LABELS[n] for _, n in ranked[:top]]
