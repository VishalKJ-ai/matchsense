"""The fact-checker's deterministic core: every number in generated text must come from the evidence.

A narrator may round or reformat, so each number in the fact pack is expanded
into the forms a writer might use: rounded values, percentages of shares,
"1 in N" odds, and the change between a before and after pair. Small counts
(0 to 11) are always allowed, for phrases like "10 men" or "two goals".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
ALWAYS_OK = {float(n) for n in range(0, 12)}


def _numbers_in(value, out: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, str):
        out.extend(float(m.group().replace(",", ".")) for m in NUMBER.finditer(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            _numbers_in(v, out)
        before, after = value.get("before"), value.get("after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            out.append(abs(float(after) - float(before)))
        for key, v in value.items():
            if key.endswith("_before"):
                other = value.get(key[: -len("_before")] + "_after")
                if isinstance(v, (int, float)) and isinstance(other, (int, float)):
                    out.append(abs(float(other) - float(v)))
    elif isinstance(value, (list, tuple)):
        for v in value:
            _numbers_in(v, out)


def allowed_numbers(pack: dict) -> set[float]:
    raw: list[float] = []
    _numbers_in(pack, raw)
    allowed = set(ALWAYS_OK)
    for v in raw:
        for form in (v, round(v), round(v, 1), math.floor(v), math.ceil(v)):
            allowed.add(round(float(form), 1))
        if 0 < v <= 1:
            allowed.add(round(v * 100))
            allowed.add(round(v * 100, 1))
            allowed.add(float(max(2, round(1 / v))))
        if v > 3:
            allowed.add(round(v * 3.6, 1))
            allowed.add(round(v * 3.6))
    return allowed


@dataclass
class CheckResult:
    ok: bool
    ungrounded: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def check_text(texts: dict[str, str], pack: dict, limits: dict[str, int] | None = None) -> CheckResult:
    """Check generated fields against the fact pack: numbers must be grounded and lengths respected."""
    allowed = allowed_numbers(pack)
    ungrounded: list[str] = []
    problems: list[str] = []
    for name, text in texts.items():
        for m in NUMBER.finditer(text or ""):
            v = round(float(m.group().replace(",", ".")), 1)
            if v not in allowed and round(v) not in allowed:
                ungrounded.append(m.group())
        if limits and name in limits and len(text or "") > limits[name]:
            problems.append(f"{name} is {len(text)} characters; the limit is {limits[name]}")
        if not (text or "").strip():
            problems.append(f"{name} is empty")
    return CheckResult(not ungrounded and not problems, ungrounded, problems)
