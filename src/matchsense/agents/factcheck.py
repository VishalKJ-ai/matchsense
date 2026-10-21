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


WORD = re.compile(r"[^\W\d_]+")


def name_problems(texts: dict[str, str], names: set[str]) -> list[str]:
    """Words that look like a club or player name but are not one ("Winmarsh" for "Westmarsh").

    A capitalised word of five or more letters that is not a known name, but shares its first or
    last five letters with a known name of six or more letters and is within three letters of its
    length, counts as a changed name.
    """
    known = {n.lower() for n in names}
    long_names = sorted((n for n in names if len(n) >= 6), key=len, reverse=True)
    problems = []
    for text in texts.values():
        for word in WORD.findall(text or ""):
            low = word.lower()
            if len(word) < 5 or not word[0].isupper() or low in known:
                continue
            match = next((n for n in long_names if abs(len(n) - len(word)) <= 3
                          and (low[:5] == n.lower()[:5] or low[-5:] == n.lower()[-5:])), None)
            if match:
                problems.append(f"wrote '{word}' where the name is '{match}': copy names exactly")
    return problems


def check_text(texts: dict[str, str], pack: dict, limits: dict[str, int] | None = None,
               names: set[str] | None = None) -> CheckResult:
    """Check generated fields against the fact pack: numbers grounded, names unchanged, lengths respected."""
    allowed = allowed_numbers(pack)
    ungrounded: list[str] = []
    problems: list[str] = name_problems(texts, names) if names else []
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
