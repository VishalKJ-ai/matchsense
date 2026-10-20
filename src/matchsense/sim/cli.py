"""Command line: generate synthetic matches and check how realistic they are.

    matchsense-sim generate --storyline comeback --search 40 --out data/matches/comeback
    matchsense-sim report data/matches/comeback
    matchsense-sim storylines
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .engine import simulate
from .io import load_match, write_match
from .report import format_report, realism_checks
from .squads import DEFAULT_SQUAD_DIR, load_team
from .storyline import STORYLINES


def _outline(args: tuple[str, str, int, str]):
    home_path, away_path, seed, story = args
    result = simulate(load_team(home_path), load_team(away_path), seed=seed, storyline=story)
    return seed, result.outline()


def _find_seed(home: str, away: str, start: int, attempts: int, story: str) -> int | None:
    """Return the lowest seed in [start, start + attempts) whose result fits the storyline."""
    predicate = STORYLINES[story].predicate
    if predicate is None:
        return start
    workers = max(1, (os.cpu_count() or 2) - 1)
    batch = workers * 2
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for lo in range(start, start + attempts, batch):
            seeds = range(lo, min(lo + batch, start + attempts))
            outcomes = list(pool.map(_outline, [(home, away, s, story) for s in seeds]))
            for seed, outline in outcomes:
                print(f"  seed {seed}: {outline.final_score[0]}-{outline.final_score[1]} "
                      f"(HT {outline.half_time_score[0]}-{outline.half_time_score[1]})", file=sys.stderr)
            fits = [seed for seed, outline in outcomes if predicate(outline)]
            if fits:
                return min(fits)
    return None


def cmd_generate(a: argparse.Namespace) -> int:
    seed = a.seed
    if a.search:
        print(f"searching {a.search} seeds for a '{a.storyline}' match...", file=sys.stderr)
        found = _find_seed(a.home, a.away, a.seed, a.search, a.storyline)
        if found is None:
            print("no seed matched the storyline; try a larger --search", file=sys.stderr)
            return 1
        seed = found
    start = time.time()
    result = simulate(load_team(a.home), load_team(a.away), seed=seed, storyline=a.storyline)
    out = write_match(result, a.out)
    m = result.metadata
    print(f"{m['teams']['home']['name']} {m['score']['home']}-{m['score']['away']} {m['teams']['away']['name']} "
          f"(seed {seed}, {len(result.events)} events, {m['frames']} frames, {time.time() - start:.1f}s) -> {out}")
    if a.report:
        checks, detail = realism_checks(result.metadata, result.events, result.tracking)
        print(format_report(result.metadata, checks, detail))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    match = load_match(a.path)
    checks, detail = realism_checks(match.metadata, match.events, match.tracking)
    print(format_report(match.metadata, checks, detail))
    return 0 if all(c.ok for c in checks) else 2


def cmd_storylines(_: argparse.Namespace) -> int:
    for name, story in STORYLINES.items():
        print(f"{name}: {story.description}")
        for p in story.phases:
            print(f"    {p.minute:>3}'  {('home', 'away')[p.team]:<4}  {p.note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matchsense-sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="simulate a match and write it to a folder")
    g.add_argument("--home", default=str(DEFAULT_SQUAD_DIR / "ironvale.json"))
    g.add_argument("--away", default=str(DEFAULT_SQUAD_DIR / "westmarsh.json"))
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--storyline", default="balanced", choices=sorted(STORYLINES))
    g.add_argument("--search", type=int, default=0, help="try this many seeds and keep the first that fits the storyline")
    g.add_argument("--out", type=Path, default=Path("data/matches/latest"))
    g.add_argument("--report", action="store_true", help="print the realism report")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("report", help="check a generated match against realistic ranges")
    r.add_argument("path", type=Path)
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("storylines", help="list built-in storylines")
    s.set_defaults(func=cmd_storylines)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
