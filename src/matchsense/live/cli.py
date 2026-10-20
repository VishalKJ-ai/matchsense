"""Command line for the live analytics.

    matchsense-live analyse data/matches/comeback          # whole match, as fast as possible
    matchsense-live replay data/matches/comeback --speed 30  # paced like a broadcast, 30x real time
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..analytics.engine import LiveAnalytics, analyse
from ..sim.io import load_match
from .feed import replay


def _describe(s, ctx) -> str:
    names = ", ".join(ctx.player_name(p) for p in s.players[:2])
    team = ctx.team_name(s.team) if s.team else ""
    f = s.facts
    detail = {
        "goal": lambda: f"{f['situation']}, {f['score'][ctx.home]}-{f['score'][ctx.away]}, xG {f['xg']}",
        "tactical_shift": lambda: f"{f['label']} ({f['measure']} {f['before']} -> {f['after']})",
        "big_chance": lambda: f"xG {f['xg']} {f['outcome']}",
        "save": lambda: f"xG faced {f['xg_faced']}, {f['shot_speed_kmh']} km/h",
        "hard_pass": lambda: f"difficulty {f['difficulty']} ({', '.join(f['reasons']) or 'tight window'})",
        "key_pass": lambda: f"shot xG {f['shot_xg']}",
        "speed_burst": lambda: f"{f['speed_kmh']} km/h",
        "fastest_sprint": lambda: f"{f['speed_kmh']} km/h, fastest of the match",
        "shot_speed": lambda: f"{f['speed_kmh']} km/h",
        "momentum_swing": lambda: f"{f['momentum_before']} -> {f['momentum_now']}",
        "chaos_spell": lambda: f"chaos index {f['chaos_index']}",
        "control_spell": lambda: f"chaos index {f['chaos_index']}, possession {f.get('possession')}%",
    }.get(s.kind, lambda: "")()
    return f"{s.clock:>6}  {s.kind:<18} {team:<10} {names:<28} {detail}"


def cmd_analyse(a: argparse.Namespace) -> int:
    match = load_match(a.path)
    start = time.time()
    analytics = analyse(match, latency=a.latency)
    out = a.out or (Path(a.path) / "insights.jsonl")
    with Path(out).open("w", encoding="utf-8") as f:
        for s in analytics.emitted:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False, default=float) + "\n")
    summary = {"snapshot": analytics.snapshot(), "players": analytics.player_table()}
    (Path(out).parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
                                                   encoding="utf-8")
    print(f"{len(analytics.emitted)} signals in {time.time() - start:.1f}s -> {out}")
    for s in analytics.emitted:
        if s.significance >= a.min_significance:
            print(_describe(s, analytics.ctx))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    match = load_match(a.path)
    analytics = LiveAnalytics(match.metadata)
    wall0 = time.time()
    t0 = None
    for kind, item in replay(match, latency=a.latency):
        if kind == "frame":
            if t0 is None:
                t0 = item.index
            target = (item.index - t0) / 10.0 / a.speed
            delay = target - (time.time() - wall0)
            if delay > 0:
                time.sleep(delay)
            new = analytics.on_frame(item)
        else:
            new = analytics.on_event(item)
        for s in new:
            if s.significance >= a.min_significance:
                print(_describe(s, analytics.ctx), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matchsense-live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    an = sub.add_parser("analyse", help="run the analytics over a whole match and save the signals")
    an.add_argument("path", type=Path)
    an.add_argument("--out", type=Path)
    an.add_argument("--latency", type=float, default=2.0, help="event feed delay in seconds")
    an.add_argument("--min-significance", type=float, default=0.4)
    an.set_defaults(func=cmd_analyse)
    rp = sub.add_parser("replay", help="replay a match in paced real time, printing signals as they happen")
    rp.add_argument("path", type=Path)
    rp.add_argument("--speed", type=float, default=30.0, help="multiple of real time")
    rp.add_argument("--latency", type=float, default=2.0)
    rp.add_argument("--min-significance", type=float, default=0.4)
    rp.set_defaults(func=cmd_replay)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
