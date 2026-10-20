"""Print the overlay cues a viewer profile would receive for a stored match.

    python scripts/preview_cues.py data/matches/comeback casual
    python scripts/preview_cues.py data/matches/comeback player WMA-07
    python scripts/preview_cues.py data/matches/comeback club WMA
"""

import sys

from matchsense.analytics.engine import LiveAnalytics
from matchsense.live.feed import replay
from matchsense.render import Producer, Profile
from matchsense.sim.io import load_match

path, mode = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "casual"
extra = sys.argv[3] if len(sys.argv) > 3 else None
profile = Profile(mode=mode, player=extra if mode == "player" else None, team=extra if mode == "club" else None)
match = load_match(path)
analytics = LiveAnalytics(match.metadata)
producer = Producer(analytics.ctx, profile)
cues = []
for kind, item in replay(match):
    if kind == "frame":
        signals = analytics.on_frame(item)
        now = item.index
        cues += producer.tick(now)
    else:
        signals = analytics.on_event(item)
    for s in signals:
        cues += producer.on_signal(s, now)
shown = [c for c in cues if c["k"] == "cue"]
print(f"{mode}: {len(shown)} cues, {sum(c['k'] == 'cue_end' for c in cues)} pre-empted")
for c in shown:
    print(f"{c['clock']:>6} [{c['layer']:<11}] {c['title']}  |  {c['body']}")
