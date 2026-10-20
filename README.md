# MatchSense

Explainable, personalised match intelligence for every fan, built on a
tracking-grade synthetic football simulator. Working name, built for the
Microsoft x Premier League *Inside the Game* hackathon. All clubs, players and
matches are fictional.

## Status

| Stage | What | State |
| --- | --- | --- |
| Data | Tracking-grade match simulator (this repo, `matchsense.sim`) | done |
| Interpret | Live stats engine: pass difficulty, speeds, pressing, momentum, control vs chaos, key moments (`matchsense.analytics`) | done |
| Explain | Agents on Microsoft Foundry: analyst, tactician, narrators, producer, fact-checker | planned |
| Render | Timed, machine-readable overlay cues, a producer that schedules them, and a live 2D broadcast view (`matchsense.render`, `matchsense.web`) | done |
| Personalise | Casual, analyst, player-focus, club and audio-description modes; languages follow with the agents | done (English) |

## The synthetic data

Most synthetic football data is a list of random events. This simulator plays
the match: all 22 players and the ball move ten times a second under per-player
limits on speed and acceleration, and every event comes out of that movement.

- **Players decide by expected value.** Each pass option is scored with a
  geometric model (passer accuracy under pressure, interception risk along the
  lane, who reaches the ball first), plus an expected-threat gain. Shots,
  carries, dribbles, crosses, through balls and clearances compete in the same
  softmax, with a temperature that reflects the player's skill.
- **Team shape** shifts with the ball, presses up to a style-defined line,
  marks goal-side, squeezes the ball-goal corridor near its own box, and holds
  an offside line. Forwards time runs in behind it, so offsides happen
  physically, not by dice roll.
- **Set pieces and restarts** are all modelled: throw-ins, corners, goal kicks,
  free kicks, penalties, kick-offs, substitutions, cards and added time.
- **Storylines** change team styles mid-match (for example a switch to a high
  press at the hour). A seed search keeps only the matches whose result fits
  the story. The scripted changes are logged as ground truth, so we can test
  whether the analytics spot them from the data alone.
- **Ground truth** (`truth.jsonl`) records the simulator's own pass and shot
  probabilities, for checking the stats engine's pass-difficulty and xG
  estimates.

### Realism

`matchsense-sim report` checks a match against typical top-flight ranges, using
only the published events and tracking. Across 12 random seeds, matches pass
about 36 of 38 checks on average. Typical per-team figures: about 15 shots,
1.8 goals, 480 passes at 81% completion, 4 corners and 16 crosses. Players
cover 10–12 km with top speeds of 29–33 km/h, and the ball is in play for
about 60 minutes.

### Files

A match is a folder:

| File | Contents |
| --- | --- |
| `match.json` | teams, line-ups, players, venue, referee, score, storyline |
| `events.jsonl` | time-ordered events; the contract is in `src/matchsense/schema.py` |
| `tracking.npz` | `xy[frame, slot, 2]`, `ball[frame, 3]`, `player_index`, `owner`, `state`, `t`, `period` at 10 Hz |
| `truth.jsonl` | simulator-internal probabilities, for evaluation only |

Coordinates are metres on a 105 x 68 pitch, origin bottom-left. Each event's
`frame` is the key for synchronising overlays with tracking.

## The live stats engine

`matchsense.analytics` consumes the tracking and event feeds as they arrive and
turns them into **signals**: verifiable facts the explanation layer can narrate.
Every number a narrator may quote is in a signal's `facts`, and its `evidence`
lists the events it came from, so a fact-checker can confirm every number.

- **Live feed replay.** Tracking arrives at 10 Hz. Events arrive two seconds
  after they finish, as with commercial feeds. A whole match is analysed in
  about 3.5 s, so the engine has plenty of headroom for real time.
- **Pass difficulty (0-100).** A logistic model on features a broadcast feed
  provides at the moment of the pass: length, height, pressure on the passer,
  defenders in the lane, marking on the receiver, forward progress and pass
  type. Its coefficients give the reasons ("tight marking on the receiver,
  ball in the air"). Trained on 13 simulated matches (12,400 passes) and tested
  on 3 held-out matches: AUC 0.83, well calibrated (predicted 71% vs actual 71%
  completion in the middle band), and 0.63 correlation with the simulator's
  hidden probabilities, which it never sees.
- **Physical.** Distance, top speed and sprints (over 25.2 km/h for at least 1 s)
  from half-second smoothed tracking. On-screen speed tags fire above 31 km/h,
  and distance tags at 10 and 12 km.
- **Tactics in rolling windows.** Possession, pressing, PPDA (opponent passes
  allowed per defensive action), line height, tempo, directness, field tilt,
  momentum, and a control-vs-chaos index (turnovers, loose balls, duels and
  possession length, scaled to measured percentiles).
- **Tactical shift detection.** Compares the last 8 minutes with the 18 before.
  A pressing change is reported with high confidence only when two
  tracking-based measures move together: line height while the opponent builds
  from the back, and the distance from the nearest presser to the ball
  carrier. On 16 simulated matches (`scripts/evaluate_shifts.py`) it caught 14
  of 16 scripted pressing changes, with a median delay of 7.5 minutes, and
  raised 1 false alarm in 8 matches with no scripted change.
- **Key moments.** Goals with context (equaliser, against the run of play,
  passes in the move, shot speed), big chances, saves, key passes, hard passes,
  shot speeds over 100 km/h, momentum swings, chaos and control spells, cards,
  substitutions, and half and full-time summaries.

```bash
.venv/bin/matchsense-live analyse data/matches/comeback
```

```bash
.venv/bin/matchsense-live replay data/matches/comeback --speed 30
```

`analyse` writes `insights.jsonl` (signals) and `summary.json` (final
snapshot and player table) next to the match.

## Overlays, personalisation and the live view

**Overlay cues.** The producer (`matchsense.render.producer`) turns signals into
cues a graphics renderer can use as they are: layer (lower third or tag),
start and end frame, priority, anchor (screen region, a player, or a pitch
point), headline, body, template data and evidence. Like a broadcast graphics
operator, it enforces a budget per layer, lets a much more important cue bump
a lesser one, queues the rest and drops stale ones. The format is at
`/api/schema/cues`.

**One match, five experiences.** Each viewer's profile shapes what they get:

| Mode | What it shows |
| --- | --- |
| Casual | About 18 overlays a match, in plain words: "a chance scored about 1 in 3 times" |
| Analyst | About 100 cues: PPDA, line height, press distance, xG, pass difficulty, momentum, and a full signal log |
| Player focus | Pick any player: their sprints, distance milestones, passes and chances, a live stats card and a focus ring on the pitch |
| My club | Your team's moments come first, written from your side ("GOAL! IRONVALE!" vs a muted "Goal" for the opposition) |
| Audio description | A spoken track for blind and partially sighted fans: who has the ball and where, every incident, score reminders. Goals interrupt routine lines |

Templates use only values from the signal, so every number on screen traces
back to events. They are also the fast path the narrator agents will improve on.

**Live view.** `matchsense-web` streams the whole pipeline per viewer over a
websocket: feed replay, analytics, producer and describer, paced in real time
(1x to 32x). The browser draws a 2D broadcast-style pitch with ball-carrier
name tags and speeds, a scorebug, lower thirds, tags anchored to players, a
momentum strip, timeline markers for goals, cards and tactical shifts, and
speaks the audio description with the browser's speech engine.

```bash
.venv/bin/matchsense-web --port 8000
```

Then open http://localhost:8000. Try `?mode=player&player=WMA-07&start=84`
for Leon Hartley's late double, or `?mode=analyst&start=56` for the pressing
change.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/matchsense-sim generate --storyline comeback --search 60 --out data/matches/comeback --report
```

```bash
.venv/bin/python -m pytest -q
```

`matchsense-sim storylines` lists the built-in storylines. The scripts in
`scripts/` are the tuning tools used to calibrate the simulator.

## Layout

```
data/squads/            fictional squads (Ironvale FC, Westmarsh Athletic)
data/matches/           generated matches
src/matchsense/schema.py  event contract shared by every stage
src/matchsense/sim/     simulator: engine, models, pitch, squads, storylines, report, CLI
src/matchsense/live/    feed replay and live CLI
src/matchsense/analytics/  live stats engine: passing, physical, tactical, signals
src/matchsense/render/  overlay cues, producer, viewer profiles, audio description
src/matchsense/web/     FastAPI service, live websocket session, browser frontend
tests/                  models, full-match invariants, analytics, render and websocket tests
```
