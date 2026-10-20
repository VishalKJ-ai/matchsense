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
| Explain | Narration crew on Microsoft Foundry: tactician, narrator, fact-checker, recapper, with a local fallback (`matchsense.agents`) | done; live on Foundry (gpt-5-mini prompt agents) |
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

## The narration crew (Microsoft Foundry)

```
signal ──► Tactician ──► Narrator ──► Fact-checker ──► on-air text
           why it        one per      every number      ungrounded: one repair,
           matters       audience     from evidence     then the template
period end ──► Recapper ──► Fact-checker ──► recap card
```

- **Tactician** explains why a moment matters. Its analysis is cached on a
  shared blackboard per moment, so it runs once however many viewers there are.
- **Narrator** writes the graphic for one audience: style, club, focus player
  and language (English, Spanish, French, Portuguese, Arabic, Hindi, Japanese,
  Swahili and more). It avoids repeating that audience's recent lines.
- **Fact-checker** is deterministic. Every number in generated text must appear
  in the signal's fact pack, allowing for rounding, percentages, "1 in N" odds
  and before-and-after changes. Failures get one repair request, then the
  template is used.
- **Recapper** writes the half-time and full-time stories.

Every step fails soft, so the broadcast never waits for a model. Each lower
third goes on air at once with a deterministic "why it matters" line. The
crew's text replaces it a moment later when Foundry is configured. With no
Foundry settings, a timeout, an error or a rate limit, the local path stays
on screen.

Models are called through the Foundry project's Responses API with strict JSON
schemas, `store=False`, and Microsoft Entra ID sign-in (managed identity in
Azure, `az login` locally): no keys anywhere. With `FOUNDRY_USE_AGENTS=1`
(the default on Azure), each role is registered as a versioned Foundry prompt
agent (`matchsense-tactician`, `-narrator`, `-recapper`) that owns its
instructions, output schema and reasoning effort, and is called by reference.
Traces go to Application Insights. `/api/narration` reports the backend,
calls, failures and tokens.

On gpt-5-mini with minimal reasoning, a live run over the comeback match's
goal, tactical shift and big chance, for casual, analyst and Spanish viewers,
plus the full-time recap, made 13 calls: every one returned schema-valid text
that passed the fact-check, in about 2 seconds each. At "low" reasoning, most
calls spent their output budget on reasoning and fell back to templates.

## Deploy to Azure

Everything is defined in `infra/` (Bicep) and `azure.yaml` (Azure Developer
CLI). `azd provision` creates one resource group containing:

| Resource | Purpose |
| --- | --- |
| Azure Container Apps | the web service and websocket stream; one warm replica, scales to 3. The template also fits [express](https://learn.microsoft.com/azure/container-apps/express-overview) environments |
| Azure Container Registry | the image when deploying with `azd deploy` (CI publishes to GitHub Container Registry instead) |
| Microsoft Foundry account, project and model deployment | the narration crew (`gpt-5-mini` by default) |
| User-assigned managed identities | keyless access. App: AcrPull, Azure AI User, Monitoring Metrics Publisher. Deploy (optional): GitHub OIDC trust for `main`, Container Apps Contributor on the app, Reader |
| Log Analytics and Application Insights | metrics, logs and traces through OpenTelemetry |
| Budget (optional) | email alerts at 50%, 80% and 100% of a monthly amount |

1. Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
   and the [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd),
   then sign in once and let azd reuse it:

```bash
az login
```

```bash
azd config set auth.useAzCliAuth true
```

2. Create an environment in a region your subscription allows, where your
   model has quota. Check both first:

```bash
az policy assignment list --query "[?displayName=='Allowed resource deployment regions'].parameters"
```

```bash
az cognitiveservices usage list --location francecentral --query "[?contains(name.value,'gpt-5-mini')]" -o table
```

```bash
azd env new matchsense --location francecentral
```

3. Optionally pick another model, set a budget alert, and let a GitHub
   repository deploy from its `main` branch:

```bash
azd env set FOUNDRY_MODEL_NAME gpt-4.1-mini
```

```bash
azd env set BUDGET_CONTACT_EMAIL you@example.com
```

```bash
azd env set GITHUB_REPOSITORY "$(gh api repos/your-name/matchsense --jq '"\(.owner.login)@\(.owner.id)/\(.name)@\(.id)"')"
```

   The value must match the subject in GitHub's OIDC token, which now carries
   immutable IDs (`owner@ownerId/repo@repoId`). Older setups use plain
   `owner/repo`; the deploy job's login error shows the subject GitHub sent.

4. Provision, then deploy the image:

```bash
azd provision
```

```bash
azd deploy
```

`azd deploy` builds the image in Azure Container Registry. Some subscriptions,
including Azure for Students, do not allow those ACR Tasks builds
(`TasksOperationsNotAllowed`), and Container Apps express environments could
not pull from ACR with a managed identity. Use continuous deployment instead:
copy the `AZURE_DEPLOY_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP` and `AZURE_CONTAINER_APP_NAME` values from
`azd env get-values` into the GitHub repository's Actions variables (the first
as `AZURE_CLIENT_ID`). None of them is a secret. Every push to `main` then
builds the image, publishes it to GitHub Container Registry with the
workflow's own token, rolls the app forward with the deploy identity's OIDC
token, and checks `/api/health`. Make the package public once after the first
run (the job links to its settings); Azure pulls it anonymously, and anyone can
run it locally:

```bash
docker run -p 8000:8000 ghcr.io/vishalkj-ai/matchsense
```

Later `azd provision` runs keep the deployed image.

Student subscriptions also limit regions (an "Allowed resource deployment
regions" policy) and model quota (gpt-5-mini, gpt-4.1-mini and o4-mini had
quota on ours; the gpt-6 family had none). New Container Apps environments may
be created as express, which is why the environment sets no workload profiles or
log destination.

To use Foundry from your laptop, copy `FOUNDRY_PROJECT_ENDPOINT` and
`FOUNDRY_MODEL_DEPLOYMENT` from `azd env get-values` into your shell (see
`.env.example`) and start `matchsense-web`; `az login` is the only credential.
Tear everything down with `azd down --purge`.

CI (`.github/workflows/ci.yml`) runs the tests, builds and smoke-tests the
container, compiles the Bicep on every push, and deploys from `main`.

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
src/matchsense/agents/  narration crew: tactician, narrator, fact-checker, recapper; Foundry and local backends
src/matchsense/web/     FastAPI service, live websocket session, browser frontend
infra/, azure.yaml      Azure Container Apps, Foundry and monitoring (Bicep, azd)
Dockerfile              container image for the web service
tests/                  models, full-match invariants, analytics, render and websocket tests
```
