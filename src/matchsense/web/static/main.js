import { PitchView, lerpFrame } from "./pitch.js";
import { MomentumStrip, Overlays, Panel, Scorebug, timelinePct } from "./overlays.js";
import { Speaker } from "./speech.js";

const $ = (sel) => document.querySelector(sel);
const params = new URLSearchParams(location.search);

const pitch = new PitchView($("#pitch"));
const overlays = new Overlays(pitch, { lowerThird: $("#lower-third"), tags: $("#tags") });
const scorebug = new Scorebug();
const momentum = new MomentumStrip($("#momentum"));
const panel = new Panel();
const speaker = new Speaker();

const S = {
  matchId: params.get("match"),
  mode: params.get("mode") || "casual",
  team: params.get("team") || "",
  player: params.get("player") || "",
  lang: params.get("lang") || "en",
  speed: Number(params.get("speed") || 8),
  ws: null, init: null, teams: {}, playersByIndex: {}, slots: [],
  prev: null, cur: null, recvAt: 0, interval: 0.1, paused: false, ended: false,
};

// ------------------------------------------------------------------ setup

async function boot() {
  const matches = await fetch("/api/matches").then((r) => r.json());
  if (!matches.length) return banner("No matches found. Generate one with matchsense-sim.");
  const chosen = matches.find((m) => m.id === S.matchId) || matches.find((m) => m.id === "comeback") || matches[0];
  S.matchId = chosen.id;
  $("#fixture").textContent = `${chosen.home} v ${chosen.away} · ${chosen.venue} · ${chosen.competition}`;
  $("#speed").value = String(S.speed);
  bindControls();
  connect(Number(params.get("start") || 0));
  requestAnimationFrame(loop);
}

function connect(startMinute = 0) {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); }
  overlays.clear();
  momentum.reset();
  $("#markers").innerHTML = "";
  panel.setMode(S.mode);
  S.prev = S.cur = null; S.ended = false; S.paused = false;
  $("#play").textContent = "❚❚";
  banner(startMinute ? `Catching up to ${startMinute}'…` : "Kick-off…");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({ mode: S.mode, team: S.team, player: S.player, lang: S.lang, speed: S.speed, start: startMinute,
                                  audio: S.mode === "audio" ? 1 : 0 });
  const ws = new WebSocket(`${proto}://${location.host}/ws/matches/${S.matchId}?${q}`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { if (!S.ended) banner("Connection closed. Reload to reconnect."); };
  S.ws = ws;
}

function send(msg) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(msg)); }

function sendProfile() {
  send({ type: "profile", mode: S.mode, team: S.team, player: S.player, lang: S.lang, audio: S.mode === "audio" });
}

// --------------------------------------------------------------- messages

function handle(m) {
  switch (m.k) {
    case "init": return onInit(m);
    case "f": return onFrame(m);
    case "roster": S.slots = m.slots.map((i) => S.playersByIndex[i]); return;
    case "state":
      scorebug.update(m); panel.update(m);
      if (S.cur) momentum.push(timelinePct(S.cur.p, S.cur.t), m.momentum);
      return carrier(m.carrier);
    case "cue": overlays.cue(m); return feedCue(m);
    case "cue_text": overlays.updateCue(m.id, m); return panel.updateItem(m.id, m);
    case "recap": return panel.recap(m);
    case "cue_end": return overlays.remove(m.id);
    case "ev": return onEvent(m);
    case "sig": return onSignal(m);
    case "say":
      speaker.say(m, S.speed);
      if (S.mode === "audio") panel.add(m.clock || clockNow(), m.text, "", `say p${m.priority}`);
      return;
    case "focus": return panel.focus(S.players.find((p) => p.id === m.player), m.stats);
    case "end": S.ended = true; return banner("Full-time · thanks for watching");
    case "error": return banner(m.message);
  }
}

function onInit(m) {
  S.init = m;
  S.players = m.players;
  S.playersByIndex = Object.fromEntries(m.players.map((p) => [p.index, p]));
  const home = m.teams.home, away = m.teams.away;
  S.teams = { [home.id]: { ...home, side: "home" }, [away.id]: { ...away, side: "away" } };
  for (const p of m.players) p.side = p.team === home.id ? "home" : "away";
  pitch.setTeams(home, away);
  overlays.setTeams(S.teams);
  scorebug.setTeams(home, away);
  momentum.setTeams(home, away);
  panel.setTeams(home, away, m.players);
  fillSelectors(home, away, m.players);
  const ls = $("#lang-select");
  for (const opt of ls.options) {
    const available = m.narration?.languages?.includes(opt.value);
    opt.disabled = !available;
    opt.textContent = opt.textContent.replace(" · with Azure", "") + (available ? "" : " · with Azure");
  }
  ls.value = S.lang;
  for (const [x, v] of m.history?.momentum || []) momentum.push(x, v);
  for (const mk of m.history?.markers || []) addMarker(mk);
  if (S.mode === "analyst") {
    for (const s of m.history?.signals || []) onSignal(s);
  } else if (S.mode !== "audio") {
    for (const mo of m.history?.moments || []) panel.add(mo.clock, mo.title, mo.body);
  }
  banner(null);
}

function onFrame(m) {
  const frame = { f: m.f, t: m.t, p: m.p, owner: m.o, ball: m.b, xy: [] };
  for (let i = 0; i < 22; i++) {
    const x = m.xy[2 * i], y = m.xy[2 * i + 1];
    frame.xy.push(x == null ? null : [x, y]);
  }
  const now = performance.now() / 1000;
  if (S.cur) S.interval = Math.max(0.02, Math.min(1, (frame.f - S.cur.f) / (10 * S.speed)));
  S.prev = S.cur; S.cur = frame; S.recvAt = now;
  $("#progress").style.width = `${timelinePct(frame.p, frame.t)}%`;
}

function onEvent(m) {
  const team = S.teams[m.team];
  if (m.type === "goal") addMarker({ kind: "goal", side: team.side, x: timelinePct(m.p, m.t), team: m.team });
  if (m.type === "card") addMarker({ kind: "booking", red: m.attributes.card !== "yellow", x: timelinePct(m.p, m.t) });
  if (m.type === "goal") momentum.mark(timelinePct(m.p, m.t), team.side);
}

function onSignal(s) {
  if (s.kind === "tactical_shift" && s.facts.metric === "press") addMarker({ kind: "shift", x: timelinePct(s.period, s.t), team: s.team });
  if (S.mode !== "analyst") return;
  const team = s.team ? S.teams[s.team].short_name : "";
  panel.add(s.clock, `${s.kind.replace(/_/g, " ")} ${team ? "· " + team : ""}`, summarise(s), "");
}

function feedCue(c) {
  if (S.mode === "analyst" || S.mode === "audio") return;
  if (c.layer === "tag" && S.mode !== "player") return;
  panel.add(c.clock, c.title, c.body, "", c.id, c.why);
}

function summarise(s) {
  const f = s.facts;
  const keep = Object.entries(f).filter(([k, v]) => typeof v !== "object" || v === null).slice(0, 5);
  return keep.map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ");
}

function addMarker(mk) {
  const el = document.createElement("div");
  el.className = `marker ${mk.kind}${mk.red ? " red" : ""}`;
  el.style.left = `${mk.x}%`;
  if (mk.kind === "goal") el.style.background = visibleColor(S.teams[mk.team]?.colors);
  el.title = mk.kind;
  $("#markers").appendChild(el);
}

function visibleColor(colors) {
  if (!colors) return "#fff";
  return colors.primary.toLowerCase() === "#1b2a4a" ? colors.secondary : colors.primary;
}

function carrier(c) {
  const el = $("#carrier");
  if (!c || S.mode === "audio") { el.hidden = true; return; }
  el.hidden = false;
  el.style.borderLeftColor = visibleColor(S.teams[c.team]?.colors);
  el.innerHTML = `${c.name}<small>${c.speed_kmh.toFixed(1)} km/h</small>`;
}

function clockNow() { return $("#sb-clock").textContent; }

function banner(text) {
  const el = $("#banner");
  el.hidden = !text;
  el.textContent = text || "";
}

// ----------------------------------------------------------------- render

function loop() {
  requestAnimationFrame(loop);
  if (!S.cur) { pitch.render(null, [], {}); return; }
  const alpha = S.paused ? 1 : Math.min(1, (performance.now() / 1000 - S.recvAt) / S.interval);
  const frame = lerpFrame(S.prev, S.cur, alpha);
  const opts = {
    focusPlayer: S.mode === "player" ? S.player : null,
    highlightTeam: S.mode === "club" ? S.team : null,
  };
  pitch.render(frame, S.slots, opts);
  const positions = {};
  frame.xy.forEach((p, slot) => { if (p && S.slots[slot]) positions[S.slots[slot].id] = pitch.toScreen(p[0], p[1]); });
  overlays.update(S.cur.f, positions);
}

// --------------------------------------------------------------- controls

function fillSelectors(home, away, players) {
  const ps = $("#player-select"), ts = $("#team-select");
  ps.innerHTML = [home, away].map((t) => `<optgroup label="${t.name}">${players.filter((p) => p.team === t.id)
    .map((p) => `<option value="${p.id}">${p.number} · ${p.name} (${p.position})</option>`).join("")}</optgroup>`).join("");
  ts.innerHTML = [home, away].map((t) => `<option value="${t.id}">${t.name}</option>`).join("");
  if (!S.player) S.player = players.find((p) => p.id === "WMA-07")?.id || players[0].id;
  if (!S.team) S.team = home.id;
  ps.value = S.player; ts.value = S.team;
  applyModeUI();
}

function applyModeUI() {
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("active", b.dataset.mode === S.mode));
  $("#player-select").hidden = S.mode !== "player";
  $("#team-select").hidden = S.mode !== "club";
  document.body.className = `mode-${S.mode}`;
  speaker.enable(S.mode === "audio");
}

function bindControls() {
  document.querySelectorAll("#modes button").forEach((b) => b.addEventListener("click", () => {
    S.mode = b.dataset.mode;
    applyModeUI();
    panel.setMode(S.mode);
    sendProfile();
  }));
  $("#player-select").addEventListener("change", (e) => { S.player = e.target.value; panel.setMode(S.mode); sendProfile(); });
  $("#team-select").addEventListener("change", (e) => { S.team = e.target.value; sendProfile(); });
  $("#lang-select").addEventListener("change", (e) => { S.lang = e.target.value; sendProfile(); });
  $("#speed").addEventListener("change", (e) => { S.speed = Number(e.target.value); send({ type: "speed", value: S.speed }); });
  $("#play").addEventListener("click", () => {
    if (S.ended) return connect(0);
    S.paused = !S.paused;
    send({ type: S.paused ? "pause" : "resume" });
    $("#play").textContent = S.paused ? "▶" : "❚❚";
    $("#play").setAttribute("aria-label", S.paused ? "Play" : "Pause");
  });
  document.querySelectorAll("#jumps button").forEach((b) => b.addEventListener("click", () => connect(Number(b.dataset.jump))));
  $("#timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const pct = ((e.clientX - r.left) / r.width) * 100;
    connect(pct < 50 ? Math.min(44, Math.round(pct)) : Math.round(45 + (pct - 50)));
  });
}

boot();
