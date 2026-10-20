// Overlay graphics driven by cues, plus the scorebug, momentum strip and side panel.

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// x position (0-100) on the match timeline: each half gets 50 units (45 minutes plus up to 5 added).
export function timelinePct(period, t) {
  const inHalf = period === 2 ? t - 2700 : t;
  return (period - 1) * 50 + Math.min(inHalf / 60, 50);
}

// Readable on-screen time in real seconds, however fast the match is being replayed.
const MIN_WALL_S = { lower_third: 6, tag: 2.5 };

export class Overlays {
  constructor(pitch, { lowerThird, tags }) {
    this.pitch = pitch;
    this.lt = lowerThird;
    this.tagsEl = tags;
    this.cues = new Map();
    this.teams = {};
  }

  setTeams(teamsById) { this.teams = teamsById; }

  clear() {
    for (const { el } of this.cues.values()) el.remove();
    this.cues.clear();
  }

  colorsFor(teamId) {
    const t = this.teams[teamId];
    return t ? [t.colors.primary, t.colors.secondary] : ["#39d0c4", "#39d0c4"];
  }

  cue(c) {
    if (c.layer === "lower_third") {
      for (const [id, entry] of this.cues) if (entry.cue.layer === "lower_third") this.remove(id);
      const el = document.createElement("div");
      const [p, s] = this.colorsFor(c.team);
      el.className = `lt kind-${c.kind}`;
      el.innerHTML = `<div class="bar" style="background:linear-gradient(${p} 0 55%, ${s} 55%)"></div>
        <div class="content"><div class="title">${esc(c.title)}</div><div class="body">${esc(c.body)}</div>
        <div class="why"${c.why ? "" : " hidden"}><b>Why it matters</b> <span>${esc(c.why || "")}</span></div>
        <div class="meta">${esc(c.clock)} · evidence: ${c.evidence.length} event${c.evidence.length === 1 ? "" : "s"}<span class="src"></span></div></div>`;
      this.lt.appendChild(el);
      this.cues.set(c.id, { cue: c, el, shown: performance.now() / 1000 });
    } else {
      const el = document.createElement("div");
      el.className = "tag";
      const [p] = this.colorsFor(c.team);
      el.style.borderTopColor = p === "#1B2A4A" || p === "#1b2a4a" ? this.colorsFor(c.team)[1] : p;
      let meter = "";
      if (c.kind === "hard_pass") meter = `<div class="meter"><span style="width:${c.data.facts.difficulty}%"></span></div>`;
      el.innerHTML = `<div class="t">${esc(c.title)}</div><div class="b">${esc(c.body)}</div>${meter}`;
      this.tagsEl.appendChild(el);
      this.cues.set(c.id, { cue: c, el, shown: performance.now() / 1000 });
    }
  }

  // Better text from the narration crew arrives a moment after the cue; swap it in if it is still showing.
  updateCue(id, text) {
    const entry = this.cues.get(id);
    if (!entry || entry.cue.layer !== "lower_third") return;
    const el = entry.el;
    el.querySelector(".title").textContent = text.title;
    el.querySelector(".body").textContent = text.body;
    if (text.why) {
      el.querySelector(".why").hidden = false;
      el.querySelector(".why span").textContent = text.why;
    }
    el.querySelector(".src").textContent = " · written by Foundry";
  }

  remove(id) {
    const entry = this.cues.get(id);
    if (!entry) return;
    this.cues.delete(id);
    entry.el.classList.add("out");
    setTimeout(() => entry.el.remove(), entry.cue.layer === "lower_third" ? 350 : 0);
  }

  update(frameIndex, positions) {
    const now = performance.now() / 1000;
    for (const [id, { cue, el, shown }] of this.cues) {
      if (frameIndex >= cue.end_frame && now - shown >= MIN_WALL_S[cue.layer]) { this.remove(id); continue; }
      if (cue.layer !== "tag") continue;
      let xy = null;
      if (cue.anchor.type === "player") xy = positions[cue.anchor.player];
      else if (cue.anchor.type === "pitch") xy = this.pitch.toScreen(cue.anchor.x, cue.anchor.y);
      if (xy) { el.style.left = `${xy[0]}px`; el.style.top = `${xy[1]}px`; el.style.display = ""; }
      else el.style.display = "none";
    }
  }
}

export class Scorebug {
  constructor() {
    this.home = document.querySelector("#sb-home");
    this.away = document.querySelector("#sb-away");
    this.score = document.querySelector("#sb-score");
    this.clock = document.querySelector("#sb-clock");
  }
  setTeams(home, away) {
    this.ids = [home.id, away.id];
    for (const [el, t] of [[this.home, home], [this.away, away]]) {
      el.querySelector(".name").textContent = t.short_name;
      el.querySelector(".chip").style.background = `linear-gradient(${t.colors.primary} 0 60%, ${t.colors.secondary} 60%)`;
    }
  }
  update(state) {
    if (!this.ids) return;
    this.score.textContent = `${state.score[this.ids[0]]} – ${state.score[this.ids[1]]}`;
    this.clock.textContent = state.clock;
  }
}

export class MomentumStrip {
  constructor(canvas) {
    this.canvas = canvas;
    this.points = new Map(); // rounded timeline x -> momentum (-100..100, + = home)
    this.marks = [];
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  setTeams(home, away) { this.home = home; this.away = away; this.draw(); }
  reset() { this.points.clear(); this.marks = []; this.draw(); }
  push(x, value) { this.points.set(Math.floor(x), value); this.draw(); }
  mark(x, side) { this.marks.push({ x, side }); this.draw(); }
  draw() {
    const c = this.canvas, r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    c.width = r.width * dpr; c.height = r.height * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height, mid = h / 2;
    g.clearRect(0, 0, w, h);
    g.strokeStyle = "#2a394d"; g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
    g.beginPath(); g.moveTo(w / 2, 4); g.lineTo(w / 2, h - 4); g.stroke();
    if (!this.home) return;
    const bw = w / 100;
    const homeCol = visible(this.home.colors), awayCol = visible(this.away.colors);
    for (const [x, v] of this.points) {
      const bh = (Math.abs(v) / 100) * (mid - 6);
      g.fillStyle = v >= 0 ? homeCol : awayCol;
      g.fillRect(x * bw + 0.5, v >= 0 ? mid - bh : mid, Math.max(1, bw - 1), bh);
    }
    for (const m of this.marks) {
      const x = m.x * bw;
      g.fillStyle = "#fff";
      g.beginPath(); g.arc(x, m.side === "home" ? 7 : h - 7, 4, 0, Math.PI * 2); g.fill();
    }
    g.fillStyle = "#8d9aab"; g.font = "11px Inter, sans-serif";
    g.textAlign = "left"; g.fillText(this.home.short_name, 4, 12);
    g.fillText(this.away.short_name, 4, h - 4);
  }
}

function visible(colors) {
  // navy on a dark panel is hard to see; fall back to the secondary colour
  const p = colors.primary.toLowerCase();
  return ["#1b2a4a"].includes(p) ? colors.secondary : colors.primary;
}

export class Panel {
  constructor() {
    this.stats = document.querySelector("#stats");
    this.gauges = document.querySelector("#gauges");
    this.feed = document.querySelector("#feed");
    this.feedTitle = document.querySelector("#feed-title");
    this.statsTitle = document.querySelector("#stats-title");
    this.focusCard = document.querySelector("#card-focus");
    this.focusHead = document.querySelector("#focus-head");
    this.focusGrid = document.querySelector("#focus-grid");
    this.mode = "casual";
  }

  setTeams(home, away, players) { this.home = home; this.away = away; this.players = players; }

  setMode(mode) {
    this.mode = mode;
    this.feed.innerHTML = "";
    this.feedTitle.textContent = { analyst: "Signal log", audio: "Audio description", player: "Their moments",
                                   club: "Your club's match", casual: "Key moments" }[mode];
    this.statsTitle.textContent = mode === "analyst" ? "Live metrics" : "Match so far";
    this.focusCard.hidden = mode !== "player";
  }

  row(label, h, a, fmt = (v) => v, share = null) {
    const hv = h ?? "–", av = a ?? "–";
    let bar = "";
    if (share !== null && h != null && a != null) {
      const tot = (share ? h + a : 1) || 1;
      const hp = share ? (100 * h) / tot : h;
      bar = `<div class="bar"><i style="width:${hp}%;background:${visible(this.home.colors)}"></i><i style="width:${100 - hp}%;background:${visible(this.away.colors)}"></i></div>`;
    }
    return `<div class="stat-row"><div class="v">${h == null ? "–" : fmt(hv)}</div><div class="mid">${esc(label)}${bar}</div><div class="v r">${a == null ? "–" : fmt(av)}</div></div>`;
  }

  update(state) {
    if (!this.home) return;
    const H = this.home.id, A = this.away.id;
    const m = state.match.teams, r = state.recent.teams;
    if (!m[H]) return;
    const pct = (v) => `${Math.round(v)}%`, dec = (v) => v.toFixed(2), one = (v) => v.toFixed(1);
    let html = this.row("Possession", m[H].possession, m[A].possession, pct, true)
      + this.row("Shots", m[H].shots, m[A].shots, (v) => v, true);
    if (this.mode === "analyst") {
      html += this.row("Possession, last 10'", r[H].possession, r[A].possession, pct, true)
        + this.row("xG", m[H].xg, m[A].xg, dec, true)
        + this.row("Pass completion", m[H].pass_completion, m[A].pass_completion, pct)
        + this.row("PPDA, last 10' (lower = harder press)", r[H].ppda, r[A].ppda, one)
        + this.row("Line height, last 10' (m)", r[H].line_height, r[A].line_height, one)
        + this.row("Press distance, last 10' (m)", r[H].press_gap, r[A].press_gap, one);
    } else {
      html += this.row("Passing accuracy", m[H].pass_completion, m[A].pass_completion, pct);
    }
    this.stats.innerHTML = html;
    if (this.mode === "analyst") {
      const leader = state.momentum >= 0 ? this.home.short_name : this.away.short_name;
      this.gauges.innerHTML = `
        <div class="gauge"><div class="label">Control ↔ chaos (10')</div><div class="value">${state.recent.chaos ?? "–"}</div><div class="sub">${esc(state.recent.rhythm ?? "")}</div></div>
        <div class="gauge"><div class="label">Momentum</div><div class="value">${Math.abs(state.momentum)}</div><div class="sub">towards ${esc(leader)}</div></div>`;
    } else {
      this.gauges.innerHTML = "";
    }
  }

  focus(player, stats) {
    if (!stats || !player) return;
    const team = player.team === this.home.id ? this.home : this.away;
    this.focusHead.innerHTML = `<div class="shirt" style="background:${team.colors.primary};border-color:${team.colors.secondary};color:${team.colors.text || "#fff"}">${player.number}</div>
      <div><div class="nm">${esc(player.name)}</div><div class="sub">${esc(team.name)} · ${esc(player.position)} · ${stats.minutes} min</div></div>`;
    const cell = (label, value, sub = "") => `<div class="gauge"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
    this.focusGrid.innerHTML = [
      cell("Distance", `${stats.distance_km.toFixed(1)} km`, `${stats.sprints} sprints`),
      cell("Top speed", `${stats.top_speed_kmh.toFixed(1)}`, "km/h"),
      cell("Touches", stats.touches),
      cell("Passes", `${stats.passes_completed}/${stats.passes}`, stats.pass_accuracy_pct != null ? `${stats.pass_accuracy_pct}%` : ""),
      cell("Pass difficulty", stats.avg_difficulty_completed ?? "–", "avg, completed"),
      cell("Threat added", stats.xt.toFixed(2), "xT"),
      cell("Shots", stats.shots, `xG ${stats.xg.toFixed(2)}`),
      cell("Chances created", stats.key_passes, `${stats.assists} assists`),
      cell("Goals", stats.goals),
    ].join("");
  }

  add(clock, title, body, cls = "", id = null, why = null) {
    const li = document.createElement("li");
    li.className = cls;
    if (id) li.dataset.id = id;
    li.innerHTML = `<div class="h"><span class="c">${esc(clock)}</span><span class="t">${esc(title)}</span></div>`
      + (body ? `<div class="b">${esc(body)}</div>` : "")
      + (why ? `<div class="w">${esc(why)}</div>` : "");
    this.feed.prepend(li);
    while (this.feed.children.length > 120) this.feed.lastChild.remove();
  }

  updateItem(id, text) {
    const li = this.feed.querySelector(`li[data-id="${id}"]`);
    if (!li) return;
    li.querySelector(".t").textContent = text.title;
    const b = li.querySelector(".b");
    if (b) b.textContent = text.body;
    if (text.why) {
      let w = li.querySelector(".w");
      if (!w) { w = document.createElement("div"); w.className = "w"; li.appendChild(w); }
      w.textContent = text.why;
    }
    li.classList.add("ai");
  }

  recap(r) {
    const card = document.querySelector("#card-recap");
    card.hidden = false;
    card.querySelector("h2").textContent = r.period === 1 ? "Half-time recap" : "Full-time recap";
    card.querySelector(".headline").textContent = r.headline;
    card.querySelector(".recap").textContent = r.recap;
    card.querySelector(".src").textContent = r.source === "foundry" ? "Written by the recap agent on Microsoft Foundry" : "Automatic recap";
  }
}
