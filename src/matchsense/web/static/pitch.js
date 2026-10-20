// 2D broadcast-style pitch: grass, markings, players, ball, name tags.
// Pitch coordinates are metres (105 x 68, origin bottom-left); y is drawn upwards.

const L = 105, W = 68, MARGIN = 4;

export class PitchView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.bg = document.createElement("canvas");
    this.teams = null;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  setTeams(home, away) {
    this.teams = { home, away };
  }

  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    const fullL = L + 2 * MARGIN, fullW = W + 2 * MARGIN;
    this.scale = Math.min(this.w / fullL, this.h / fullW);
    this.ox = (this.w - L * this.scale) / 2;
    this.oy = (this.h - W * this.scale) / 2;
    this.drawBackground();
  }

  toScreen(x, y) {
    return [this.ox + x * this.scale, this.oy + (W - y) * this.scale];
  }

  drawBackground() {
    const c = this.bg, dpr = this.dpr;
    c.width = this.canvas.width; c.height = this.canvas.height;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#1b6236";
    g.fillRect(0, 0, this.w, this.h);
    const stripes = 14, sw = (L * this.scale) / stripes;
    for (let i = 0; i < stripes; i++) {
      g.fillStyle = i % 2 ? "#2a8247" : "#25783f";
      g.fillRect(this.ox + i * sw, this.oy, sw + 0.5, W * this.scale);
    }
    const vignette = g.createRadialGradient(this.w / 2, this.h / 2, this.h * 0.2, this.w / 2, this.h / 2, this.w * 0.75);
    vignette.addColorStop(0, "#0000"); vignette.addColorStop(1, "#0004");
    g.fillStyle = vignette; g.fillRect(0, 0, this.w, this.h);

    const s = this.scale;
    g.strokeStyle = "rgba(255,255,255,0.82)";
    g.lineWidth = Math.max(1.2, 0.12 * s);
    const P = (x, y) => this.toScreen(x, y);
    const rect = (x0, y0, x1, y1) => {
      const [a, b] = P(x0, y1), [c2, d] = P(x1, y0);
      g.strokeRect(a, b, c2 - a, d - b);
    };
    rect(0, 0, L, W);
    g.beginPath(); let [hx, hy] = P(L / 2, 0); g.moveTo(hx, hy); [hx, hy] = P(L / 2, W); g.lineTo(hx, hy); g.stroke();
    const [cx, cy] = P(L / 2, W / 2);
    g.beginPath(); g.arc(cx, cy, 9.15 * s, 0, Math.PI * 2); g.stroke();
    g.fillStyle = "rgba(255,255,255,0.85)";
    g.beginPath(); g.arc(cx, cy, Math.max(1.5, 0.25 * s), 0, Math.PI * 2); g.fill();
    for (const side of [0, 1]) {
      const x0 = side ? L - 16.5 : 0, x1 = side ? L : 16.5;
      rect(x0, W / 2 - 20.16, x1, W / 2 + 20.16);
      const g0 = side ? L - 5.5 : 0, g1 = side ? L : 5.5;
      rect(g0, W / 2 - 9.16, g1, W / 2 + 9.16);
      const spotX = side ? L - 11 : 11;
      const [sx, sy] = P(spotX, W / 2);
      g.beginPath(); g.arc(sx, sy, Math.max(1.5, 0.25 * s), 0, Math.PI * 2); g.fill();
      g.beginPath();
      const a = Math.acos(5.5 / 9.15);
      if (side) g.arc(sx, sy, 9.15 * s, Math.PI - a, Math.PI + a);
      else g.arc(sx, sy, 9.15 * s, -a, a);
      g.stroke();
      const [gx, gy0] = P(side ? L : 0, W / 2 + 3.66), [, gy1] = P(side ? L : 0, W / 2 - 3.66);
      g.fillStyle = "rgba(255,255,255,0.18)";
      g.fillRect(side ? gx : gx - 1.6 * s, gy0, 1.6 * s, gy1 - gy0);
      g.strokeRect(side ? gx : gx - 1.6 * s, gy0, 1.6 * s, gy1 - gy0);
      g.fillStyle = "rgba(255,255,255,0.85)";
    }
  }

  // frame: {xy: [[x,y]|null x22], ball: [x,y,z], owner}
  // players: slot -> {id, number, short_name, team, side, position}
  render(frame, players, opts = {}) {
    const g = this.ctx, s = this.scale;
    if (!this.bg.width || !this.bg.height) { this.resize(); if (!this.bg.width) return; }
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(this.bg, 0, 0);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    if (!frame || !this.teams) return;
    const r = Math.max(9, 1.3 * s);
    const order = [...frame.xy.keys()].sort((a, b) => (frame.xy[b]?.[1] ?? 0) - (frame.xy[a]?.[1] ?? 0));
    for (const slot of order) {
      const p = frame.xy[slot], pl = players[slot];
      if (!p || !pl) continue;
      const [x, y] = this.toScreen(p[0], p[1]);
      const team = this.teams[pl.side];
      const gk = pl.position === "GK";
      const focus = opts.focusPlayer === pl.id;
      const dim = opts.highlightTeam && opts.highlightTeam !== pl.team;
      g.globalAlpha = dim ? 0.55 : 1;
      if (focus) {
        const pulse = 1 + 0.25 * Math.sin(performance.now() / 220);
        g.beginPath(); g.arc(x, y, r * 1.9 * pulse, 0, Math.PI * 2);
        g.fillStyle = "rgba(57,208,196,0.22)"; g.fill();
        g.lineWidth = 2; g.strokeStyle = "#39d0c4"; g.stroke();
      }
      g.beginPath(); g.ellipse(x + r * 0.15, y + r * 0.35, r * 0.95, r * 0.55, 0, 0, Math.PI * 2);
      g.fillStyle = "rgba(0,0,0,0.28)"; g.fill();
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = gk ? team.colors.secondary : team.colors.primary;
      g.fill();
      g.lineWidth = Math.max(1.5, r * 0.22);
      g.strokeStyle = gk ? team.colors.primary : team.colors.secondary;
      g.stroke();
      g.fillStyle = gk ? team.colors.primary : (team.colors.text || "#fff");
      g.font = `700 ${Math.round(r * 1.05)}px "Barlow Condensed", sans-serif`;
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(String(pl.number), x, y + r * 0.06);
      g.globalAlpha = 1;
    }
    // name tags: ball carrier and focus player
    const tagged = new Set();
    if (frame.owner >= 0) tagged.add(frame.owner);
    if (opts.focusPlayer) {
      const slot = players.findIndex((p) => p && p.id === opts.focusPlayer);
      if (slot >= 0) tagged.add(slot);
    }
    for (const slot of tagged) {
      const p = frame.xy[slot], pl = players[slot];
      if (!p || !pl) continue;
      const [x, y] = this.toScreen(p[0], p[1]);
      this.label(x, y - r - 6, pl.short_name, this.teams[pl.side].colors.primary, slot === frame.owner);
    }
    // ball with height
    const [bx, by, bz] = frame.ball;
    const [gx, gy] = this.toScreen(bx, by);
    const lift = (bz || 0) * s * 0.55;
    const br = Math.max(4.5, 0.55 * s) * (1 + (bz || 0) * 0.06);
    g.beginPath(); g.ellipse(gx, gy, br * 0.9, br * 0.5, 0, 0, Math.PI * 2);
    g.fillStyle = `rgba(0,0,0,${Math.max(0.12, 0.38 - (bz || 0) * 0.04)})`; g.fill();
    g.beginPath(); g.arc(gx, gy - lift, br, 0, Math.PI * 2);
    g.fillStyle = "#ffffff"; g.fill();
    g.lineWidth = 1.5; g.strokeStyle = "#111"; g.stroke();
  }

  label(x, y, text, color, strong) {
    const g = this.ctx;
    g.font = `600 ${strong ? 16 : 15}px "Barlow Condensed", sans-serif`;
    const w = g.measureText(text).width + 16, h = 22;
    g.fillStyle = "rgba(10,15,22,0.88)";
    roundRect(g, x - w / 2, y - h, w, h, 4); g.fill();
    g.fillStyle = color; g.fillRect(x - w / 2, y - h, 3, h);
    g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(text.toUpperCase(), x + 1.5, y - h / 2 + 0.5);
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

export function lerpFrame(a, b, alpha) {
  if (!a) return b;
  if (!b) return a;
  const xy = b.xy.map((p, i) => {
    const q = a.xy[i];
    if (!p || !q) return p;
    return [q[0] + (p[0] - q[0]) * alpha, q[1] + (p[1] - q[1]) * alpha];
  });
  const ball = b.ball.map((v, i) => a.ball[i] + (v - a.ball[i]) * alpha);
  return { ...b, xy, ball };
}
