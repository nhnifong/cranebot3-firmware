#!/usr/bin/env python

"""Live view of each spool's tension against the onboard tension regulation in spool_dm.

    python experiments/spool_tension_viz.py
    python experiments/spool_tension_viz.py --log spool_diag_1760000000.jsonl --port 4258

Start a spool log on the robot first: type `spoollog` in the debug box (and `spoollog off`
when done). That turns on every anchor's SPOOL_DIAG and has the observer write each spool's
records to spool_diag_<time>.jsonl in its working directory. This tails the newest such file
in --dir (or the one given with --log, which also replays a finished log), and serves a page
at http://127.0.0.1:4258/ with a panel per line showing:

    - which tension the regulator is aiming for: the floor, the lower payout floor (while
      the motion controller pays line out), a hold target, or regulation off
    - whether the soft mute is stopping the payout, and how far (1 is free, 0 is stopped)
    - the reported tension against that target and its hysteresis band, and the soft mute's
      own reading, which credits less friction
    - the friction both readings take off the motor torque: the holding torque the reported
      tension uses, by the way the shaft last turned, and the capped credit the mute uses
    - the correction speed the regulator adds, and the commanded line speed before and after

Anchors on firmware older than the regulator fields in the diag row send only the first ten;
for those the mode is inferred from the commanded speed without the power line's extra
tension or a hold target, the credit is recovered from the torque, and the page says so.
"""

import argparse
import collections
import glob
import json
import logging
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

N_LINES = 4
# seconds of history the page plots
WINDOW_S = 20.0
# samples per second kept per line, of the spool loop's 50
KEEP_HZ = 25
# spool_dm.default_conf_dm, for drawing; a savevar override in the log header wins
DEFAULTS = {"TENSION_BAND_N": 0.1, "TENSION_FLOOR_N": 0.1, "PAYOUT_TENSION_FLOOR_N": 0.05,
            "PAYOUT_FRICTION_CREDIT_NM": 0.005, "SOFT_MUTE_ENABLED": 1}
MODES = {0: "off", 1: "floor", 2: "payout floor", 3: "hold"}


def newest_log(directory):
    logs = glob.glob(os.path.join(directory, "spool_diag_*.jsonl"))
    return max(logs, key=os.path.getmtime) if logs else None


def sample_from_diag(row, tension_at):
    """One page sample from a SPOOL_DIAG row. tension_at(t) is the line record's tension
    nearest t, for old rows that do not carry it."""
    t, torque, hold, motor_vel, mpr, aim, mute, wanted, reg, mute_tension = row[:10]
    s = {"t": t, "torque": torque, "hold": hold, "mute": mute, "aim": aim, "wanted": wanted,
         "mute_tension": mute_tension, "reg": reg, "vel": motor_vel, "mpr": mpr}
    if len(row) >= 15:
        s.update(tension=row[10], mode=MODES.get(row[11], "?"), target=row[12],
                 credit=row[13], correction=row[14], inferred=False)
    else:
        # mute_tension = -(torque - credit) * 2pi / mpr, with SMOOTH_FACTOR at its default 1
        s.update(tension=tension_at(t), credit=torque + mute_tension * mpr / (2 * math.pi),
                 mode="off" if not reg else ("payout floor" if aim > 0 else "floor"),
                 target=None, correction=None, inferred=True)
    return s


class Spools:
    """The last WINDOW_S of every line, fed from a spool log as it is written."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lines = [collections.deque() for _ in range(N_LINES)]
        self.last_tension = [collections.deque(maxlen=200) for _ in range(N_LINES)]
        self.config = dict(DEFAULTS)
        self.path = None
        self.last_t = None
        self.has_diag = False

    def reset(self, path, header):
        with self.lock:
            for q in self.lines:
                q.clear()
            self.path = path
            self.config = dict(DEFAULTS)
            for k, v in (header.get("component_vars") or {}).items():
                if k in self.config:
                    try:
                        self.config[k] = float(v)
                    except ValueError:
                        pass
            self.has_diag = False

    def add_tick(self, tick):
        with self.lock:
            self.last_t = tick["t"]
            for line_no, line in enumerate(tick.get("lines", [])[:N_LINES]):
                # line records are (t, length, speed, tension)
                for r in line.get("rows") or []:
                    self.last_tension[line_no].append((r[0], r[3]))
                rows = line.get("diag") or []
                if rows:
                    self.has_diag = True
                    recs = self.last_tension[line_no]

                    def tension_at(t, recs=recs):
                        return min(recs, key=lambda r: abs(r[0] - t))[1] if recs else None

                    samples = [sample_from_diag(r, tension_at) for r in rows]
                else:
                    # no diag from this anchor: the tension alone
                    samples = [{"t": r[0], "tension": r[3], "mode": None}
                               for r in line.get("rows") or []]
                q = self.lines[line_no]
                for s in samples:
                    if q and int(s["t"] * KEEP_HZ) == int(q[-1]["t"] * KEEP_HZ):
                        # keep the newest in each slot, but never lose a mute that bit
                        if s.get("mute", 1) > q[-1].get("mute", 1) and "mute" in s:
                            s["mute"] = q[-1]["mute"]
                        q[-1] = s
                    else:
                        q.append(s)
                while q and q[-1]["t"] - q[0]["t"] > WINDOW_S:
                    q.popleft()

    def snapshot(self):
        with self.lock:
            return json.dumps({
                "path": self.path,
                "now": self.last_t,
                "age_s": None if self.last_t is None else time.time() - self.last_t,
                "has_diag": self.has_diag,
                "config": self.config,
                "window_s": WINDOW_S,
                "lines": [list(q) for q in self.lines],
            }).encode()


def follow(spools, path, directory, replay_rate):
    """Feed `spools` from the log at `path`, or from the newest log in `directory`, switching
    to a newer one when it appears. A finished log is replayed at replay_rate x real time."""
    current, handle, waited = None, None, False
    while True:
        target = path or newest_log(directory)
        if target is None:
            if not waited:
                logger.info(f"no spool_diag_*.jsonl in {directory} yet; type spoollog in the debug box")
                waited = True
            time.sleep(1.0)
            continue
        if target != current:
            if handle:
                handle.close()
            handle = open(target)
            header = json.loads(handle.readline() or "{}")
            spools.reset(target, header)
            current = target
            logger.info(f"following {target}")
        line = handle.readline()
        if not line or not line.endswith("\n"):
            if line:
                handle.seek(handle.tell() - len(line))
            time.sleep(0.05)
            continue
        tick = json.loads(line)
        # pace a replay; ticks of a log still being written arrive at their own pace
        if path and replay_rate and time.time() - tick["t"] > 2.0:
            last = spools.last_t
            if last is not None:
                time.sleep(max(0.0, (tick["t"] - last) / replay_rate))
        spools.add_tick(tick)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spool tension</title>
<style>
  :root { --bg:#0e1116; --card:#161b22; --line:#2a313c; --text:#e6e9ee; --muted:#8b95a5;
          --tension:#5aa9ff; --mute:#c0c6d0; --target:#ffaa33; --stop:#ff5d5d; --ok:#3ccf7a; }
  html, body { margin:0; height:100%; overflow:hidden; background:var(--bg); color:var(--text);
               font:14px system-ui, sans-serif; }
  /* everything fits the window: the plots take whatever height is left */
  body { display:flex; flex-direction:column; }
  header { padding:8px 16px; color:var(--muted); display:flex; gap:4px 16px; flex-wrap:wrap; flex:none; }
  header b { color:var(--text); font-weight:600; }
  main { flex:1; min-height:0; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr;
         gap:10px; padding:0 10px 10px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 14px;
          display:flex; flex-direction:column; min-height:0; min-width:0; overflow:hidden; }
  .title { display:flex; justify-content:space-between; align-items:baseline; }
  .title h2 { margin:0; font-size:16px; }
  .badges { display:flex; gap:8px; margin:8px 0; flex-wrap:wrap; }
  .badge { padding:4px 10px; border-radius:999px; font-weight:700; font-size:13px; border:1px solid var(--line); }
  .grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:6px 12px; margin-bottom:8px; }
  .grid > div { min-width:0; }
  .k, .v { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .k { color:var(--muted); font-size:12px; }
  .v { font-size:clamp(14px, 1.7vh, 18px); font-weight:650; font-variant-numeric:tabular-nums; }
  .v small { font-size:12px; color:var(--muted); font-weight:500; }
  .plot { flex:1; min-height:60px; position:relative; }
  .plot canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
  .legend { display:flex; gap:14px; color:var(--muted); font-size:12px; margin-top:4px; white-space:nowrap;
            overflow:hidden; }
  .sw { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:5px; }
  .note { color:var(--muted); font-size:12px; margin-top:4px; white-space:nowrap; overflow:hidden;
          text-overflow:ellipsis; min-height:15px; }
  /* a short window gives the plots the legend's room */
  @media (max-height: 800px) {
    .legend { display:none; }
    .badges { margin:4px 0; }
    .badge { padding:2px 8px; font-size:12px; }
    .card { padding:6px 10px; }
  }
  .stale { color:var(--stop); }
</style>
</head>
<body>
<header id="head">waiting for a spool log…</header>
<main id="cards"></main>
<script>
const NAMES = ["upper (direct)", "lower (indirect)"];
const MODE_COLOUR = { "floor": "#5aa9ff", "payout floor": "#b48cff", "hold": "#ffaa33", "off": "#6b7380" };
const cards = document.getElementById("cards");
for (let i = 0; i < 4; i++) {
  cards.insertAdjacentHTML("beforeend", `
  <section class="card" id="c${i}">
    <div class="title"><h2>Line ${i}</h2><span class="k">anchor ${i >> 1} · ${NAMES[i % 2]}</span></div>
    <div class="badges"><span class="badge" data-f="mode"></span><span class="badge" data-f="muteb"></span><span class="badge" data-f="corrb"></span></div>
    <div class="grid">
      <div><div class="k">tension</div><div class="v" data-f="tension"></div></div>
      <div><div class="k">target ± band</div><div class="v" data-f="target"></div></div>
      <div><div class="k">soft mute's reading</div><div class="v" data-f="mute_tension"></div></div>
      <div><div class="k">line speed told → sent</div><div class="v" data-f="speed"></div></div>
      <div><div class="k">tension credits friction</div><div class="v" data-f="hold"></div></div>
      <div><div class="k">adding</div><div class="v" data-f="hold_n"></div></div>
      <div><div class="k">mute credits friction</div><div class="v" data-f="credit"></div></div>
      <div><div class="k">adding</div><div class="v" data-f="credit_n"></div></div>
    </div>
    <div class="plot"><canvas></canvas></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--tension)"></span>tension</span>
      <span><span class="sw" style="background:var(--mute)"></span>mute's reading</span>
      <span><span class="sw" style="background:var(--target)"></span>target ± band</span>
      <span><span class="sw" style="background:rgba(255,93,93,0.5);height:10px"></span>soft mute holding back payout</span>
      <span>strip: regulator mode</span>
    </div>
    <div class="note" data-f="note"></div>
  </section>`);
}

const f = (i, name) => document.querySelector(`#c${i} [data-f="${name}"]`);
const N = v => v == null ? "–" : v.toFixed(2);
const TWO_PI = 2 * Math.PI;

function badge(el, text, colour) {
  el.textContent = text;
  el.style.color = colour; el.style.borderColor = colour;
  el.style.display = text ? "" : "none";
}

function draw(canvas, samples, now, cfg, windowS) {
  const dpr = devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  const L = 40, R = 8, T = 6, B = 26, strip = 6;
  const pw = W - L - R, ph = H - T - B - strip - 4;
  let hi = 0.6, lo = -0.2;
  for (const s of samples) {
    for (const v of [s.tension, s.mute_tension, s.target]) {
      if (v != null) { hi = Math.max(hi, v + 0.2); lo = Math.min(lo, v - 0.2); }
    }
  }
  const x = t => L + pw * (1 - (now - t) / windowS);
  const y = v => T + ph * (1 - (v - lo) / (hi - lo));

  g.fillStyle = "#11161d"; g.fillRect(L, T, pw, ph);
  // soft mute holding back the payout
  for (let k = 0; k < samples.length - 1; k++) {
    const s = samples[k];
    if (s.mute != null && s.mute < 0.999) {
      g.fillStyle = `rgba(255,93,93,${0.15 + 0.45 * (1 - s.mute)})`;
      g.fillRect(x(s.t), T, Math.max(1, x(samples[k + 1].t) - x(s.t)), ph);
    }
  }
  // gridlines
  g.strokeStyle = "#232a35"; g.lineWidth = 1; g.fillStyle = "#8b95a5"; g.font = "11px system-ui";
  g.textAlign = "right";
  // the finest step that keeps the labels 18 px apart
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10].find(st => ph * st / (hi - lo) >= 18) || 20;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    g.beginPath(); g.moveTo(L, y(v)); g.lineTo(L + pw, y(v)); g.stroke();
    g.fillText(v.toFixed(1), L - 6, y(v) + 4);
  }
  g.textAlign = "center";
  for (let s = 0; s <= windowS; s += 5) g.fillText(s ? `-${s}s` : "now", L + pw * (1 - s / windowS), H - 8);
  g.save(); g.translate(11, T + ph / 2); g.rotate(-Math.PI / 2); g.fillText("N", 0, 0); g.restore();

  g.save();
  g.beginPath(); g.rect(L, T, pw, ph); g.clip();
  // target and its band, as steps
  const band = cfg.TENSION_BAND_N;
  for (let k = 0; k < samples.length - 1; k++) {
    const s = samples[k], t2 = samples[k + 1].t;
    if (s.target == null) continue;
    g.fillStyle = "rgba(255,170,51,0.13)";
    g.fillRect(x(s.t), y(s.target + band), x(t2) - x(s.t), y(s.target - band) - y(s.target + band));
    g.strokeStyle = "#ffaa33"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x(s.t), y(s.target)); g.lineTo(x(t2), y(s.target)); g.stroke();
  }
  function series(key, colour, dash) {
    g.strokeStyle = colour; g.lineWidth = 1.8; g.setLineDash(dash || []);
    g.beginPath(); let on = false;
    for (const s of samples) {
      if (s[key] == null) { on = false; continue; }
      on ? g.lineTo(x(s.t), y(s[key])) : g.moveTo(x(s.t), y(s[key])); on = true;
    }
    g.stroke(); g.setLineDash([]);
  }
  series("mute_tension", "#c0c6d0", [5, 4]);
  series("tension", "#5aa9ff");
  g.restore();
  // regulator mode strip
  for (let k = 0; k < samples.length - 1; k++) {
    const s = samples[k];
    g.fillStyle = MODE_COLOUR[s.mode] || "#2a313c";
    g.fillRect(x(s.t), T + ph + 4, Math.max(1, x(samples[k + 1].t) - x(s.t)), strip);
  }
}

function render(d) {
  const head = document.getElementById("head");
  if (!d.path) return;
  const stale = d.age_s == null || d.age_s > 2;
  const cfg = d.config;
  head.innerHTML = `<span><b>${d.path.split("/").pop()}</b></span>` +
    `<span class="${stale ? "stale" : ""}">${d.age_s == null ? "no ticks yet" : stale ? `last tick ${d.age_s.toFixed(0)} s ago (spoollog off, or a replay)` : "live"}</span>` +
    `<span>floor <b>${cfg.TENSION_FLOOR_N} N</b> · payout floor <b>${cfg.PAYOUT_TENSION_FLOOR_N} N</b> · band ±<b>${cfg.TENSION_BAND_N} N</b> · credit cap <b>${cfg.PAYOUT_FRICTION_CREDIT_NM} N·m</b> · soft mute <b>${cfg.SOFT_MUTE_ENABLED ? "enabled" : "disabled"}</b> (defaults or saved overrides; setvar changes are not logged)</span>`;
  if (!d.has_diag) head.insertAdjacentHTML("beforeend", `<span class="stale">no diag rows: only tension is known</span>`);

  d.lines.forEach((samples, i) => {
    const s = samples[samples.length - 1];
    const card = document.getElementById(`c${i}`);
    if (!s) { badge(f(i, "mode"), "no data", "#6b7380"); badge(f(i, "muteb"), "", ""); badge(f(i, "corrb"), "", ""); return; }
    const diag = s.mode != null;
    const mpr = s.mpr;
    badge(f(i, "mode"), diag ? (s.mode === "off" ? "regulation off" : s.mode) : "no diag",
          MODE_COLOUR[s.mode] || "#6b7380");
    if (!diag || s.mode === "off") badge(f(i, "muteb"), "", "");
    else if (!cfg.SOFT_MUTE_ENABLED && s.mute >= 0.999) badge(f(i, "muteb"), "soft mute disabled", "#6b7380");
    else if (s.mute < 0.01) badge(f(i, "muteb"), "soft mute: payout stopped", "#ff5d5d");
    else if (s.mute < 0.999) badge(f(i, "muteb"), `soft mute: payout ×${s.mute.toFixed(2)}`, "#ffb347");
    else badge(f(i, "muteb"), s.aim > 0 ? "soft mute: clear, paying out" : "soft mute: clear", "#3ccf7a");
    const c = s.correction;
    badge(f(i, "corrb"), c == null || Math.abs(c) < 1e-6 ? "" :
          (c < 0 ? `reeling in ${(-c * 1000).toFixed(0)} mm/s` : `paying out ${(c * 1000).toFixed(0)} mm/s`),
          c < 0 ? "#5aa9ff" : "#ffaa33");

    f(i, "tension").innerHTML = `${N(s.tension)} <small>N</small>`;
    f(i, "target").innerHTML = s.target == null ? (diag && s.mode !== "off" ? "<small>unknown</small>" : "–")
      : `${N(s.target)} <small>± ${cfg.TENSION_BAND_N} N</small>`;
    f(i, "mute_tension").innerHTML = s.mute_tension == null ? "–" : `${N(s.mute_tension)} <small>N</small>`;
    f(i, "speed").innerHTML = s.aim == null ? "–" :
      `${(s.aim * 1000).toFixed(0)} → ${(s.wanted * 1000).toFixed(0)} <small>mm/s</small>`;
    // Both readings are -(torque - friction) * 2pi / mpr in the spool's frame. The tension's
    // friction is the holding torque for the way the shaft last turned, which the diag sends in
    // the motor's frame; recover it in the spool's frame from the reading itself (exact while
    // SMOOTH_FACTOR is its default 1). The mute's is the capped credit.
    const held = s.tension == null || s.torque == null || !mpr ? null : s.tension * mpr / TWO_PI + s.torque;
    const signed = (v, unit, k) => v == null ? "–" : `${v >= 0 ? "+" : ""}${(v * k).toFixed(unit === "N" ? 2 : 1)} <small>${unit}</small>`;
    f(i, "hold").innerHTML = signed(held, "mN·m", 1000);
    f(i, "hold_n").innerHTML = signed(held == null ? null : held * TWO_PI / mpr, "N", 1);
    f(i, "credit").innerHTML = signed(s.credit, "mN·m", 1000);
    f(i, "credit_n").innerHTML = signed(s.credit == null || !mpr ? null : s.credit * TWO_PI / mpr, "N", 1);
    const notes = [];
    if (s.inferred) notes.push("older anchor firmware: mode inferred from the commanded speed, without the power line's extra tension or a hold target; credit recovered from the torque");
    if (diag && s.mode !== "off" && s.tension != null && s.mute_tension != null)
    {
      const gap = s.tension - s.mute_tension;
      notes.push(`the mute reads ${Math.abs(gap).toFixed(2)} N ${gap >= 0 ? "below" : "above"} the reported tension: the difference between the friction each credits`);
    }
    f(i, "note").textContent = notes.join(" · ");
    f(i, "note").title = notes.join("\n");
    draw(card.querySelector("canvas"), samples, d.now, cfg, d.window_s);
  });
}

async function poll() {
  for (;;) {
    try { render(await (await fetch("/data")).json()); } catch (e) {}
    await new Promise(r => setTimeout(r, 200));
  }
}
poll();
</script>
</body>
</html>
"""


def make_handler(spools):
    page = PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body, kind = page, "text/html; charset=utf-8"
            elif self.path == "/data":
                body, kind = spools.snapshot(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=".",
                        help="where the observer writes spool_diag_*.jsonl (its working directory)")
    parser.add_argument("--log", default=None, help="a particular spool log, live or finished")
    parser.add_argument("--replay_rate", type=float, default=1.0,
                        help="with --log, how much faster than real time to replay; 0 loads it all at once")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4258)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spools = Spools()
    threading.Thread(target=follow, args=(spools, args.log, args.dir, args.replay_rate),
                     daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(spools))
    server.daemon_threads = True
    logger.info(f"open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
