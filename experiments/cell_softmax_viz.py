#!/usr/bin/env python

"""Live illustration of how the visual servo net picks a point: a softmax over cells, then its centre of mass near the peak.

    python experiments/cell_softmax_viz.py
    python experiments/cell_softmax_viz.py --local_models --port 4250

Reads the gripper MJPEG feed, runs VisualServoNet on it at up to 30 fps, and serves a page
at http://127.0.0.1:4250/ that draws, over the frame the model actually saw:

    - the cell grid: 18 x 32 cells over 1.25x the frame, so the edge cells are off-frame
    - the softmax over all cells, as orange heat
    - the winning cell and the window around it that the centre of mass is taken over
    - the centre of mass, which is the answer, with the grasp axis averaged over the
      same window drawn through it
    - a magnified inset of the window, each cell labelled with its share of the weight

Press p on the page to toggle the 18 x 32 backbone patch grid.

The frame and the prediction are sent together, so the overlay never lags the image.
"""

import argparse
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from nf_robot.ml.visual_servoing.dataset import state_vector
from nf_robot.ml.visual_servoing.model import (
    CANVAS_SCALE, CENTROID_RADIUS, local_centroid, window_average)
from nf_robot.ml.visual_servoing.servo import load_model, prepare_frame

logger = logging.getLogger(__name__)

FPS = 30


class Latest:
    """The newest value of something, with a way to wait for a newer one."""

    def __init__(self):
        self.cond = threading.Condition()
        self.value = None
        self.seq = 0

    def put(self, value):
        with self.cond:
            self.value = value
            self.seq += 1
            self.cond.notify_all()

    def wait_newer(self, seq, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq, timeout=timeout)
            return self.seq, self.value


def read_stream(url, frames: Latest):
    """Keep `frames` holding the newest decoded frame, reconnecting whenever the feed drops."""
    while True:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.warning(f"could not open {url}; retrying")
            time.sleep(1.0)
            continue
        logger.info(f"reading {url}")
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("stream ended; reconnecting")
                break
            frames.put(frame)
        cap.release()
        time.sleep(0.5)


def run_model(model, device, state, frames: Latest, results: Latest):
    """Run the net on the newest frame, over and over, publishing what every cell said."""
    rows, cols = model.grid
    state_t = torch.from_numpy(state_vector(state))[None].to(device)
    seq = 0
    next_tick = time.monotonic()
    slow = 0
    while True:
        # Paced to FPS: a faster feed is sampled, not queued.
        time.sleep(max(0.0, next_tick - time.monotonic()))
        next_tick = max(next_tick + 1 / FPS, time.monotonic())
        seq, bgr = frames.wait_newer(seq, timeout=5.0)
        if bgr is None:
            continue
        start = time.monotonic()
        with torch.no_grad():
            out = model(prepare_frame(bgr, model.image_size, device), state_t)
        logits = out["logits"][:1].float()
        probs = logits[0].flatten().softmax(0)
        winner = int(probs.argmax())
        # The same decode model.decode does for top_k=1, kept in pieces so the window's
        # weights can be drawn: (x, y) in cells, (K,) weights over the (2r+1)^2 window.
        centroid, weights, window = local_centroid(logits, torch.tensor([winner], device=device))
        axis = window_average(out["axis"][:1].float(), weights, window)[0]
        latency = time.monotonic() - start
        slow = slow + 1 if latency > 1 / FPS else 0
        if slow == FPS:
            logger.warning(f"inference takes {latency * 1000:.0f} ms, too slow for {FPS} fps")

        # Stretched to the model's input size, which is what it saw: prepare_frame never crops.
        shown = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        results.put(json.dumps({
            "rows": rows, "cols": cols,
            "token_rows": model.token_grid[0], "token_cols": model.token_grid[1],
            "canvas_scale": CANVAS_SCALE,
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
            "probs": base64.b64encode(probs.cpu().numpy().astype("<f4").tobytes()).decode(),
            "winner": winner,
            "radius": CENTROID_RADIUS,
            "centroid": [round(float(v), 4) for v in centroid[0]],
            # row-major over the window, dy then dx, as local_centroid builds it
            "weights": [round(float(v), 4) for v in weights[0]],
            "axis_rad": float(torch.atan2(axis[0], axis[1]) / 2.0),
            "kappa": float(axis.norm()),
            "latency_ms": round(latency * 1000),
        }).encode())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>One answer per cell</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden; }
  body { display: flex; align-items: center; justify-content: center; }
  canvas { width: min(100vw, calc(100vh * 16 / 9)); aspect-ratio: 16 / 9; display: block; }
</style>
</head>
<body>
<canvas id="c" width="1920" height="1080"></canvas>
<script>
const W = 1920, H = 1080;
const view = document.getElementById("c"), ctx = view.getContext("2d");
// Everything but the kept answer and the inset is drawn here first, so the inset
// magnifies the cells rather than the marker on top of them.
const scene = document.createElement("canvas");
scene.width = W; scene.height = H;
const sc = scene.getContext("2d");

const HEAT = "255,122,0", DOT = "#FF2DAA", WIN = "#00D1FF";
let patches = false;

document.addEventListener("keydown", e => {
  if (e.key === "p") { patches = !patches; }
});

function f32(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

async function poll() {
  let seq = 0;
  for (;;) {
    try {
      const r = await fetch(`/latest?after=${seq}`);
      seq = Number(r.headers.get("X-Seq"));
      if (r.status === 200) {
        const d = await r.json();
        const img = new Image();
        img.src = "data:image/jpeg;base64," + d.jpeg;
        await img.decode();
        render(d, img, f32(d.probs));
      }
    } catch (e) {
      await new Promise(res => setTimeout(res, 500));
    }
  }
}

function render(data, image, probs) {
  const { rows, cols, canvas_scale: s, winner } = data;
  const n = rows * cols, cw = W / cols, ch = H / rows;
  // The cells span -(s-1)/2 .. 1+(s-1)/2 of the frame; the frame sits in the middle.
  const half = (s - 1) / 2;
  const fx = half / s * W, fy = half / s * H, fw = W / s, fh = H / s;
  const g = sc;

  g.fillStyle = "#E4E8EE";
  g.fillRect(0, 0, W, H);
  g.drawImage(image, fx, fy, fw, fh);
  g.fillStyle = "#4A5563";
  g.font = "700 26px system-ui, sans-serif";
  g.textAlign = "center";
  g.fillText("off-frame cells", W / 2, H - 22);

  // softmax heat, square-rooted so the runners-up show next to a sharp winner
  let max = 0;
  for (const p of probs) max = Math.max(max, p);
  for (let i = 0; i < n; i++) {
    const a = Math.sqrt(probs[i] / max) * 0.85;
    if (a < 0.01) continue;
    g.fillStyle = `rgba(${HEAT},${a})`;
    g.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw, ch);
  }

  // cell grid, and the frame's edge
  g.strokeStyle = "rgba(0,0,0,0.5)";
  g.lineWidth = 1.5;
  g.beginPath();
  for (let c = 0; c <= cols; c++) { g.moveTo(c * cw, 0); g.lineTo(c * cw, H); }
  for (let r = 0; r <= rows; r++) { g.moveTo(0, r * ch); g.lineTo(W, r * ch); }
  g.stroke();
  g.strokeStyle = "#000";
  g.lineWidth = 4;
  g.strokeRect(fx, fy, fw, fh);

  // backbone patches, which only cover the frame
  if (patches) {
    const tr = data.token_rows, tc = data.token_cols;
    g.strokeStyle = WIN;
    g.lineWidth = 3;
    g.beginPath();
    for (let c = 0; c <= tc; c++) { g.moveTo(fx + c * fw / tc, fy); g.lineTo(fx + c * fw / tc, fy + fh); }
    for (let r = 0; r <= tr; r++) { g.moveTo(fx, fy + r * fh / tr); g.lineTo(fx + fw, fy + r * fh / tr); }
    g.stroke();
  }

  ctx.drawImage(scene, 0, 0);

  const R = data.radius, span = 2 * R + 1;
  const c = winner % cols, r = Math.floor(winner / cols);
  const [mx, my] = data.centroid;          // cells, centres at i + 0.5
  const x = mx * cw, y = my * ch;

  // the window the centre of mass is taken over, then the winning cell inside it
  function frame(g, x0, y0, w, h, colour, width) {
    g.strokeStyle = "#000"; g.lineWidth = width + 4; g.strokeRect(x0, y0, w, h);
    g.strokeStyle = colour; g.lineWidth = width; g.strokeRect(x0, y0, w, h);
  }
  ctx.setLineDash([14, 8]);
  frame(ctx, (c - R) * cw, (r - R) * ch, span * cw, span * ch, "#fff", 3);
  ctx.setLineDash([]);
  frame(ctx, c * cw, r * ch, cw, ch, WIN, 4);

  // the answer: centre of mass, with the window-averaged grasp axis through it
  function answer(g, px, py, len, dot) {
    const dx = Math.cos(data.axis_rad) * len, dy = Math.sin(data.axis_rad) * len;
    g.beginPath(); g.moveTo(px - dx, py - dy); g.lineTo(px + dx, py + dy);
    g.strokeStyle = "#000"; g.lineWidth = 9; g.stroke();
    g.strokeStyle = WIN; g.lineWidth = 4; g.stroke();
    g.beginPath(); g.arc(px, py, dot, 0, Math.PI * 2);
    g.fillStyle = DOT; g.fill(); g.lineWidth = 3; g.strokeStyle = "#000"; g.stroke();
  }
  answer(ctx, x, y, 60, 10);

  // magnified inset of the window, always bottom left
  const size = 400, k = size / (span * cw), kh = size / (span * ch);
  const ix = 24, iy = H - size - 70;
  ctx.fillStyle = "#E4E8EE";
  ctx.fillRect(ix, iy, size, size);
  ctx.drawImage(scene, (c - R) * cw, (r - R) * ch, span * cw, span * ch, ix, iy, size, size);
  ctx.textAlign = "center";
  ctx.font = "700 20px system-ui, sans-serif";
  for (let j = 0; j < span; j++) {
    for (let i = 0; i < span; i++) {
      const w = data.weights[j * span + i];
      const gx = c - R + i, gy = r - R + j;
      if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) continue;
      const tx = ix + (i + 0.5) * cw * k, ty = iy + (j + 0.5) * ch * kh + 7;
      const label = w >= 0.995 ? "100" : (w * 100).toFixed(w < 0.1 ? 1 : 0);
      ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.strokeText(label, tx, ty);
      ctx.fillStyle = "#fff"; ctx.fillText(label, tx, ty);
    }
  }
  frame(ctx, ix + R * cw * k, iy + R * ch * kh, cw * k, ch * kh, WIN, 4);
  answer(ctx, ix + (mx - (c - R)) * cw * k, iy + (my - (r - R)) * ch * kh, 90, 14);
  ctx.lineWidth = 6; ctx.strokeStyle = "#000";
  ctx.strokeRect(ix, iy, size, size);
  ctx.fillStyle = "#000";
  ctx.fillRect(ix - 3, iy + size, size + 6, 46);
  ctx.fillStyle = "#fff";
  ctx.font = "700 22px system-ui, sans-serif";
  const deg = data.axis_rad * 180 / Math.PI;
  ctx.fillText(`p = ${probs[winner].toFixed(2)}  ·  window % shown  ·  axis ${deg.toFixed(0)}° κ ${data.kappa.toFixed(1)}`,
               ix + size / 2, iy + size + 32);
}

ctx.fillStyle = "#fff";
ctx.fillRect(0, 0, W, H);
ctx.fillStyle = "#000";
ctx.font = "700 40px system-ui, sans-serif";
ctx.textAlign = "center";
ctx.fillText("waiting for the first prediction…", W / 2, H / 2);
poll();
</script>
</body>
</html>
"""


def make_handler(results: Latest, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = page
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif url.path == "/latest":
                after = int(parse_qs(url.query).get("after", ["0"])[0])
                seq, body = results.wait_newer(after, timeout=5.0)
                if body is None or seq <= after:
                    self.send_response(204)
                    self.send_header("X-Seq", str(seq))
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Seq", str(seq))
            else:
                self.send_error(404)
                return
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="http://127.0.0.1:4246/stream.mjpeg")
    parser.add_argument("--local_models", action="store_true",
                        help="Use the local model from models/ rather than downloading the "
                             "production model from huggingface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4250)
    parser.add_argument("--device", default=None)
    # The mjpeg feed carries no telemetry, so the state the net is conditioned on is fixed.
    parser.add_argument("--range", type=float, default=0.5, help="laser rangefinder reading, metres")
    parser.add_argument("--finger_angle", type=float, default=0.0, help="degrees")
    parser.add_argument("--target_force", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    model, _ = load_model(device, local_models=args.local_models)
    logger.info(f"model on {device}: {model.grid[0]}x{model.grid[1]} cells, input {model.image_size}")

    state = {"laser_rangefinder": args.range, "finger_angle": args.finger_angle,
             "target_force": args.target_force}
    frames, results = Latest(), Latest()
    threading.Thread(target=read_stream, args=(args.stream, frames), daemon=True).start()
    threading.Thread(target=run_model, args=(model, device, state, frames, results), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(results, PAGE.encode()))
    server.daemon_threads = True
    logger.info(f"open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
