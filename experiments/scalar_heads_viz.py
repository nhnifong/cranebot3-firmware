#!/usr/bin/env python

"""Live readout of the visual servo net's scalar heads, and what each one is made of.

    python experiments/scalar_heads_viz.py
    python experiments/scalar_heads_viz.py --telemetry ws://127.0.0.1:4245 --port 4254

The companion of cell_softmax_viz.py. Reads the gripper MJPEG feed, runs VisualServoNet on it
at up to 30 fps, and serves a page at http://127.0.0.1:4254/ showing the frame with one
panel per corner:

    close      should the close have begun; reads the cell grid pooled into 4 x 6 bins
    present    is any graspable target in view
    holding    are we holding something now
    pressure   the grip pressure the object will need

and finger speed along the bottom. present, holding, pressure and finger all read the global
[CLS]/register vector beside the state; close reads its pooled bins beside the state.

Each panel breaks its head's pre-activation into contributions, as gradient x input against
a zero baseline: the image, and each of the three state inputs. Gradient x input is a
first-order split, so the parts need not add up to the logit; the rest is shown as the
remainder (biases and nonlinearity). Press c to toggle the close head's image contribution
drawn over the frame, bin by bin.

The state the net is conditioned on (rangefinder, finger angle, target force) comes live
from the observer's local telemetry websocket, the same GripperSensors readings the servo
feeds it. Frames are skipped until the first reading arrives, and the page shows how old the
reading is.
"""

import argparse
import base64
import json
import logging
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import torch
import torch.nn.functional as F

from websockets.sync.client import connect as websocket_connect

from cell_softmax_viz import Latest, make_handler, read_stream
from nf_robot.generated.nf import telemetry
from nf_robot.ml.grid_head import adaptive_avg_pool2d
from nf_robot.ml.gripper_grid import CANVAS_SCALE
from nf_robot.ml.visual_servoing.dataset import STATE_KEYS, state_vector
from nf_robot.ml.visual_servoing.model import CLOSE_POOL
from nf_robot.ml.visual_servoing.servo import load_model, prepare_frame

logger = logging.getLogger(__name__)

FPS = 30


class LiveState:
    """The newest gripper sensor readings from the observer's telemetry, as the net's state dict."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {}
        self.stamp = None

    def update(self, sensors):
        fields = {"laser_rangefinder": sensors.range, "finger_angle": sensors.angle,
                  "target_force": sensors.target_force}
        with self.lock:
            self.state.update({k: float(v) for k, v in fields.items() if v is not None})
            self.stamp = time.monotonic()

    def get(self):
        """(state, age in seconds), or (None, None) until every input has been heard."""
        with self.lock:
            if len(self.state) < len(STATE_KEYS):
                return None, None
            return dict(self.state), time.monotonic() - self.stamp


def read_telemetry(url, live: LiveState):
    """Keep `live` holding the newest gripper sensors, reconnecting whenever the socket drops.
    Only listens; it sends the observer nothing."""
    while True:
        try:
            with websocket_connect(url) as websocket:
                logger.info(f"telemetry from {url}")
                for message in websocket:
                    for item in telemetry.TelemetryBatchUpdate().parse(message).updates:
                        if item.grip_sensors is not None:
                            live.update(item.grip_sensors)
        except Exception as e:
            logger.warning(f"telemetry {url}: {e}; retrying")
        time.sleep(1.0)


def contributions(value, inputs):
    """Gradient x input of a scalar for each tensor in `inputs`."""
    grads = torch.autograd.grad(value, inputs, retain_graph=True)
    return [(g * x).detach() for g, x in zip(grads, inputs)]


def explain(model, images, state_t):
    """Every scalar head's output, pre-activation, and its split into image and state parts."""
    with torch.no_grad():
        x, global_vec = model.cell_map(images, state_t)
    global_vec = global_vec.float().detach().requires_grad_(True)
    state = state_t.float().detach().requires_grad_(True)
    x = x.float()

    flags = model.global_head(torch.cat([global_vec, state], dim=-1))[0]
    with torch.no_grad():
        bins = adaptive_avg_pool2d(F.gelu(model.close_reduce(x)), CLOSE_POOL)
    bins = bins.detach().requires_grad_(True)
    close_logit = model.close_head(
        torch.cat([bins.flatten(1), state[:, model.close_state_index]], dim=-1))[0, 0]

    def head(pre, out, image_input):
        image, st = contributions(pre, [image_input, state])
        parts = {"image": float(image.sum())}
        # a state input the head never reads has a zero gradient, and is left off its panel
        keys = model.close_state_keys if image_input is bins else STATE_KEYS
        parts.update({k: float(v) for k, v in zip(STATE_KEYS, st[0]) if k in keys})
        parts["remainder"] = float(pre) - sum(parts.values())
        return {"value": float(out), "pre": float(pre), "parts": parts}, image

    heads = {}
    heads["close"], close_image = head(close_logit, close_logit.sigmoid(), bins)
    heads["present"], _ = head(flags[1], flags[1].sigmoid(), global_vec)
    heads["holding"], _ = head(flags[2], flags[2].sigmoid(), global_vec)
    heads["pressure"], _ = head(flags[3], F.softplus(flags[3]), global_vec)
    heads["finger"], _ = head(flags[0], torch.tanh(flags[0]), global_vec)
    # (rows, cols) of each bin's summed contribution to the close logit
    close_bins = close_image[0].sum(0).cpu().tolist()
    return heads, close_bins


def run_model(model, device, live: LiveState, frames: Latest, results: Latest):
    """Run the net on the newest frame with the newest state, over and over, publishing every
    scalar head."""
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
        state, age = live.get()
        if state is None:
            continue
        start = time.monotonic()
        state_t = torch.from_numpy(state_vector(state))[None].to(device)
        heads, close_bins = explain(model, prepare_frame(bgr, model.image_size, device), state_t)
        latency = time.monotonic() - start
        slow = slow + 1 if latency > 1 / FPS else 0
        if slow == FPS:
            logger.warning(f"inference takes {latency * 1000:.0f} ms, too slow for {FPS} fps")

        # Stretched to the model's input aspect, which is what it saw: prepare_frame never crops.
        shown = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        results.put(json.dumps({
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
            "heads": heads,
            "close_bins": close_bins,
            "canvas_scale": CANVAS_SCALE,
            "state": {k: round(v, 3) for k, v in state.items()},
            "state_age_s": round(age, 2),
            "latency_ms": round(latency * 1000),
        }).encode())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scalar heads</title>
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
const ctx = document.getElementById("c").getContext("2d");
const POS = "255,122,0", NEG = "0,140,255";
let showBins = true;

document.addEventListener("keydown", e => {
  if (e.key === "c") { showBins = !showBins; }
});

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
        render(d, img);
      }
    } catch (e) {
      await new Promise(res => setTimeout(res, 500));
    }
  }
}

const LABELS = {
  image: "image", laser_rangefinder: "rangefinder", finger_angle: "finger angle",
  target_force: "target force", remainder: "remainder",
};
const UNITS = { laser_rangefinder: " m", finger_angle: "°", target_force: "" };

function panel(x, y, title, head, fmt, state) {
  const w = 520, h = 330;
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);

  ctx.textAlign = "left";
  ctx.fillStyle = "#000";
  ctx.font = "700 30px system-ui, sans-serif";
  ctx.fillText(title, x + 20, y + 44);
  ctx.textAlign = "right";
  ctx.font = "800 52px system-ui, sans-serif";
  ctx.fillText(fmt(head.value), x + w - 20, y + 58);
  ctx.font = "500 20px system-ui, sans-serif";
  ctx.fillStyle = "#4A5563";
  ctx.fillText(`pre-activation ${head.pre.toFixed(2)}`, x + w - 20, y + 86);

  // contributions, signed bars around a centre line
  const keys = Object.keys(head.parts);
  let max = 1;
  for (const k of keys) max = Math.max(max, Math.abs(head.parts[k]));
  const top = y + 112, rh = 42, mid = x + 330, half = 160;
  ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(mid, top - 6); ctx.lineTo(mid, top + keys.length * rh); ctx.stroke();
  keys.forEach((k, i) => {
    const v = head.parts[k], ry = top + i * rh;
    const len = v / max * half;
    ctx.fillStyle = `rgb(${v >= 0 ? POS : NEG})`;
    ctx.fillRect(len >= 0 ? mid : mid + len, ry + 6, Math.abs(len), rh - 14);
    ctx.textAlign = "left";
    ctx.fillStyle = k === "remainder" ? "#4A5563" : "#000";
    ctx.font = "600 20px system-ui, sans-serif";
    ctx.fillText(LABELS[k], x + 20, ry + 22);
    if (k in state) {
      ctx.font = "500 16px system-ui, sans-serif";
      ctx.fillStyle = "#4A5563";
      ctx.fillText(`= ${state[k]}${UNITS[k]}`, x + 20, ry + 40);
    }
    ctx.font = "600 18px system-ui, sans-serif";
    ctx.fillStyle = "#000";
    ctx.textAlign = v >= 0 ? "right" : "left";
    ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(2), v >= 0 ? mid - 8 : mid + 8, ry + 26);
  });
}

function render(data, image) {
  const s = data.canvas_scale;
  const half = (s - 1) / 2;
  const fx = half / s * W, fy = half / s * H, fw = W / s, fh = H / s;
  ctx.fillStyle = "#E4E8EE";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(image, fx, fy, fw, fh);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 4;
  ctx.strokeRect(fx, fy, fw, fh);

  // the close head's pooled bins span the whole canvas, edge cells included
  if (showBins) {
    const bins = data.close_bins, rows = bins.length, cols = bins[0].length;
    let max = 1e-6;
    for (const row of bins) for (const v of row) max = Math.max(max, Math.abs(v));
    const bw = W / cols, bh = H / rows;
    ctx.font = "700 22px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = bins[r][c];
        ctx.fillStyle = `rgba(${v >= 0 ? POS : NEG},${Math.abs(v) / max * 0.55})`;
        ctx.fillRect(c * bw, r * bh, bw, bh);
        ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 1.5;
        ctx.strokeRect(c * bw, r * bh, bw, bh);
        const label = (v >= 0 ? "+" : "") + v.toFixed(2);
        ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.strokeText(label, (c + 0.5) * bw, (r + 0.5) * bh);
        ctx.fillStyle = "#fff"; ctx.fillText(label, (c + 0.5) * bw, (r + 0.5) * bh);
      }
    }
  }

  const h = data.heads, st = data.state, m = 16;
  const pct = v => `${(v * 100).toFixed(0)}%`;
  panel(m, m, "close", h.close, pct, st);
  panel(W - 520 - m, m, "present", h.present, pct, st);
  panel(m, H - 330 - m, "holding", h.holding, pct, st);
  panel(W - 520 - m, H - 330 - m, "pressure", h.pressure, v => v.toFixed(2), st);

  // finger speed, -1 open .. +1 close, bottom centre
  const f = h.finger, bw = 700, bx = (W - bw) / 2, by = H - 110;
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillRect(bx - 20, by - 50, bw + 40, 140);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
  ctx.strokeRect(bx - 20, by - 50, bw + 40, 140);
  ctx.fillStyle = "#000"; ctx.textAlign = "left";
  ctx.font = "700 26px system-ui, sans-serif";
  ctx.fillText(`finger speed ${f.value.toFixed(2)}`, bx, by - 14);
  ctx.font = "500 18px system-ui, sans-serif"; ctx.fillStyle = "#4A5563";
  ctx.textAlign = "right";
  const fp = Object.entries(f.parts).filter(([k]) => k !== "remainder")
    .map(([k, v]) => `${LABELS[k]} ${(v >= 0 ? "+" : "") + v.toFixed(2)}`).join("  ");
  ctx.fillText(fp, bx + bw, by - 14);
  ctx.fillStyle = "#E4E8EE"; ctx.fillRect(bx, by, bw, 40);
  const len = f.value * bw / 2;
  ctx.fillStyle = `rgb(${f.value >= 0 ? POS : NEG})`;
  ctx.fillRect(len >= 0 ? bx + bw / 2 : bx + bw / 2 + len, by, Math.abs(len), 40);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.strokeRect(bx, by, bw, 40);
  ctx.beginPath(); ctx.moveTo(bx + bw / 2, by - 4); ctx.lineTo(bx + bw / 2, by + 44); ctx.stroke();
  ctx.font = "600 18px system-ui, sans-serif"; ctx.fillStyle = "#000";
  ctx.textAlign = "left"; ctx.fillText("open", bx, by + 66);
  ctx.textAlign = "right"; ctx.fillText("close", bx + bw, by + 66);
  ctx.textAlign = "center"; ctx.fillStyle = "#4A5563";
  ctx.fillStyle = data.state_age_s > 1 ? "#D00" : "#4A5563";
  ctx.fillText(`${data.latency_ms} ms · state ${data.state_age_s.toFixed(1)} s old`, bx + bw / 2, by + 66);
}

ctx.fillStyle = "#fff";
ctx.fillRect(0, 0, W, H);
ctx.fillStyle = "#000";
ctx.font = "700 40px system-ui, sans-serif";
ctx.textAlign = "center";
ctx.fillText("waiting for the first frame and telemetry…", W / 2, H / 2);
poll();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="http://127.0.0.1:4246/stream.mjpeg")
    parser.add_argument("--telemetry", default="ws://127.0.0.1:4245",
                        help="the observer's local telemetry websocket")
    parser.add_argument("--local_models", action="store_true",
                        help="Use the local model from models/ rather than downloading the "
                             "production model from huggingface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4254)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    model, _ = load_model(device, local_models=args.local_models)
    logger.info(f"model on {device}: input {model.image_size}")

    live, frames, results = LiveState(), Latest(), Latest()
    threading.Thread(target=read_telemetry, args=(args.telemetry, live), daemon=True).start()
    threading.Thread(target=read_stream, args=(args.stream, frames), daemon=True).start()
    threading.Thread(target=run_model, args=(model, device, live, frames, results), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(results, PAGE.encode()))
    server.daemon_threads = True
    logger.info(f"open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
