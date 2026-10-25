#!/usr/bin/env python

"""Live illustration of how the basket centering net finds the drop point, and what find_basket would do with it.

    python experiments/creator_tools/basket_viz.py
    python experiments/creator_tools/basket_viz.py --local_models --port 4256

Reads the gripper MJPEG feed and live state from the observer's telemetry, runs BasketNet at
up to 30 fps, and serves a page at http://127.0.0.1:4256/ that draws, over the frame the
model saw:

    - the softmax over the 18 x 32 canvas cells as orange heat, with the window the center
      of mass is taken over
    - the drop point: where the basket is, the center of mass
    - where that point would be if the basket were centered: straight below the jaws at the
      same depth, with the find_basket tolerance drawn around it as a ring on the basket's
      plane, and an arrow from the one to the other - the sideways move
    - the drop points of the last few frames, the ones one find_basket look would take the
      median of
    - press d to swap the softmax heat for the per-cell distance head, in meters

and a panel with the score, the move, the drop height change, and the verdict a find_basket
look would reach on the last few frames: no basket, frames disagree, centered, or the step
it would take.
"""

import argparse
import base64
import collections
import json
import logging
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import torch

from cell_softmax_viz import Latest, make_handler, read_stream
from scalar_heads_viz import LiveState, read_telemetry
from nf_robot.host.maneuvers.find_basket import (
    FRAMES_PER_LOOK, GAIN, LATERAL_TOL_M, MAX_STEP_M, MIN_GOOD_FRAMES, SCORE_MIN, SPREAD_MAX_M,
    VERTICAL_TOL_M,
)
from nf_robot.ml.basket.geometry import target_body
from nf_robot.ml.basket.model import load_model, return_body_of
from nf_robot.ml.gripper_grid import CANVAS_SCALE, decode_position
from nf_robot.ml.grid_head import CENTROID_RADIUS
from nf_robot.ml.image_input import input_batch
from nf_robot.ml.visual_servoing.dataset import state_vector
from nf_robot.ml.visual_servoing.geometry import CAMERA_POS_BODY, CAMERA_ROT_BODY, JAW_POS_BODY
from nf_robot.ml.visual_servoing.uv_methods import (
    gripper_camera_calibration, project_camera, unproject,
)

logger = logging.getLogger(__name__)

FPS = 30
RING_POINTS = 48


def body_to_uv(point_body, calibration):
    """A body-frame point as normalized (u, v) in the gripper camera, or None if behind it."""
    point_cam = CAMERA_ROT_BODY.inv().apply(np.asarray(point_body) - CAMERA_POS_BODY)
    projected = project_camera(point_cam, calibration)
    return None if projected is None else [round(projected[0], 4), round(projected[1], 4)]


def look_verdict(recent):
    """What find_basket._look would conclude from these frames, plus the step it would take."""
    good = [r["move"] for r in recent if r["score"] >= SCORE_MIN]
    if len(good) < MIN_GOOD_FRAMES:
        return {"verdict": "no basket", "detail": f"{len(good)} of {len(recent)} frames saw one"}
    moves = np.array(good)
    median = np.median(moves, axis=0)
    spread = float(np.median(np.linalg.norm(moves[:, :2] - median[:2], axis=1)))
    if spread > SPREAD_MAX_M:
        return {"verdict": "frames disagree", "detail": f"spread {spread * 100:.0f} cm",
                "spread": spread}
    lateral = float(np.linalg.norm(median[:2]))
    if lateral <= LATERAL_TOL_M and abs(median[2]) <= VERTICAL_TOL_M:
        return {"verdict": "centered", "detail": f"spread {spread * 100:.0f} cm", "spread": spread}
    step = median * GAIN
    norm = float(np.linalg.norm(step))
    if norm > MAX_STEP_M:
        step = step * MAX_STEP_M / norm
    return {"verdict": "move", "detail": f"spread {spread * 100:.0f} cm", "spread": spread,
            "step": [round(float(v), 3) for v in step]}


def run_model(model, device, live: LiveState, frames: Latest, results: Latest):
    """Run the net on the newest frame with the newest state, over and over, publishing the
    cells, the decoded drop point, and the find_basket view of it."""
    rows, cols = model.grid
    calibration = gripper_camera_calibration()
    recent = collections.deque(maxlen=FRAMES_PER_LOOK)
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
        with torch.no_grad():
            out = model(input_batch(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model.image_size, device),
                        state_t)
            logits = out["logits"][:1].float()
            probs = logits[0].flatten().softmax(0)
            winner = int(probs.argmax())
            uv, distance, weights, _ = decode_position(
                logits, out["log_distance"][:1].float(), model.grid,
                torch.tensor([winner], device=device))
        uv = uv[0].cpu().numpy().astype(np.float64)
        distance = float(distance[0])
        rise = float(out["rise"][0])
        score = float(probs[winner])
        move = return_body_of(uv, distance, rise)[0]
        latency = time.monotonic() - start
        slow = slow + 1 if latency > 1 / FPS else 0
        if slow == FPS:
            logger.warning(f"inference takes {latency * 1000:.0f} ms, too slow for {FPS} fps")

        # Where the drop point would sit in the image if the basket were centered: straight
        # below the jaws at the same depth, and the lateral tolerance as a ring on that plane.
        drop_body = target_body(unproject(uv[0], uv[1], distance, calibration))
        centered = np.array([JAW_POS_BODY[0], JAW_POS_BODY[1], drop_body[2]])
        ring = [body_to_uv(centered + LATERAL_TOL_M * np.array([np.cos(a), np.sin(a), 0.0]),
                           calibration)
                for a in np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)]

        recent.append({"uv": [round(float(v), 4) for v in uv], "score": score, "move": move})

        # Stretched to the model's input aspect, which is what it saw: input_batch never crops.
        shown = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        distances = out["log_distance"][0].float().exp().cpu().numpy().astype("<f4")
        results.put(json.dumps({
            "rows": rows, "cols": cols, "canvas_scale": CANVAS_SCALE, "radius": CENTROID_RADIUS,
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
            "probs": base64.b64encode(probs.cpu().numpy().astype("<f4").tobytes()).decode(),
            "distances": base64.b64encode(distances.tobytes()).decode(),
            "winner": winner,
            "weights": [round(float(v), 4) for v in weights[0]],
            "uv": [round(float(v), 4) for v in uv],
            "centered_uv": body_to_uv(centered, calibration),
            "ring": [p for p in ring if p is not None],
            "trail": [r["uv"] for r in recent],
            "trail_good": [r["score"] >= SCORE_MIN for r in recent],
            "score": score,
            "score_min": SCORE_MIN,
            "distance_m": round(distance, 3),
            "move": [round(float(v), 3) for v in move],
            "lateral_m": round(float(np.linalg.norm(move[:2])), 3),
            "lateral_tol": LATERAL_TOL_M, "vertical_tol": VERTICAL_TOL_M,
            "look": look_verdict(recent),
            "state": {k: round(v, 3) for k, v in state.items()},
            "state_age_s": round(age, 2),
            "latency_ms": round(latency * 1000),
        }).encode())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Basket centering</title>
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
const HEAT = "255,122,0", DEPTH = "0,140,255", DOT = "#FF2DAA", WIN = "#00D1FF", GOOD = "#2BD46B";
let showDepth = false;

document.addEventListener("keydown", e => {
  if (e.key === "d") { showDepth = !showDepth; }
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
        render(d, img, f32(d.probs), f32(d.distances));
      }
    } catch (e) {
      await new Promise(res => setTimeout(res, 500));
    }
  }
}

function outline(text, x, y, fill) {
  ctx.lineWidth = 5; ctx.strokeStyle = "#000"; ctx.strokeText(text, x, y);
  ctx.fillStyle = fill || "#fff"; ctx.fillText(text, x, y);
}

function render(data, image, probs, distances) {
  const { rows, cols, canvas_scale: s, winner } = data;
  const n = rows * cols, cw = W / cols, ch = H / rows;
  // The cells span -(s-1)/2 .. 1+(s-1)/2 of the frame; the frame sits in the middle.
  const half = (s - 1) / 2;
  const fx = half / s * W, fy = half / s * H, fw = W / s, fh = H / s;
  const px = uv => [fx + uv[0] * fw, fy + uv[1] * fh];

  ctx.fillStyle = "#E4E8EE";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(image, fx, fy, fw, fh);

  if (showDepth) {
    // per-cell distance head, nearer is stronger
    let lo = Infinity, hi = 0;
    for (const d of distances) { lo = Math.min(lo, d); hi = Math.max(hi, d); }
    ctx.font = "600 15px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let i = 0; i < n; i++) {
      const a = 0.75 * (1 - (distances[i] - lo) / Math.max(hi - lo, 1e-6));
      const c = i % cols, r = Math.floor(i / cols);
      ctx.fillStyle = `rgba(${DEPTH},${a})`;
      ctx.fillRect(c * cw, r * ch, cw, ch);
      if (r % 2 === 0 && c % 2 === 0) outline(distances[i].toFixed(2), (c + 0.5) * cw, (r + 0.5) * ch + 5);
    }
  } else {
    // softmax heat, square-rooted so the runners-up show next to a sharp winner
    let max = 0;
    for (const p of probs) max = Math.max(max, p);
    for (let i = 0; i < n; i++) {
      const a = Math.sqrt(probs[i] / max) * 0.85;
      if (a < 0.01) continue;
      ctx.fillStyle = `rgba(${HEAT},${a})`;
      ctx.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw, ch);
    }
  }

  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= cols; c++) { ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); }
  for (let r = 0; r <= rows; r++) { ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); }
  ctx.stroke();
  ctx.strokeStyle = "#000"; ctx.lineWidth = 4;
  ctx.strokeRect(fx, fy, fw, fh);

  // the window the center of mass is taken over
  const R = data.radius, c0 = winner % cols, r0 = Math.floor(winner / cols);
  ctx.setLineDash([14, 8]);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 7;
  ctx.strokeRect((c0 - R) * cw, (r0 - R) * ch, (2 * R + 1) * cw, (2 * R + 1) * ch);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
  ctx.strokeRect((c0 - R) * cw, (r0 - R) * ch, (2 * R + 1) * cw, (2 * R + 1) * ch);
  ctx.setLineDash([]);

  // the tolerance ring around where a centered basket would be
  const centered = data.centered_uv && px(data.centered_uv);
  const inside = data.lateral_m <= data.lateral_tol;
  if (data.ring.length) {
    ctx.beginPath();
    data.ring.forEach((p, i) => { const [x, y] = px(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
    ctx.strokeStyle = "#000"; ctx.lineWidth = 8; ctx.stroke();
    ctx.strokeStyle = inside ? GOOD : "#fff"; ctx.lineWidth = 4; ctx.stroke();
  }
  if (centered) {
    ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centered[0] - 18, centered[1]); ctx.lineTo(centered[0] + 18, centered[1]);
    ctx.moveTo(centered[0], centered[1] - 18); ctx.lineTo(centered[0], centered[1] + 18);
    ctx.stroke();
  }

  // the last few frames' drop points, the ones one look would take the median of
  data.trail.forEach((p, i) => {
    if (i === data.trail.length - 1) return;
    const [x, y] = px(p);
    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = data.trail_good[i] ? "rgba(255,45,170,0.6)" : "rgba(120,120,120,0.6)";
    ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#000"; ctx.stroke();
  });

  // arrow from the drop point to where it would be centered: the sideways move, as the image sees it
  const [dx, dy] = px(data.uv);
  if (centered && !inside) {
    const [tx, ty] = centered, ang = Math.atan2(ty - dy, tx - dx);
    for (const [color, width] of [["#000", 10], [WIN, 5]]) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(tx - 18 * Math.cos(ang), ty - 18 * Math.sin(ang)); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - 30 * Math.cos(ang - 0.4), ty - 30 * Math.sin(ang - 0.4));
      ctx.lineTo(tx - 30 * Math.cos(ang + 0.4), ty - 30 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
    }
  }
  ctx.beginPath(); ctx.arc(dx, dy, 13, 0, Math.PI * 2);
  ctx.fillStyle = data.score >= data.score_min ? DOT : "#888"; ctx.fill();
  ctx.lineWidth = 3; ctx.strokeStyle = "#000"; ctx.stroke();
  ctx.font = "700 22px system-ui, sans-serif"; ctx.textAlign = "left";
  outline(`${data.distance_m.toFixed(2)} m`, dx + 20, dy - 14);

  panel(data);
}

function panel(data) {
  const x = 16, y = 16, w = 440, h = 400, look = data.look;
  ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#000"; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h);
  let ty = y + 40;
  const line = (label, value, color, size) => {
    ctx.textAlign = "left"; ctx.fillStyle = "#4A5563";
    ctx.font = "600 20px system-ui, sans-serif"; ctx.fillText(label, x + 18, ty);
    ctx.textAlign = "right"; ctx.fillStyle = color || "#000";
    ctx.font = `700 ${size || 24}px system-ui, sans-serif`; ctx.fillText(value, x + w - 18, ty);
    ty += size ? size + 14 : 34;
  };
  const cm = v => `${(v * 100).toFixed(0)} cm`;
  const sees = data.score >= data.score_min;
  line("peak cell", `${(data.score * 100).toFixed(1)}%`, sees ? "#000" : "#D00");
  line("", sees ? "sees a basket" : `below ${(data.score_min * 100).toFixed(0)}%: no basket`,
       sees ? "#4A5563" : "#D00", 18);
  line("sideways", `${cm(data.lateral_m)}  (x ${cm(data.move[0])}, y ${cm(data.move[1])})`,
       data.lateral_m <= data.lateral_tol ? "#1A8F47" : "#000");
  line("up / down", (data.move[2] >= 0 ? "up " : "down ") + cm(Math.abs(data.move[2])),
       Math.abs(data.move[2]) <= data.vertical_tol ? "#1A8F47" : "#000");
  line("tolerance", `${cm(data.lateral_tol)} sideways, ${cm(data.vertical_tol)} vertical`, "#4A5563", 18);
  ty += 6;
  ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x + 18, ty - 22); ctx.lineTo(x + w - 18, ty - 22); ctx.stroke();
  const verdictColor = { "centered": "#1A8F47", "move": "#000" }[look.verdict] || "#D00";
  line(`look of ${data.trail.length}`, look.verdict, verdictColor, 30);
  line("", look.detail, "#4A5563", 18);
  if (look.step) {
    const s = look.step;
    line("step", `x ${cm(s[0])}, y ${cm(s[1])}, z ${cm(s[2])}`, "#000", 20);
  }
  const st = data.state;
  ctx.textAlign = "left"; ctx.font = "500 17px system-ui, sans-serif";
  ctx.fillStyle = data.state_age_s > 1 ? "#D00" : "#4A5563";
  ctx.fillText(`range ${st.laser_rangefinder.toFixed(2)} m · finger ${st.finger_angle.toFixed(0)}° · ` +
               `force ${st.target_force.toFixed(2)}`, x + 18, y + h - 40);
  ctx.fillText(`state ${data.state_age_s.toFixed(1)} s old · ${data.latency_ms} ms`, x + 18, y + h - 16);
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
    parser.add_argument("--port", type=int, default=4256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    model, _ = load_model(device, local_models=args.local_models)
    logger.info(f"model on {device}: {model.grid[0]}x{model.grid[1]} cells, input {model.image_size}")

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
