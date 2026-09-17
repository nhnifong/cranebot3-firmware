#!/usr/bin/env python

"""Live illustration of how the ortho target net finds targets: an offset and an objectness per cell.

    python experiments/ortho_cell_viz.py
    python experiments/ortho_cell_viz.py --local_models --port 4252

The ortho counterpart of cell_softmax_viz.py. Reads the floor projection MJPEG feed, runs
OrthoTargetNet on it at up to 30 fps (the feed itself is 10), and serves a page at
http://127.0.0.1:4252/ that draws, over the frame the model saw:

    - the 128 x 128 cell grid
    - every cell's own sigmoid objectness, as orange heat on an absolute scale: unlike the
      visual servo softmax, a cell's score does not depend on any other cell
    - every cell's offset answer, a small dot where that cell says the target would be
    - every peak above the checkpoint's threshold outlined, its answer drawn large - any
      number of them are kept, not one
    - magnified insets of the four strongest kept peaks, with the offset drawn from the
      cell's corner

Peaks are found the way ortho_target.decode finds them: local maxima after a 5x5 max
pool, the top 16, then the threshold.
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

from cell_softmax_viz import Latest, make_handler, read_stream
from nf_robot.ml.ortho_target import (DEFAULT_MODEL_PATH, TARGETING_MODEL_FILENAME,
                                      TARGETING_MODEL_REPOID, load_checkpoint, prepare_ortho_image)

logger = logging.getLogger(__name__)

FPS = 30
# The same as the observer's ORTHO_MAX_CANDIDATES and decode's default NMS radius.
MAX_PEAKS = 16
NMS_RADIUS = 2


def run_model(model, device, threshold, frames: Latest, results: Latest):
    """Run the net on the newest frame, over and over, publishing what every cell said."""
    grid = model.grid
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
        # The streamer encodes BGR; the model was trained on RGB.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            logits, offsets = model(prepare_ortho_image(rgb, model.image_size, device))
        probs = logits[0].float().sigmoid()                     # (grid, grid), each cell on its own
        offsets = offsets[0].float().sigmoid()                  # (2, grid, grid): x, y in the cell
        pooled = F.max_pool2d(probs[None, None], NMS_RADIUS * 2 + 1, stride=1, padding=NMS_RADIUS)[0, 0]
        peaks = torch.where(probs >= pooled, probs, torch.zeros_like(probs)).flatten()
        scores, index = peaks.topk(MAX_PEAKS)
        kept = [int(i) for s, i in zip(scores.tolist(), index.tolist()) if s >= threshold]
        latency = time.monotonic() - start
        slow = slow + 1 if latency > 1 / FPS else 0
        if slow == FPS:
            logger.warning(f"inference takes {latency * 1000:.0f} ms, too slow for {FPS} fps")

        # Squashed to square like prepare_ortho_image does, which is what the model saw.
        shown = cv2.resize(bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        results.put(json.dumps({
            "grid": grid,
            "tokens": model.token_grid,
            "threshold": threshold,
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
            "probs": base64.b64encode(probs.cpu().numpy().astype("<f4").tobytes()).decode(),
            "offsets": base64.b64encode(offsets.cpu().numpy().astype("<f4").tobytes()).decode(),
            "kept": kept,
            "latency_ms": round(latency * 1000),
        }).encode())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Every cell on its own</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden; }
  body { display: flex; align-items: center; justify-content: center; }
  canvas { width: min(100vw, calc(100vh * 16 / 9)); aspect-ratio: 16 / 9; display: block; }
</style>
</head>
<body>
<canvas id="c" width="1920" height="1080"></canvas>
<script>
const W = 1920, H = 1080, S = 1080;                     // the floor map is S x S on the left
const view = document.getElementById("c"), ctx = view.getContext("2d");
// Everything but the kept answers is drawn here first, so the insets magnify the cells
// rather than the markers on top of them.
const scene = document.createElement("canvas");
scene.width = S; scene.height = S;
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
        render(d, img, f32(d.probs), f32(d.offsets));
      }
    } catch (e) {
      await new Promise(res => setTimeout(res, 500));
    }
  }
}

function render(data, image, probs, offsets) {
  const { grid, kept, threshold } = data;
  const n = grid * grid, cs = S / grid;
  const g = sc;

  g.drawImage(image, 0, 0, S, S);

  // each cell's own probability, on an absolute scale
  for (let i = 0; i < n; i++) {
    const a = probs[i] * 0.85;
    if (a < 0.01) continue;
    g.fillStyle = `rgba(${HEAT},${a})`;
    g.fillRect((i % grid) * cs, Math.floor(i / grid) * cs, cs, cs);
  }

  g.strokeStyle = "rgba(0,0,0,0.28)";
  g.lineWidth = 1;
  g.beginPath();
  for (let k = 0; k <= grid; k++) {
    g.moveTo(k * cs, 0); g.lineTo(k * cs, S);
    g.moveTo(0, k * cs); g.lineTo(S, k * cs);
  }
  g.stroke();

  if (patches) {
    const t = data.tokens, ps = S / t;
    g.strokeStyle = WIN;
    g.lineWidth = 2.5;
    g.beginPath();
    for (let k = 0; k <= t; k++) {
      g.moveTo(k * ps, 0); g.lineTo(k * ps, S);
      g.moveTo(0, k * ps); g.lineTo(S, k * ps);
    }
    g.stroke();
  }

  // every cell's answer to "if a target is here, where in the cell is it"
  g.globalAlpha = 0.55;
  g.fillStyle = DOT;
  for (let i = 0; i < n; i++) {
    const x = ((i % grid) + offsets[i]) * cs, y = (Math.floor(i / grid) + offsets[n + i]) * cs;
    g.fillRect(x - 1.25, y - 1.25, 2.5, 2.5);
  }
  g.globalAlpha = 1;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(scene, 0, 0);

  // every answer that clears the threshold is kept
  const peak = i => {
    const c = i % grid, r = Math.floor(i / grid);
    return { c, r, x: (c + offsets[i]) * cs, y: (r + offsets[n + i]) * cs };
  };
  for (const i of kept) {
    const { c, r, x, y } = peak(i);
    ctx.strokeStyle = "#000"; ctx.lineWidth = 6;
    ctx.strokeRect(c * cs - 1, r * cs - 1, cs + 2, cs + 2);
    ctx.strokeStyle = WIN; ctx.lineWidth = 3;
    ctx.strokeRect(c * cs - 1, r * cs - 1, cs + 2, cs + 2);
    ctx.beginPath();
    ctx.moveTo(x - 30, y); ctx.lineTo(x + 30, y); ctx.moveTo(x, y - 30); ctx.lineTo(x, y + 30);
    ctx.strokeStyle = "#000"; ctx.lineWidth = 6; ctx.stroke();
    ctx.strokeStyle = WIN; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = DOT; ctx.fill();
    ctx.lineWidth = 2.5; ctx.strokeStyle = "#000"; ctx.stroke();
  }

  // right panel
  const px = S + 30, pw = W - S - 60;
  ctx.fillStyle = "#000";
  ctx.textAlign = "left";
  ctx.font = "800 44px system-ui, sans-serif";
  ctx.fillText(`${kept.length} kept`, px, 66);
  ctx.font = "700 26px system-ui, sans-serif";
  ctx.fillText(`cells with p ≥ ${threshold.toFixed(2)}, local peaks only`, px, 106);

  // insets of the strongest kept peaks
  const span = 7, size = 355, gap = 20, top = 140;
  for (let slot = 0; slot < 4; slot++) {
    const ix = px + (slot % 2) * (size + gap), iy = top + Math.floor(slot / 2) * (size + 46 + gap);
    ctx.fillStyle = "#E4E8EE";
    ctx.fillRect(ix, iy, size, size);
    ctx.lineWidth = 5; ctx.strokeStyle = "#000";
    if (slot >= kept.length) {
      ctx.strokeRect(ix, iy, size, size);
      ctx.fillStyle = "#6B7684";
      ctx.textAlign = "center";
      ctx.font = "700 26px system-ui, sans-serif";
      ctx.fillText("nothing else kept", ix + size / 2, iy + size / 2 + 9);
      ctx.textAlign = "left";
      continue;
    }
    const i = kept[slot], { c, r } = peak(i), k = size / (span * cs), h = (span - 1) / 2;
    ctx.drawImage(scene, (c - h) * cs, (r - h) * cs, span * cs, span * cs, ix, iy, size, size);
    const cx = ix + h * cs * k, cy = iy + h * cs * k;
    ctx.strokeStyle = "#000"; ctx.lineWidth = 9;
    ctx.strokeRect(cx, cy, cs * k, cs * k);
    ctx.strokeStyle = WIN; ctx.lineWidth = 5;
    ctx.strokeRect(cx, cy, cs * k, cs * k);
    const dx = cx + offsets[i] * cs * k, dy = cy + offsets[n + i] * cs * k;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(dx, dy);
    ctx.strokeStyle = "#000"; ctx.lineWidth = 8; ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.stroke();
    ctx.beginPath(); ctx.arc(dx, dy, 13, 0, Math.PI * 2);
    ctx.fillStyle = DOT; ctx.fill(); ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.stroke();
    ctx.lineWidth = 5; ctx.strokeStyle = "#000";
    ctx.strokeRect(ix, iy, size, size);
    ctx.fillStyle = "#000";
    ctx.fillRect(ix - 2.5, iy + size, size + 5, 46);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.font = "700 23px system-ui, sans-serif";
    ctx.fillText(`offset (${offsets[i].toFixed(2)}, ${offsets[n + i].toFixed(2)})  ·  p = ${probs[i].toFixed(2)}`,
                 ix + size / 2, iy + size + 31);
    ctx.textAlign = "left";
  }
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="http://127.0.0.1:8747/stream.mjpeg")
    parser.add_argument("--local_models", action="store_true",
                        help="Use the local model from models/ rather than downloading the "
                             "production model from huggingface")
    parser.add_argument("--threshold", type=float, default=None,
                        help="objectness to keep a peak at; default is the checkpoint's own")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4252)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    if args.local_models:
        path = DEFAULT_MODEL_PATH
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=TARGETING_MODEL_REPOID, filename=TARGETING_MODEL_FILENAME)
    logger.info(f"loading ortho target model from {path}")
    model, _ = load_checkpoint(path, device)
    threshold = model.threshold if args.threshold is None else args.threshold
    logger.info(f"model on {device}: {model.grid}x{model.grid} cells, input {model.image_size}, "
                f"threshold {threshold:.3f}")

    frames, results = Latest(), Latest()
    threading.Thread(target=read_stream, args=(args.stream, frames), daemon=True).start()
    threading.Thread(target=run_model, args=(model, device, threshold, frames, results), daemon=True).start()

    page = PAGE.encode()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(results, page))
    server.daemon_threads = True
    logger.info(f"open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
