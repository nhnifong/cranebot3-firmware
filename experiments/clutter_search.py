#!/usr/bin/env python

"""Search the clutter record by text: "where did the purple marker go?"

    python experiments/clutter_search.py
    python experiments/clutter_search.py --root clutter_record --model openai/clip-vit-base-patch16 --port 4260

Pick and place records every item it puts down under clutter_record/<robot>/: a frame of
the item from its grasp, a frame of where it landed, and a line in record.jsonl with where
it was picked from and dropped. This embeds those frames with CLIP and serves a page at
http://127.0.0.1:4260/ that ranks the drops against a text query and shows each one's
pictures and where it went.

Each drop is scored against the query by two images, taking the better:

    item     the middle of the grasp frame, between the jaws, where the item is
    scene    the whole grasp frame, for an item that fills it or sits off center

The frame of where it landed is shown but not searched: drops go to the same few bins, so
those frames are much the same crowded scene for every item, and they won category queries
("a stuffed animal") for whatever happened to be in the bin. --search_landed puts it back.

The grasp frames are full of other clutter, and raw CLIP similarity rises with how busy an
image is, whatever the query (see siglip2_probe.py). So each image's similarity is taken
relative to its mean similarity against a few prompts that name nothing in particular.

Embeddings are cached beside each record.jsonl, per model, and new drops are picked up on
the next search.
"""

import argparse
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# The part of a grasp frame between the jaws, as fractions of width and height.
ITEM_CROP = (0.2, 0.0, 0.8, 1.0)
# Prompts that describe nothing in particular; an image's mean similarity to these is its
# baseline, which a query has to beat.
NEUTRAL_PROMPTS = ["a photo", "a photo of clutter on the floor", "a photo of a carpet",
                   "a photo of some things", "a blurry photo", "a photo taken from above"]


def features(out):
    """The projected embedding from get_*_features: a tensor in older transformers, an output
    whose pooler_output holds it in newer ones."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class Index:
    """CLIP embeddings of every drop under root, kept up to date with the record files."""

    def __init__(self, root, model_id, device, search_landed=False):
        from transformers import CLIPModel, CLIPProcessor

        self.root = Path(root)
        self.views = ("item", "scene", "landed") if search_landed else ("item", "scene")
        self.model_id = model_id
        self.device = device
        logger.info(f"loading {model_id} on {device}")
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.lock = threading.Lock()
        self.entries = []          # dicts from record.jsonl, plus robot and embeddings
        self.mtimes = {}
        self.neutral = self.embed_text(NEUTRAL_PROMPTS)

    @torch.no_grad()
    def embed_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        v = features(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        v = features(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

    def cache_path(self, robot_dir):
        return robot_dir / f"clip_index_{self.model_id.replace('/', '__')}.npz"

    def refresh(self):
        """Re-read any record.jsonl that changed, embedding only drops not already cached."""
        with self.lock:
            changed = False
            for record in sorted(self.root.glob("*/record.jsonl")):
                mtime = record.stat().st_mtime
                if self.mtimes.get(record) == mtime:
                    continue
                self.mtimes[record] = mtime
                changed = True
            if not changed:
                return
            entries = []
            for record in sorted(self.root.glob("*/record.jsonl")):
                entries += self._load_robot(record.parent)
            self.entries = entries
            logger.info(f"{len(entries)} drops indexed")

    def _load_robot(self, robot_dir):
        rows = [json.loads(line) for line in open(robot_dir / "record.jsonl") if line.strip()]
        cache_file = self.cache_path(robot_dir)
        cache = dict(np.load(cache_file)) if cache_file.exists() else {}
        todo = {}
        for row in rows:
            for view, rel in (("item", row["image"]), ("scene", row["image"]),
                              ("landed", row.get("landed_image"))):
                key = f"{view}:{rel}"
                if rel and key not in cache and (robot_dir / rel).exists():
                    todo[key] = (view, robot_dir / rel)
        if todo:
            keys = list(todo)
            images = []
            for key in keys:
                view, path = todo[key]
                img = Image.open(path).convert("RGB")
                if view == "item":
                    w, h = img.size
                    x0, y0, x1, y1 = ITEM_CROP
                    img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
                images.append(img)
            for start in range(0, len(keys), 32):
                vecs = self.embed_images(images[start:start + 32])
                for key, v in zip(keys[start:start + 32], vecs):
                    cache[key] = v
            np.savez(cache_file, **cache)
            logger.info(f"embedded {len(keys)} images under {robot_dir}")
        out = []
        for row in rows:
            emb = {}
            for view, rel in (("item", row["image"]), ("scene", row["image"]),
                              ("landed", row.get("landed_image"))):
                if rel and f"{view}:{rel}" in cache:
                    emb[view] = cache[f"{view}:{rel}"]
            out.append({**row, "robot": robot_dir.name, "emb": emb})
        return out

    def search(self, query, limit):
        self.refresh()
        q = self.embed_text([query])[0]
        results = []
        with self.lock:
            for e in self.entries:
                scores = {}
                for view, v in e["emb"].items():
                    if view not in self.views:
                        continue
                    # how much better the query fits this image than nothing in particular does
                    scores[view] = float(v @ q - (self.neutral @ v).mean())
                if not scores:
                    continue
                best = max(scores, key=scores.get)
                results.append({k: v for k, v in e.items() if k != "emb"} |
                               {"score": scores[best], "matched": best, "scores": scores})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit], len(self.entries)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clutter search</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#2a2f3a; --text:#e7eaf0; --muted:#8e97a8; --accent:#ffa040; }
  html, body { margin:0; background:var(--bg); color:var(--text); font:15px system-ui, sans-serif; }
  main { max-width:1200px; margin:0 auto; padding:24px 16px 48px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:18px; }
  form { display:flex; gap:8px; margin-bottom:20px; }
  input { flex:1; padding:12px 14px; font-size:17px; border-radius:8px; border:1px solid var(--line);
          background:#0b0d11; color:var(--text); }
  button { padding:0 20px; font-size:16px; border-radius:8px; border:0; background:var(--accent); color:#111; font-weight:650; cursor:pointer; }
  .card { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr) 260px; gap:14px; background:var(--card);
          border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:14px; }
  .card.top { border-color:var(--accent); }
  figure { margin:0; }
  figure img { width:100%; border-radius:6px; display:block; background:#000; }
  figcaption { color:var(--muted); font-size:13px; margin-top:6px; }
  .matched figcaption { color:var(--accent); }
  .info h2 { margin:0 0 6px; font-size:17px; }
  .info .row { margin:4px 0; }
  .info .k { color:var(--muted); font-size:13px; }
  .score { font-variant-numeric:tabular-nums; color:var(--muted); font-size:13px; margin-top:10px; }
  svg { width:100%; max-width:200px; height:auto; margin-top:10px; background:#0b0d11; border-radius:6px; display:block; }
  .empty { color:var(--muted); }
  @media (max-width: 760px) { .card { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
  <h1>Clutter search</h1>
  <div class="sub" id="sub">Search everything pick and place has put away, by what it looks like.</div>
  <form id="f"><input id="q" placeholder="e.g. purple marker, green plush, a sock" autofocus><button>Search</button></form>
  <div id="results"></div>
</main>
<script>
const fmt = v => v.map(c => c.toFixed(2)).join(", ");
function when(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { weekday:"short", month:"short", day:"numeric", hour:"numeric", minute:"2-digit" });
}
function map(r, bounds) {
  // top-down room view, x right and y up, the drop and where it came from
  // a square of the larger extent, so both axes share a scale
  const [x0, y0, x1, y1] = bounds, span = Math.max(x1 - x0, y1 - y0), W = 200, H = 200;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const px = p => [(p[0] - cx) / span * W + W / 2, H / 2 - (p[1] - cy) / span * H];
  const [ax, ay] = px(r.picked_from), [bx, by] = px(r.dropped_at);
  return `<svg viewBox="0 0 ${W} ${H}">
    <rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="#2a2f3a"/>
    <line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="#8e97a8" stroke-dasharray="4 3"/>
    <circle cx="${ax}" cy="${ay}" r="5" fill="none" stroke="#8e97a8" stroke-width="2"/>
    <circle cx="${bx}" cy="${by}" r="7" fill="#ffa040"/>
  </svg>`;
}
async function search(q) {
  const res = await (await fetch(`/search?q=${encodeURIComponent(q)}`)).json();
  document.getElementById("sub").textContent =
    `${res.total} drops indexed with ${res.model}. Ranked by how much better "${q}" fits each picture than a generic description does.`;
  const out = document.getElementById("results");
  if (!res.results.length) { out.innerHTML = `<p class="empty">Nothing recorded yet.</p>`; return; }
  out.innerHTML = res.results.map((r, i) => {
    const img = p => p ? `/img/${encodeURIComponent(r.robot)}/${p}` : "";
    const fig = (src, cap, views) => src ? `<figure class="${views.includes(r.matched) ? "matched" : ""}">
        <img src="${src}" loading="lazy"><figcaption>${cap}</figcaption></figure>` : `<figure><figcaption>no picture</figcaption></figure>`;
    const dest = r.destination ? r.destination.replace(/_/g, " ") : "unknown";
    return `<section class="card ${i === 0 ? "top" : ""}">
      ${fig(img(r.image), "picked up" + (r.matched !== "landed" ? ` · matched (${r.matched === "item" ? "between the jaws" : "whole frame"})` : ""), ["item", "scene"])}
      ${fig(img(r.landed_image), "where it landed" + (r.matched === "landed" ? " · matched" : ""), ["landed"])}
      <div class="info">
        <h2>${i === 0 ? "Best match" : "#" + (i + 1)}</h2>
        <div class="row"><span class="k">dropped at</span><br><b>${dest}</b> (${fmt(r.dropped_at)}) m</div>
        <div class="row"><span class="k">when</span><br>${when(r.time)}</div>
        <div class="row"><span class="k">picked up from</span><br>(${fmt(r.picked_from)}) m</div>
        <div class="row"><span class="k">robot</span><br>${r.robot}</div>
        ${map(r, res.bounds)}
        <div class="score">score ${r.score.toFixed(3)} · ${Object.entries(r.scores).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(" · ")}</div>
      </div>
    </section>`;
  }).join("");
}
document.getElementById("f").addEventListener("submit", e => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (q) { history.replaceState(null, "", `?q=${encodeURIComponent(q)}`); search(q); }
});
const initial = new URLSearchParams(location.search).get("q");
if (initial) { document.getElementById("q").value = initial; search(initial); }
</script>
</body>
</html>
"""


def make_handler(index, limit):
    page = PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif url.path == "/search":
                q = parse_qs(url.query).get("q", [""])[0].strip()
                results, total = index.search(q, limit) if q else ([], len(index.entries))
                points = [p for e in index.entries for p in (e["picked_from"], e["dropped_at"])]
                pts = np.array(points)[:, :2] if points else np.zeros((1, 2))
                lo, hi = pts.min(0) - 0.5, pts.max(0) + 0.5
                self.send(json.dumps({"results": results, "total": total, "model": index.model_id,
                                      "bounds": [*lo.tolist(), *hi.tolist()]}).encode(), "application/json")
            elif url.path.startswith("/img/"):
                # /img/<robot>/images/<file>, only from inside the record
                parts = unquote(url.path[len("/img/"):]).split("/")
                path = (index.root / Path(*parts)).resolve()
                if index.root.resolve() not in path.parents or not path.is_file():
                    self.send_error(404)
                    return
                self.send(path.read_bytes(), "image/jpeg")
            else:
                self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="clutter_record", help="the observer's clutter_record directory")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14",
                        help="a CLIP checkpoint; openai/clip-vit-base-patch16 is smaller and faster")
    parser.add_argument("--limit", type=int, default=10, help="results shown per search")
    parser.add_argument("--search_landed", action="store_true",
                        help="also match the query against the frame of where each item landed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4260)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    index = Index(args.root, args.model, device, args.search_landed)
    index.refresh()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, args.limit))
    server.daemon_threads = True
    logger.info(f"open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
