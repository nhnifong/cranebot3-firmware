import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import re
import json
import numpy as np
import argparse
import random
import uuid
import shutil
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, unquote
from importlib.resources import files

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_REPO_ID = "naavox/target-heatmap-dataset"
DEFAULT_MODEL_PATH = "models/target_heatmap.pth"
LOCAL_DATASET_ROOT = "target_heatmap_data"
HEATMAP_UNPROCESSED_DIR = "target_heatmap_data_unlabeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Network Input Resolution
# 960x544 is divisible by 32 (standard for CNNs), ensuring perfect alignment
# through pooling and upsampling layers without rounding errors.
HM_IMAGE_RES = (960, 544)

# How many frames before/after a labeled frame to also extract from the source
# lerobot video and tag with the same points (see extract_from_lerobot / label_mode).
LABEL_FRAME_OFFSET = 5
LABEL_SERVER_PORT = 8770
PREVIEW_SERVER_PORT = 8771

MINIMUM_CONFIDENCE = 0.95 # during eval

# ==========================================
# MODEL DEFINITION
# ==========================================

class TargetHeatmapNet(nn.Module):
    """
    Learns a heatmap from images that have one or more labeled points.
    Input images are expected to be 960x544.
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder
        # Since input is divisible by 8 (2^3), we can use standard fixed Upsample layers
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        # Dimensions align perfectly with 960x544 input
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))

# ==========================================
# DATASET & UTILS
# ==========================================

def generate_blob(x_grid, y_grid, cx, cy, sigma=15):
    """Generates a Gaussian blob at (cx, cy)."""
    return np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))

class DobbyDataset(Dataset):
    def __init__(self, root_dir):
        self.data_dir = os.path.join(root_dir, "train")
        self.metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Could not find metadata at {self.metadata_path}")
            
        self.samples = []
        with open(self.metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        h, w = img.shape[:2]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        x_grid = np.arange(0, w, 1, float)
        y_grid = np.arange(0, h, 1, float)[:, np.newaxis]
        
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        
        for pt in item.get("points", []):
            if isinstance(pt, (list, tuple)):
                cx, cy = pt[0], pt[1]
            elif isinstance(pt, dict):
                cx, cy = pt['x'], pt['y']
            else:
                continue
                
            combined_heatmap = np.maximum(combined_heatmap, generate_blob(x_grid, y_grid, cx, cy))
            
        return img_tensor, torch.from_numpy(combined_heatmap).float().unsqueeze(0)

def extract_targets_from_heatmap(heatmap: np.ndarray, top_n: int = 10, threshold: float = 0.5):
    """
    Extracts the centers of high-confidence blobs from a heatmap.
    Returns sorted list of (norm_x, norm_y, confidence).
    """
    mask = (heatmap > threshold).astype(np.uint8) * 255

    # RETR_EXTERNAL to ignore holes inside blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        roi = heatmap[y:y+h, x:x+w]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        
        global_x = x + max_loc[0]
        global_y = y + max_loc[1]
        
        candidates.append((global_x, global_y, max_val))

    candidates.sort(key=lambda k: k[2], reverse=True)

    height, width = heatmap.shape
    results = []
    
    for c in candidates[:top_n]:
        norm_x = c[0] / width
        norm_y = c[1] / height
        confidence = c[2]
        if confidence > MINIMUM_CONFIDENCE:
            results.append((norm_x, norm_y, confidence))

    return np.array(results)

# ==========================================
# TRAINING LOOP
# ==========================================

def train(args):
    from huggingface_hub import snapshot_download
    print(f"Downloading/Loading dataset from {args.dataset_id}...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    print(f"Dataset available at: {dataset_path}")

    dataset = DobbyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model = TargetHeatmapNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss() 

    print(f"Starting training on {len(dataset)} images for {args.epochs} epochs...")
    print(f"Device: {DEVICE}")

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for imgs, maps in dataloader:
            imgs, maps = imgs.to(DEVICE), maps.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, maps)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(dataloader):.5f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

# ==========================================
# PREVIEW TOOL (browser-based)
# ==========================================
# Qualitative check of a trained model against a live video source or a
# dataset's eval split. No opencv GUI (headless-only in this repo): frames are
# pushed to the browser over an MJPEG stream, same approach as MjpegStreamer in
# nf_robot.host.video_streamer.

def _send_bytes(handler, body, content_type, status=200):
    """Shared by PreviewHandler and LabelHandler's non-streaming responses."""
    handler.send_response(status)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

def _send_json(handler, obj, status=200):
    _send_bytes(handler, json.dumps(obj).encode('utf-8'), 'application/json', status)

def run_inference(model, img_bgr):
    """
    Helper to run model on a single BGR image and return overlay.
    """
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
    batch = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        heatmap_out = model(batch)
    heatmap_np = heatmap_out.squeeze().cpu().numpy()
    
    # img_bgr is used directly for background (no channel swap needed)
    img_display = img_bgr.copy()
    
    heatmap_vis = (heatmap_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img_display, 0.8, heatmap_color, 0.4, 0)

    targets = extract_targets_from_heatmap(heatmap_np)
    
    for x, y, confidence in targets:
        x = int(x * HM_IMAGE_RES[0])
        y = int(y * HM_IMAGE_RES[1])
        
        box_size = 20
        top_left =     (x - box_size, y - box_size)
        bottom_right = (x + box_size, y + box_size)
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)
        conf_text = f"{confidence:.2f}"
        cv2.putText(overlay, conf_text, (x - 10, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return overlay

class _MjpegBroadcaster:
    """Holds the latest JPEG-encoded frame and wakes any browser clients blocked
    in /stream.mjpeg when a new one arrives."""
    def __init__(self):
        self.latest_frame = None
        self.condition = threading.Condition()

    def push(self, bgr_frame):
        ok, buf = cv2.imencode('.jpg', bgr_frame)
        if not ok:
            return
        with self.condition:
            self.latest_frame = buf.tobytes()
            self.condition.notify_all()

class PreviewSession:
    """
    Drives the browser preview in one of two modes:
      - 'video': a background thread continuously reads args.uri, runs inference,
        and pushes overlays to the stream (optionally recording to args.record).
      - 'dataset': random samples from a dataset's eval split are pushed to the
        stream on demand, one per '/next' click.
    """
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.broadcaster = _MjpegBroadcaster()
        self.done = threading.Event()
        self.stop_source = threading.Event()
        self.mode = 'video' if args.uri else 'dataset'

        self.samples = []
        self.data_dir = None
        if self.mode == 'dataset':
            self._load_eval_samples()

    def _load_eval_samples(self):
        from huggingface_hub import snapshot_download
        print(f"Downloading dataset {self.args.dataset_id} for samples...")
        dataset_path = snapshot_download(repo_id=self.args.dataset_id, repo_type="dataset")
        self.data_dir = os.path.join(dataset_path, "eval")
        metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No eval metadata found at {metadata_path}. Did you run 'split' on the dataset?")
        with open(metadata_path, 'r') as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(self.samples)} evaluation samples.")

    def show_next_sample(self):
        if not self.samples:
            return None
        sample = random.choice(self.samples)
        img = cv2.imread(os.path.join(self.data_dir, sample["file_name"]))
        if img is None:
            return None
        self.broadcaster.push(run_inference(self.model, img))
        return sample["file_name"]

    def start_video_stream(self):
        source = int(self.args.uri) if self.args.uri.isdigit() else self.args.uri
        print(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0
        print(f"Stream FPS: {fps}")

        recorder = None
        if self.args.record:
            # We are writing raw BGR24 frames to stdin
            command = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{HM_IMAGE_RES[0]}x{HM_IMAGE_RES[1]}',
                '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
                self.args.record,
            ]
            print(f"Starting recording: {' '.join(command)}")
            try:
                recorder = subprocess.Popen(command, stdin=subprocess.PIPE)
            except FileNotFoundError:
                print("Error: ffmpeg not found. Please install ffmpeg to record.")

        def _reader():
            try:
                while not self.stop_source.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        print("End of stream.")
                        break

                    frame_resized = cv2.resize(frame, HM_IMAGE_RES)
                    overlay = run_inference(self.model, frame_resized)

                    if recorder:
                        try:
                            recorder.stdin.write(overlay.tobytes())
                        except BrokenPipeError:
                            print("FFmpeg recording stopped unexpectedly.")

                    self.broadcaster.push(overlay)
            finally:
                cap.release()
                if recorder:
                    recorder.stdin.close()
                    recorder.wait()
                    print(f"Saved recording to {self.args.record}")
                self.done.set()

        threading.Thread(target=_reader, daemon=True).start()

    def request_quit(self):
        self.stop_source.set()
        if self.mode == 'dataset':
            self.done.set()

_PREVIEW_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Target Heatmap Preview</title>
<style>
  body { background:#111; color:#eee; font-family: sans-serif; margin:0; padding:20px; }
  img { max-width: 90vw; border:2px solid #444; display:block; }
  .controls { margin-top: 14px; }
  button { font-size: 16px; padding: 8px 16px; margin-right: 8px; cursor:pointer; }
  #status { margin-top: 10px; font-size: 14px; color: #9c9; }
</style>
</head>
<body>
<h2>Target Heatmap Preview (__MODE__ mode)</h2>
<img id="stream" src="/stream.mjpeg" />
<div class="controls">
  __NEXT_BUTTON__
  <button id="quit">Quit</button>
</div>
<div id="status"></div>
<script>
const status = document.getElementById('status');

const nextBtn = document.getElementById('next');
if (nextBtn) {
  nextBtn.onclick = async () => {
    const res = await fetch('/next', {method: 'POST'});
    const data = await res.json();
    status.textContent = data.filename ? ('Showing: ' + data.filename) : 'No samples available';
  };
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    if (e.code === 'Space') { e.preventDefault(); nextBtn.click(); }
  });
}

document.getElementById('quit').onclick = async () => {
  await fetch('/quit', {method: 'POST'});
  document.body.innerHTML = '<h2>Preview stopped. You can close this tab.</h2>';
};
</script>
</body>
</html>
"""

class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # suppress internal logging

    def do_GET(self):
        session = self.server.session
        path = urlparse(self.path).path

        if path == '/':
            next_button = '<button id="next">Next Sample (Space)</button>' if session.mode == 'dataset' else ''
            page = _PREVIEW_PAGE.replace('__MODE__', session.mode).replace('__NEXT_BUTTON__', next_button)
            _send_bytes(self, page.encode('utf-8'), 'text/html')
        elif path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            broadcaster = session.broadcaster
            last_sent = None
            try:
                while True:
                    with broadcaster.condition:
                        # In dataset mode pushes are sparse (one per click), so a client
                        # connecting after a push already happened must get that frame
                        # immediately rather than blocking for the next one.
                        while broadcaster.latest_frame is last_sent:
                            broadcaster.condition.wait()
                        frame = broadcaster.latest_frame
                    last_sent = frame
                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception:
                pass # client disconnected
        else:
            self.send_error(404)

    def do_POST(self):
        session = self.server.session
        path = urlparse(self.path).path

        if path == '/next':
            if session.mode != 'dataset':
                self.send_error(400)
                return
            _send_json(self, {'filename': session.show_next_sample()})
        elif path == '/quit':
            _send_json(self, {'status': 'ok'})
            session.request_quit()
        else:
            self.send_error(404)

class PreviewHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def _resolve_model_path(args):
    """--model_repo_id takes precedence over --model_path, downloading the model
    from the hub for machines that don't have it locally (mirrors the
    hf_hub_download pattern observer.py uses for TARGETING_MODEL_REPOID)."""
    if args.model_repo_id:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from {args.model_repo_id}...")
        return hf_hub_download(repo_id=args.model_repo_id, filename="target_heatmap.pth")
    return args.model_path

def preview_mode(args):
    model_path = _resolve_model_path(args)
    print(f"Loading model from {model_path}...")
    model = TargetHeatmapNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    session = PreviewSession(model, args)
    if session.mode == 'dataset':
        session.show_next_sample()
    else:
        session.start_video_stream()

    server = PreviewHTTPServer((args.bind, args.port), PreviewHandler)
    server.session = session

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    display_host = 'localhost' if args.bind in ('0.0.0.0', '::', '') else args.bind
    print(f"Preview running at http://{display_host}:{args.port}/")
    print("Open that URL in a browser. Quit from the page when done.")

    session.done.wait()
    server.shutdown()
    server.server_close()

# ==========================================
# LEROBOT FRAME EXTRACTION
# ==========================================

ANCHOR_CAMERA_SUBSTR = "anchor_camera"

def extract_from_lerobot(args):
    """
    Pulls random frames from a lerobot dataset's anchor camera videos into
    HEATMAP_UNPROCESSED_DIR for labeling.

    For each sampled (episode, camera, frame) it also grabs the frames
    `args.offset` steps before and after it (clamped to the episode), so a single
    labeling pass can tag a triplet with the same points via the filename
    convention `{group}__off{N}.jpg` (see label_mode / _sibling_files below).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading lerobot dataset {args.source_dataset_id}...")
    dataset = LeRobotDataset(repo_id=args.source_dataset_id, root=args.root)

    anchor_keys = [k for k in dataset.meta.video_keys if ANCHOR_CAMERA_SUBSTR in k]
    if not anchor_keys:
        print(f"No anchor camera keys found in dataset (video_keys={dataset.meta.video_keys})")
        return
    print(f"Anchor camera keys: {anchor_keys}")

    os.makedirs(HEATMAP_UNPROCESSED_DIR, exist_ok=True)

    n_episodes = dataset.meta.total_episodes
    saved_groups = 0
    for i in range(args.count):
        ep_idx = random.randrange(n_episodes)
        ep = dataset.meta.episodes[ep_idx]
        ep_from = ep["dataset_from_index"]
        ep_to = ep["dataset_to_index"]
        if ep_to <= ep_from:
            continue

        center = random.randrange(ep_from, ep_to)
        cam_key = random.choice(anchor_keys)
        group_id = uuid.uuid4()

        for off in sorted({-args.offset, 0, args.offset}):
            idx = min(max(center + off, ep_from), ep_to - 1)
            item = dataset[idx]
            rgb = (item[cam_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fn = f"{group_id}__off{off}.jpg"
            cv2.imwrite(os.path.join(HEATMAP_UNPROCESSED_DIR, fn), bgr)

        saved_groups += 1
        print(f"[{i + 1}/{args.count}] episode {ep_idx}, cam {cam_key}, frame {center}")

    print(f"Saved {saved_groups} frame group(s) to {HEATMAP_UNPROCESSED_DIR}")

# ==========================================
# LABELING TOOL (browser-based)
# ==========================================
# No opencv GUI (headless-only in this repo): the labeler runs a local HTTP
# server and is driven from a browser tab instead of a cv2 window.

_OFFSET_RE = re.compile(r'^(?P<base>.+)__off(?P<off>-?\d+)\.jpg$')

def _sibling_files(unprocessed_dir, fn):
    """Other frames extracted alongside `fn` (see extract_from_lerobot) that should
    receive the same label points. Files with no __off marker (e.g. legacy live
    snapshots from observer.py) have no siblings."""
    m = _OFFSET_RE.match(fn)
    if not m:
        return [fn]
    prefix = f"{m.group('base')}__off"
    return sorted(f for f in os.listdir(unprocessed_dir) if f.startswith(prefix) and f.endswith('.jpg'))

def _list_candidates(unprocessed_dir):
    """Files the labeler should offer up: __off0 frames (one per group) plus any
    frames with no offset marker at all."""
    candidates = []
    for fn in os.listdir(unprocessed_dir):
        if not fn.endswith('.jpg'):
            continue
        m = _OFFSET_RE.match(fn)
        if m:
            if int(m.group('off')) == 0:
                candidates.append(fn)
        else:
            candidates.append(fn)
    return candidates

def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f} min"

class LabelSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = threading.Event()

        # Pace tracking for the "N remain, finishing in ~M at this rate" estimate.
        # Timed from the first save rather than session start, so time spent
        # reading instructions/loading the page isn't counted against the rate.
        self._first_save_time = None
        self._groups_saved = 0

        self.train_dir = os.path.join(LOCAL_DATASET_ROOT, "train")
        self.metadata_path = os.path.join(self.train_dir, "metadata.jsonl")
        os.makedirs(self.train_dir, exist_ok=True)

        readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write("---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train/metadata.jsonl\n---\n")

    def pick_next(self):
        if not os.path.exists(HEATMAP_UNPROCESSED_DIR):
            return None
        candidates = _list_candidates(HEATMAP_UNPROCESSED_DIR)
        return random.choice(candidates) if candidates else None

    def remaining_count(self):
        if not os.path.exists(HEATMAP_UNPROCESSED_DIR):
            return 0
        return len(_list_candidates(HEATMAP_UNPROCESSED_DIR))

    def image_path(self, fn):
        return os.path.join(HEATMAP_UNPROCESSED_DIR, os.path.basename(fn))

    def save(self, fn, points):
        """Applies `points` (in the clicked image's own pixel space) to fn and all
        of its offset siblings, resizing each to HM_IMAGE_RES and moving it into
        the training set. Returns (images_saved, groups_remaining, eta_seconds)."""
        safe = os.path.basename(fn)
        src_path = self.image_path(safe)
        if not os.path.exists(src_path):
            raise FileNotFoundError(safe)

        siblings = _sibling_files(HEATMAP_UNPROCESSED_DIR, safe)
        saved = 0
        with self.lock:
            for sib in siblings:
                sib_path = self.image_path(sib)
                img = cv2.imread(sib_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                scale_x = HM_IMAGE_RES[0] / w
                scale_y = HM_IMAGE_RES[1] / h
                scaled_points = [(int(px * scale_x), int(py * scale_y)) for px, py in points]

                resized = cv2.resize(img, HM_IMAGE_RES, interpolation=cv2.INTER_AREA)
                new_fn = f"{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(self.train_dir, new_fn), resized)

                with open(self.metadata_path, 'a') as f:
                    f.write(json.dumps({"file_name": new_fn, "points": scaled_points}) + "\n")

                os.remove(sib_path)
                saved += 1

            now = time.time()
            if self._first_save_time is None:
                self._first_save_time = now
            self._groups_saved += 1
            elapsed = now - self._first_save_time
            rate = self._groups_saved / elapsed if elapsed > 0 else None

        remaining = self.remaining_count()
        if remaining == 0:
            eta_seconds = 0.0
        elif rate:
            eta_seconds = remaining / rate
        else:
            eta_seconds = None
        return saved, remaining, eta_seconds

_LABEL_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Target Heatmap Labeler</title>
<style>
  body { background:#111; color:#eee; font-family: sans-serif; margin:0; padding:20px; }
  #stage { position:relative; display:inline-block; border:2px solid #444; line-height:0; }
  #stage img { display:block; max-width: 90vw; height:auto; cursor: crosshair; }
  #dots { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  .controls { margin-top: 14px; }
  button { font-size: 16px; padding: 8px 16px; margin-right: 8px; cursor:pointer; }
  #status { margin-top: 10px; font-size: 14px; color: #9c9; }
  #empty { font-size: 20px; margin-top: 40px; }
</style>
</head>
<body>
<h2>Target Heatmap Labeler</h2>
<div id="app">
  <div id="stage">
    <img id="img" src="" />
    <svg id="dots"></svg>
  </div>
  <div class="controls">
    <button id="undo">Undo Last Point</button>
    <button id="save">Save &amp; Next (Space)</button>
    <button id="skip">Skip (N)</button>
    <button id="quit">Quit</button>
  </div>
  <div id="status">Points: 0</div>
  <div id="stats"></div>
</div>
<div id="empty" style="display:none">No more files to label.</div>

<script>
let points = [];
let current = null;
const img = document.getElementById('img');
const dots = document.getElementById('dots');
const status = document.getElementById('status');
const stats = document.getElementById('stats');

function formatDuration(seconds) {
  if (seconds < 60) return Math.round(seconds) + 's';
  return Math.round(seconds / 60) + ' min';
}

function renderStats(remaining, etaSeconds) {
  if (remaining === undefined || remaining === null) { stats.textContent = ''; return; }
  let text = remaining + ' remain';
  if (etaSeconds) text += ', finishing in ~' + formatDuration(etaSeconds) + ' at this rate';
  stats.textContent = text;
}

function render() {
  dots.innerHTML = '';
  for (const [x, y] of points) {
    const cx = (x / img.naturalWidth) * 100;
    const cy = (y / img.naturalHeight) * 100;
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', cx + '%');
    c.setAttribute('cy', cy + '%');
    c.setAttribute('r', 6);
    c.setAttribute('fill', '#0f0');
    c.setAttribute('stroke', '#000');
    dots.appendChild(c);
  }
  status.textContent = 'Points: ' + points.length + (current ? (' | ' + current) : '');
}

img.addEventListener('click', (e) => {
  const rect = img.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) / rect.width * img.naturalWidth);
  const y = Math.round((e.clientY - rect.top) / rect.height * img.naturalHeight);
  points.push([x, y]);
  render();
});

document.getElementById('undo').onclick = () => { points.pop(); render(); };

function showFinished() {
  document.getElementById('app').style.display = 'none';
  const empty = document.getElementById('empty');
  empty.textContent = 'All frames labeled. Finalizing the dataset in the terminal...';
  empty.style.display = 'block';
  current = null;
}

async function loadNext() {
  points = [];
  const res = await fetch('/next');
  const data = await res.json();
  renderStats(data.remaining, null);
  if (!data.filename) {
    showFinished();
    return;
  }
  current = data.filename;
  img.src = '/image/' + encodeURIComponent(data.filename) + '?t=' + Date.now();
  render();
}

document.getElementById('skip').onclick = loadNext;

document.getElementById('save').onclick = async () => {
  if (!current) return;
  const res = await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: current, points: points})
  });
  const data = await res.json();
  renderStats(data.remaining, data.eta_seconds);
  if (data.remaining === 0) {
    // The server auto-shuts-down once nothing is left to label; don't race it with another /next.
    showFinished();
  } else {
    loadNext();
  }
};

document.getElementById('quit').onclick = async () => {
  await fetch('/quit', {method: 'POST'});
  document.body.innerHTML = '<h2>Labeler stopped. You can close this tab.</h2>';
};

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'BUTTON') return;
  if (e.code === 'Space') { e.preventDefault(); document.getElementById('save').click(); }
  else if (e.key === 'n') { document.getElementById('skip').click(); }
});

loadNext();
</script>
</body>
</html>
"""

class LabelHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # suppress internal logging

    def do_GET(self):
        session = self.server.session
        path = urlparse(self.path).path

        if path == '/':
            _send_bytes(self, _LABEL_PAGE.encode('utf-8'), 'text/html')
        elif path == '/next':
            _send_json(self, {'filename': session.pick_next(), 'remaining': session.remaining_count()})
        elif path.startswith('/image/'):
            fn = unquote(path[len('/image/'):])
            img_path = session.image_path(fn)
            if not os.path.exists(img_path):
                self.send_error(404)
                return
            with open(img_path, 'rb') as f:
                _send_bytes(self, f.read(), 'image/jpeg')
        else:
            self.send_error(404)

    def do_POST(self):
        session = self.server.session
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length)) if length else {}

        if path == '/save':
            try:
                saved, remaining, eta_seconds = session.save(payload['filename'], payload['points'])
                eta_text = f" (finishing in ~{_format_duration(eta_seconds)} at this rate)" if eta_seconds else ""
                print(f"Saved {saved} image(s) from group '{payload['filename']}'. {remaining} group(s) remaining{eta_text}.")
                _send_json(self, {'status': 'ok', 'saved': saved, 'remaining': remaining, 'eta_seconds': eta_seconds})
                if remaining == 0:
                    print("All frames labeled.")
                    session.done.set()
            except FileNotFoundError:
                _send_json(self, {'status': 'error', 'message': 'file not found'}, status=404)
        elif path == '/quit':
            _send_json(self, {'status': 'ok'})
            session.done.set()
        else:
            self.send_error(404)

class LabelHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def label_mode(args):
    session = LabelSession()
    server = LabelHTTPServer((args.bind, args.port), LabelHandler)
    server.session = session

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    display_host = 'localhost' if args.bind in ('0.0.0.0', '::', '') else args.bind
    print(f"Labeler running at http://{display_host}:{args.port}/")
    print("Open that URL in a browser. Click to mark points, Save & Next to store, Quit when done.")
    print("Files are loaded from:", HEATMAP_UNPROCESSED_DIR)
    print(f"Saved Target Res: {HM_IMAGE_RES}")

    session.done.wait()
    server.shutdown()
    server.server_close()

    upload_prompt(args)

def upload_prompt(args):
    from huggingface_hub import HfApi, create_repo
    if not os.path.exists(LOCAL_DATASET_ROOT): return
    
    print("\n" + "="*30)
    print(f"Data organized in '{LOCAL_DATASET_ROOT}'")
    confirm = input(f"Upload to {args.dataset_id}? (y/n): ").strip().lower()
    
    if confirm == 'y':
        api = HfApi()
        create_repo(args.dataset_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=LOCAL_DATASET_ROOT,
            repo_id=args.dataset_id,
            repo_type="dataset"
        )
        print("Uploaded successfully.")

def split_and_upload(args):
    print(f"Preparing to split dataset in {LOCAL_DATASET_ROOT}...")
    
    train_dir = os.path.join(LOCAL_DATASET_ROOT, "train")
    eval_dir = os.path.join(LOCAL_DATASET_ROOT, "eval")
    train_meta = os.path.join(train_dir, "metadata.jsonl")
    eval_meta = os.path.join(eval_dir, "metadata.jsonl")

    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    all_samples = []

    def load_samples(meta_path, source_split):
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry['_current_split'] = source_split
                        all_samples.append(entry)

    load_samples(train_meta, "train")
    load_samples(eval_meta, "eval")

    if not all_samples:
        print("No data found in local folders.")
        return

    print(f"Found {len(all_samples)} total samples. Shuffling and splitting 90/10...")
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train_set = all_samples[:split_idx]
    eval_set = all_samples[split_idx:]
    
    print(f"New distribution -> Train: {len(train_set)} | Eval: {len(eval_set)}")

    def process_split(sample_list, target_split, target_dir, target_meta_path):
        with open(target_meta_path, 'w') as f:
            for entry in sample_list:
                current_split = entry.pop('_current_split')
                fname = entry['file_name']
                
                if current_split != target_split:
                    src_path = os.path.join(LOCAL_DATASET_ROOT, current_split, fname)
                    dst_path = os.path.join(target_dir, fname)
                    
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)
                    else:
                        print(f"Warning: File missing at {src_path}")
                
                f.write(json.dumps(entry) + "\n")

    process_split(train_set, "train", train_dir, train_meta)
    process_split(eval_set, "eval", eval_dir, eval_meta)

    readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
    with open(readme_path, "w") as f:
        f.write("---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train/metadata.jsonl\n  - split: test\n    path: eval/metadata.jsonl\n---\n")

    upload_prompt(args)

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target Heatmap ML Tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    train_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--epochs", type=int, default=250)
    train_parser.add_argument("--batch_size", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    # Preview Command (qualitative check: live video or a dataset's eval split, in a browser)
    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    preview_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    preview_parser.add_argument("--model_repo_id", type=str, default=None, help="Download the model from this HF hub repo (filename target_heatmap.pth) instead of using --model_path")
    preview_parser.add_argument("--uri", type=str, default=None, help="Video file path or camera index")
    preview_parser.add_argument("--record", type=str, default=None, help="Path to save MP4 recording (only works with --uri)")
    preview_parser.add_argument("--port", type=int, default=PREVIEW_SERVER_PORT)
    preview_parser.add_argument("--bind", type=str, default="127.0.0.1", help="Address to bind the preview tool's local web server to")

    # Extract Command (pull labeling candidates from a lerobot dataset's anchor cameras)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source_dataset_id", type=str, required=True, help="repo_id of the lerobot dataset to pull anchor camera frames from")
    extract_parser.add_argument("--root", type=str, default=None, help="Local root of the lerobot dataset (defaults to the HF cache, downloading if needed)")
    extract_parser.add_argument("--count", type=int, default=30, help="Number of frame groups (center + offsets) to extract")
    extract_parser.add_argument("--offset", type=int, default=LABEL_FRAME_OFFSET, help="Frame offset (each direction) to also extract for dataset variety")

    # Label Command
    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    label_parser.add_argument("--port", type=int, default=LABEL_SERVER_PORT)
    label_parser.add_argument("--bind", type=str, default="127.0.0.1", help="Address to bind the labeler's local web server to")

    # Split Command
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "preview":
        preview_mode(args)
    elif args.command == "extract":
        extract_from_lerobot(args)
    elif args.command == "label":
        label_mode(args)
    elif args.command == "split":
        split_and_upload(args)