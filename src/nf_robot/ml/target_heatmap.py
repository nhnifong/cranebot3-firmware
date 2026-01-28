import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
import argparse
import random
import uuid
import shutil
import subprocess
from importlib.resources import files

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_REPO_ID = "naavox/target-heatmap-dataset"
DEFAULT_MODEL_PATH = files("nf_robot.ml").joinpath("models/target_heatmap.pth")
LOCAL_DATASET_ROOT = "target_heatmap_data"
HEATMAP_UNPROCESSED_DIR = "target_heatmap_data_unlabeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Network Input Resolution
# 960x544 is divisible by 32 (standard for CNNs), ensuring perfect alignment 
# through pooling and upsampling layers without rounding errors.
HM_IMAGE_RES = (960, 544) 

# Labeling Source Resolution
SOURCE_RES = (1920, 1080)

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
# EVALUATION TOOL
# ==========================================

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

def eval_mode(args):
    from huggingface_hub import snapshot_download
    print(f"Loading model from {args.model_path}...")
    model = TargetHeatmapNet().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    window_name = "Target Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Mode 1: Video Stream
    if args.uri:
        source = args.uri
        # Convert to int if user passed a webcam index (e.g. "0")
        if source.isdigit():
            source = int(source)
            
        print(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 30.0
        print(f"Stream FPS: {fps}")

        cv2.resizeWindow(window_name, HM_IMAGE_RES[0], HM_IMAGE_RES[1])
        print("Controls: [Q] Quit")

        # FFMPEG Recording Setup
        recorder = None
        if args.record:
            # We are writing raw BGR24 frames to stdin
            command = [
                'ffmpeg',
                '-y',                    # Overwrite output
                '-f', 'rawvideo',        # Input format
                '-vcodec', 'rawvideo',
                '-s', f'{HM_IMAGE_RES[0]}x{HM_IMAGE_RES[1]}', # Size
                '-pix_fmt', 'bgr24',     # OpenCV uses BGR
                '-r', str(fps),          # Input framerate
                '-i', '-',               # Read from stdin
                '-c:v', 'libx264',       # Output codec
                '-pix_fmt', 'yuv420p',   # Pixel format for compatibility
                '-preset', 'fast',       # Encoding speed
                args.record              # Output filename
            ]
            print(f"Starting recording: {' '.join(command)}")
            try:
                recorder = subprocess.Popen(command, stdin=subprocess.PIPE)
            except FileNotFoundError:
                print("Error: ffmpeg not found. Please install ffmpeg to record.")
                return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break
            
            # Ensure frame matches network input size
            frame_resized = cv2.resize(frame, HM_IMAGE_RES)
            
            overlay = run_inference(model, frame_resized)
            
            # Write to ffmpeg stdin
            if recorder:
                try:
                    recorder.stdin.write(overlay.tobytes())
                except BrokenPipeError:
                    print("FFmpeg recording stopped unexpectedly.")
                    recorder = None

            cv2.imshow(window_name, overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if recorder:
            recorder.stdin.close()
            recorder.wait()
            print(f"Saved recording to {args.record}")
        
    # Mode 2: Dataset Evaluation
    else:
        print(f"Downloading dataset {args.dataset_id} for samples...")
        dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
        
        data_dir = os.path.join(dataset_path, "eval")
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        
        if not os.path.exists(metadata_path):
            print(f"No eval metadata found at {metadata_path}. Did you run 'split' on the dataset?")
            return

        samples = []
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
                
        print(f"Loaded {len(samples)} evaluation samples.")
        print("Controls: [SPACE] Next, [Q] Quit")
        
        cv2.resizeWindow(window_name, HM_IMAGE_RES[0] * 2, HM_IMAGE_RES[1] * 2)

        while True:
            sample = random.choice(samples)
            img_path = os.path.join(data_dir, sample["file_name"])
            img_input = cv2.imread(img_path)
            
            if img_input is None: 
                continue

            overlay = run_inference(model, img_input)
            
            cv2.imshow(window_name, overlay)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()

# ==========================================
# LABELING TOOL
# ==========================================

# State
current_clicks = []
current_image = None
current_image_path = None

def mouse_callback(event, x, y, flags, param):
    global current_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        current_clicks.append((x, y))
        print(f"Added Point (Source Res): {x}, {y}")
        draw_interface()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_clicks:
            removed = current_clicks.pop()
            print(f"Removed Point: {removed}")
            draw_interface()

def draw_interface():
    global current_image, current_clicks
    if current_image is None:
        return

    display = current_image.copy()

    for pt in current_clicks:
        cv2.circle(display, pt, 5, (0, 255, 0), -1)

    status_text = f"Points: {len(current_clicks)} | Res: {SOURCE_RES}"
    cv2.putText(display, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Labeler", display)

def label_mode(args):
    global current_clicks, current_image, current_image_path

    TRAIN_DIR = os.path.join(LOCAL_DATASET_ROOT, "train")
    METADATA_PATH = os.path.join(TRAIN_DIR, "metadata.jsonl")
    
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    
    readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train/metadata.jsonl\n---\n")

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeler", SOURCE_RES[0], SOURCE_RES[1])
    cv2.setMouseCallback("Labeler", mouse_callback)
    
    print("Labeler Started.")
    print("Files are loaded from:", HEATMAP_UNPROCESSED_DIR)
    print(f"Expected Input Res: {SOURCE_RES}")
    print(f"Saved Target Res:   {HM_IMAGE_RES}")
    print("\n--- Instructions ---")
    print("1. Left-Click:   Mark a spot.")
    print("2. Right-Click:  Undo last mark.")
    print("3. SPACE:        Save (resizes img & points) and Next.")
    print("4. 'n':          Skip to next random frame.")
    print("5. 'q':          Quit and ask to upload to Hugging Face.")

    while True:
        if not os.path.exists(HEATMAP_UNPROCESSED_DIR):
            print(f"No unprocessed directory: {HEATMAP_UNPROCESSED_DIR}")
            break
            
        files = [f for f in os.listdir(HEATMAP_UNPROCESSED_DIR) if f.endswith('.jpg')]
        if not files:
            print("No more files to process.")
            upload_prompt(args)
            break
            
        fn = random.choice(files)
        current_image_path = os.path.join(HEATMAP_UNPROCESSED_DIR, fn)
    
        current_image = cv2.imread(current_image_path)
        if current_image is None:
            continue
            
        # Ensure we are working with expected source resolution or warn/resize
        h, w = current_image.shape[:2]
        if (w, h) != SOURCE_RES:
            print(f"Warning: Image {fn} is {w}x{h}, expected {SOURCE_RES}. Visuals might be skewed.")

        current_clicks = [] 
        
        draw_interface()

        save_it = False
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                upload_prompt(args)
                return
                
            elif key == ord('n'):
                print("Skipped frame.")
                break 
                
            elif key == ord(' '):
                save_it = True
                break

        if save_it:
            # Transform points from SOURCE_RES to HM_IMAGE_RES
            scale_x = HM_IMAGE_RES[0] / SOURCE_RES[0]
            scale_y = HM_IMAGE_RES[1] / SOURCE_RES[1]
            
            scaled_points = []
            for px, py in current_clicks:
                scaled_points.append((int(px * scale_x), int(py * scale_y)))

            # Resize image
            resized_img = cv2.resize(current_image, HM_IMAGE_RES, interpolation=cv2.INTER_AREA)

            # Generate filename and save
            new_id = str(uuid.uuid4())
            new_fn = f"{new_id}.jpg"
            new_path = os.path.join(TRAIN_DIR, new_fn)
            
            cv2.imwrite(new_path, resized_img)
            
            # Remove original from unprocessed
            os.remove(current_image_path)
            
            # Write Metadata
            entry = {
                "file_name": new_fn,
                "points": scaled_points
            }
            with open(METADATA_PATH, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            
            print(f"Saved {len(scaled_points)} points -> {new_fn} (Resized to {HM_IMAGE_RES})")

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
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-3)

    # Eval Command
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    eval_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    eval_parser.add_argument("--uri", type=str, default=None, help="Video file path or camera index")
    eval_parser.add_argument("--record", type=str, default=None, help="Path to save MP4 recording (only works with --uri)")


    # Label Command
    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)

    # Split Command
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval_mode(args)
    elif args.command == "label":
        label_mode(args)
    elif args.command == "split":
        split_and_upload(args)