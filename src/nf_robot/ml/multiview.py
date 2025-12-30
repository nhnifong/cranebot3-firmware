import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import json
import glob
import cv2
import numpy as np
import shutil
import random
import uuid
import threading
import av
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
# Assuming lerobot is installed for the labeler's source data
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi, create_repo
from config_loader import load_config

DEFAULT_REPO_ID = "naavox/multiview-dataset"
DEFAULT_MODEL_PATH = "trainer/models/multiview.pth"
DATA_ROOT = "multi_view_data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
EVAL_DIR = os.path.join(DATA_ROOT, "eval")
LABELS_FILE = "labels.jsonl"

# Source for the labeler (only used during labeling)
LEROBOT_SOURCE_ID = "naavox/merged-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load actual calibration and pose data from current robot configuration
CAM_CALIBRATIONS = []
cfg = load_config()
for i in range(4):
    CAM_CALIBRATIONS.append({
        "matrix":np.array(cfg.camera_cal.intrinsic_matrix).reshape((3,3)),
        "dist":np.array(cfg.camera_cal.distortion_coeff),
        "rvec":np.array(config.anchors[i].pose.rotation),
        "tvec":np.array(config.anchors[i].pose.position),
    })

# this mapping can be used to map the camera numbers in the merged-5 dataset to the cameras described by CAM_CALIBRATIONS
# Note that this assumes we are training with the same robot naavox/merged-5 was collected with in the same room
# in naavox/merged-5 camera-0 was anchor-3 and camera-1 was anchor-2
merged_five_mapping = {0:3, 1:2}

# live camera addresses
CAM_CALIBRATIONS[0]['uri'] = 'tcp://192.168.1.151:8888'
CAM_CALIBRATIONS[1]['uri'] = 'tcp://192.168.1.157:8888'
CAM_CALIBRATIONS[2]['uri'] = 'tcp://192.168.1.153:8888'
CAM_CALIBRATIONS[3]['uri'] = 'tcp://192.168.1.154:8888'

class MultiViewDETR(nn.Module):
    def __init__(self, num_cameras=4, num_queries=10, hidden_dim=256, nheads=8, num_layers=4):
        super().__init__()
        
        # Backbone (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # Positional Embeddings (Learnable)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.cam_embed = nn.Embedding(num_cameras, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        # Object Queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Prediction Heads
        self.coord_head = nn.Linear(hidden_dim, 3)
        self.class_head = nn.Linear(hidden_dim, 1) 

    def forward(self, images):
        B, Num_Cams, C, H, W = images.shape
        imgs_flat = images.view(-1, C, H, W)
        
        # 1. Extract Features
        features = self.backbone(imgs_flat)
        features = self.input_proj(features)
        _, _, H_feat, W_feat = features.shape
        
        # 2. Prepare for Transformer
        features = features.flatten(2).permute(0, 2, 1) # (B*Num_Cams, Seq, 256)
        
        # Use reshape() instead of view()
        features = features.reshape(B, Num_Cams, -1, 256)
        
        # Add Camera Embeddings
        cam_ids = torch.arange(Num_Cams, device=images.device)
        cam_emb = self.cam_embed(cam_ids).view(1, Num_Cams, 1, 256)
        features = features + cam_emb
        
        # Add Spatial Positional Embeddings
        pos_y = self.row_embed[:H_feat].unsqueeze(1).repeat(1, W_feat, 1)
        pos_x = self.col_embed[:W_feat].unsqueeze(0).repeat(H_feat, 1, 1)
        pos = torch.cat([pos_x, pos_y], dim=2).flatten(0, 1).unsqueeze(0).unsqueeze(0)
        features = features + pos.to(features.device)

        # Flatten Cameras
        src = features.reshape(B, -1, 256)
        
        # 3. Run Transformer
        query_embed = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        hs = self.transformer(src, query_embed)
        
        # 4. Prediction
        coords = self.coord_head(hs)
        logits = self.class_head(hs)
        
        return torch.sigmoid(logits), coords

# --- TRAINING UTILITIES ---

class WorldToCameraProjector:
    def __init__(self, calibrations, device):
        self.cams = []
        for cal in calibrations:
            K = torch.from_numpy(cal["matrix"]).float().to(device)
            R, _ = cv2.Rodrigues(cal["rvec"])
            R = torch.from_numpy(R).float().to(device)
            t = torch.from_numpy(cal["tvec"]).float().to(device)
            self.cams.append({"K": K, "R": R, "t": t})

    def project(self, points_3d, cam_idx):
        """Projects 3D points to specific camera index."""
        cam = self.cams[cam_idx]
        p_cam = torch.matmul(points_3d, cam["R"].t()) + cam["t"].view(1, 1, 3)
        x = p_cam[..., 0] / (p_cam[..., 2] + 1e-6)
        y = p_cam[..., 1] / (p_cam[..., 2] + 1e-6)
        K = cam["K"]
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        return torch.stack([u, v], dim=-1)

class SockTriangulationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, LABELS_FILE)
        self.data = []
        
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {self.jsonl_path}")
        self.num_cameras = 4
        self.cam_keys = [f"camera_{i}" for i in range(self.num_cameras)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_entry = self.data[idx]
        
        # Load Images from disk
        images = []
        image_paths_map = label_entry.get("image_paths", {})
        
        for key in self.cam_keys:
            rel_path = image_paths_map.get(key)
            
            if rel_path:
                full_path = os.path.join(self.root_dir, rel_path)
                if os.path.exists(full_path):
                    # Load BGR
                    img = cv2.imread(full_path)
                    # BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Normalize 0-1 and CHW
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    images.append(img)
                else:
                    # File missing despite label? Pad with zeros.
                    images.append(torch.zeros(3, 360, 640))
            else:
                # Camera view not saved for this sample (e.g. dropped frame or not in source)
                images.append(torch.zeros(3, 360, 640))
                
        img_stack = torch.stack(images) # (4, 3, 360, 640)
        
        # Load Objects
        objects = label_entry.get("objects", [])
        
        return img_stack, objects

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch] # List of lists of objects
    return images, targets

def train(args):
    # Point to the self-contained train directory
    train_set = SockTriangulationDataset(TRAIN_DIR)
    
    if len(train_set) == 0:
        print(f"No data found in {TRAIN_DIR}. Run with --mode label first.")
        return

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Update model to use 4 cameras
    num_cameras = 4
    model = MultiViewDETR(num_cameras=num_cameras).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    projector = WorldToCameraProjector(CAM_CALIBRATIONS, DEVICE)
    
    print(f"Starting training for {args.epochs} epochs on {len(train_set)} samples...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        for batch_imgs, batch_targets in loader:
            batch_imgs = batch_imgs.to(DEVICE)
            pred_conf, pred_3d = model(batch_imgs)
            
            loss = 0
            
            for b in range(len(batch_imgs)):
                gt_objects = batch_targets[b]
                num_gt = len(gt_objects)
                
                if num_gt > 0:
                    cost_matrix = torch.zeros((num_gt, pred_3d.shape[1]), device=DEVICE)
                    
                    for i, obj_sightings in enumerate(gt_objects):
                        obj_loss = torch.zeros(pred_3d.shape[1], device=DEVICE)
                        
                        for sighting in obj_sightings:
                            cam_idx = sighting['camera_index']
                            gt_uv = torch.tensor([sighting['x'], sighting['y']], device=DEVICE)
                            
                            proj_uv = projector.project(pred_3d[b].unsqueeze(0), cam_idx).squeeze(0)
                            dist = torch.abs(proj_uv - gt_uv).sum(dim=1)
                            obj_loss += dist
                            
                        cost_matrix[i] = obj_loss

                    cost_cpu = cost_matrix.detach().cpu().numpy()
                    row_idx, col_idx = linear_sum_assignment(cost_cpu)
                    
                    loss += cost_matrix[row_idx, col_idx].mean()
                    
                    target_conf = torch.zeros_like(pred_conf[b])
                    target_conf[col_idx] = 1.0
                    loss += nn.functional.binary_cross_entropy(pred_conf[b], target_conf)
                else:
                    loss += nn.functional.binary_cross_entropy(pred_conf[b], torch.zeros_like(pred_conf[b]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        if (epoch+1) % 100 == 0:
            # Append num cameras to model path
            base_name, ext = os.path.splitext(args.model_path)
            save_path = f"{base_name}_{num_cameras}cam{ext}"
            
            model_dir = os.path.dirname(save_path)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    base_name, ext = os.path.splitext(args.model_path)
    save_path = f"{base_name}_{num_cameras}cam{ext}"
    model_dir = os.path.dirname(save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)

# --- LABELING MODE ---


class LiveStream:
    """
    Threaded PyAV reader to ensure we always get the absolute latest frame 
    from a TCP stream without buffering lag.
    """
    def __init__(self, uri):
        self.uri = uri
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        options = {
            'rtsp_transport': 'tcp',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }
        while self.running:
            try:
                container = av.open(self.uri, options=options, mode='r')
                stream = next(s for s in container.streams if s.type == 'video')
                stream.thread_type = "SLICE"
                
                for av_frame in container.decode(stream):
                    if not self.running: break
                    
                    # PyAV returns RGB, OpenCV needs BGR
                    img_rgb = av_frame.to_ndarray(format='rgb24')
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    
                    with self.lock:
                        self.latest_frame = img_bgr
                        
                container.close()
            except (av.error.TimeoutError, av.error.InvalidDataError, Exception) as e:
                # print(f"Stream error {self.uri}: {e}. Retrying...")
                if not self.running: break
                cv2.waitKey(1000) # Wait before reconnect

    def read(self):
        """Behaves like cv2.VideoCapture.read() but always returns the freshest frame."""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)

def label(args):
    source_mode = args.source
    print(f"Starting labeling in '{source_mode}' mode...")
    
    # Ensure output directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    num_cameras = 4
    cam_keys = [f"camera_{i}" for i in range(num_cameras)]
    
    for key in cam_keys:
        os.makedirs(os.path.join(TRAIN_DIR, key), exist_ok=True)

    dataset = None
    caps = []

    if source_mode == "lerobot":
        print(f"Loading source dataset {LEROBOT_SOURCE_ID}...")
        dataset = LeRobotDataset(LEROBOT_SOURCE_ID)
        # LeRobot keys we care about
        lerobot_keys_map = {
            0: "observation.images.anchor_camera_0",
            1: "observation.images.anchor_camera_1"
        }
    elif source_mode == "live":
        print("Connecting to live cameras...")
        for cal in CAM_CALIBRATIONS:
            caps.append(LiveStream(cal['uri'])) # Use threaded LiveStream

    print("Controls:")
    print("  Left Click: Add sighting for current object")
    print("  'm':        New Object (Change Color)")
    print("  SPACE:      Save & Next Frame")
    print("  's':        Skip Frame")
    print("  'q':        Quit")

    while True:
        imgs_np_display = []
        imgs_raw_np = {}
        
        # --- FETCH FRAMES ---
        if source_mode == "lerobot":
            # Pick random frame
            idx = np.random.randint(0, len(dataset))
            item = dataset[idx]
            
            # Initialize all 4 as black
            frames_buffer = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(num_cameras)]
            
            valid_frame = False
            for lr_idx, global_idx in merged_five_mapping.items():
                lr_key = lerobot_keys_map[lr_idx]
                if lr_key in item:
                    valid_frame = True
                    img_t = item[lr_key]
                    img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # Convert RGB (LeRobot) to BGR (OpenCV) for display/saving
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    frames_buffer[global_idx] = img_bgr

            if not valid_frame:
                continue
            
            # Populate raw dict
            for i, img in enumerate(frames_buffer):
                imgs_raw_np[f"camera_{i}"] = img
                imgs_np_display.append(img)

        elif source_mode == "live":
            # Grab live frames
            for i, cap in enumerate(caps):
                ret, img = cap.read()
                if not ret:
                    img = np.zeros((360, 640, 3), dtype=np.uint8)
                else:
                     # Ensure size consistency if needed
                    if img.shape[:2] != (360, 640):
                        img = cv2.resize(img, (640, 360))
                
                imgs_raw_np[f"camera_{i}"] = img
                imgs_np_display.append(img)
        
        # Detect which cameras are actually working (non-black)
        valid_cam_indices = [i for i, img in enumerate(imgs_np_display) if np.max(img) > 0]
        if not valid_cam_indices: valid_cam_indices = list(range(num_cameras)) # Fallback

        # --- TILE IMAGES (2x2 Grid) ---
        # Row 0: Cam 0, Cam 1
        # Row 1: Cam 2, Cam 3
        row0 = np.hstack([imgs_np_display[0], imgs_np_display[1]])
        row1 = np.hstack([imgs_np_display[2], imgs_np_display[3]])
        tiled_img = np.vstack([row0, row1])
        
        H, W = 360, 640
        
        # Labeling State
        current_objects = [] 
        active_obj_idx = 0
        current_objects.append([]) 
        
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

        def mouse_cb(event, x, y, flags, param):
            nonlocal active_obj_idx
            if event == cv2.EVENT_LBUTTONDOWN:
                # Determine camera index from 2x2 grid
                col = 0 if x < W else 1
                row = 0 if y < H else 1
                
                cam_idx = row * 2 + col
                
                local_x = x % W
                local_y = y % H
                
                sighting = {'camera_index': cam_idx, 'x': local_x, 'y': local_y}
                current_objects[active_obj_idx].append(sighting)
                
                # Auto-advance if we clicked all VALID cameras
                cameras_clicked = set(s['camera_index'] for s in current_objects[active_obj_idx])
                valid_clicked = cameras_clicked.intersection(set(valid_cam_indices))
                
                if len(valid_clicked) >= len(valid_cam_indices):
                    active_obj_idx += 1
                    current_objects.append([])
                    print(f"Auto-switched to Object {active_obj_idx}")

        cv2.namedWindow("Labeler")
        cv2.setMouseCallback("Labeler", mouse_cb)
        
        loop_frame = True
        while loop_frame:
            # Live stream automatically drains via thread
            # We just re-render the display loop to catch new frames if in live mode
            if source_mode == "live":
                 # Update display images with latest frames
                for i, cap in enumerate(caps):
                    ret, img = cap.read()
                    if ret:
                        if img.shape[:2] != (360, 640): img = cv2.resize(img, (640, 360))
                        imgs_np_display[i] = img
                
                # Re-Tile
                row0 = np.hstack([imgs_np_display[0], imgs_np_display[1]])
                row1 = np.hstack([imgs_np_display[2], imgs_np_display[3]])
                tiled_img = np.vstack([row0, row1])

            display = tiled_img.copy()
            
            # Draw objects
            for obj_i, sightings in enumerate(current_objects):
                color = colors[obj_i % len(colors)]
                for s in sightings:
                    idx = s['camera_index']
                    offset_x = (idx % 2) * W
                    offset_y = (idx // 2) * H
                    
                    cx = s['x'] + offset_x
                    cy = s['y'] + offset_y
                    cv2.circle(display, (cx, cy), 5, color, -1)

            cv2.putText(display, f"Active Object: {active_obj_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[active_obj_idx % len(colors)], 2)
            
            cv2.imshow("Labeler", display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                if source_mode == "live":
                    for cap in caps: cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('m'):
                active_obj_idx += 1
                current_objects.append([])
            elif key == ord('s'):
                loop_frame = False
            elif key == ord(' '):
                # Save
                final_objects = [obj for obj in current_objects if len(obj) > 0]
                
                if len(final_objects) > 0:
                    sample_id = str(uuid.uuid4())
                    image_paths_map = {}
                    
                    # Save Images
                    for key, img_bgr in imgs_raw_np.items():
                        rel_path = os.path.join(key, f"{sample_id}.jpg")
                        full_path = os.path.join(TRAIN_DIR, rel_path)
                        cv2.imwrite(full_path, img_bgr)
                        image_paths_map[key] = rel_path
                    
                    # Write Metadata
                    entry = {
                        "id": sample_id,
                        "image_paths": image_paths_map,
                        "objects": final_objects
                    }
                    
                    with open(os.path.join(TRAIN_DIR, LABELS_FILE), 'a') as f:
                        f.write(json.dumps(entry) + "\n")
                    
                    print(f"Saved sample {sample_id} with {len(final_objects)} objects.")
                else:
                    print("No objects labeled, not saving.")
                loop_frame = False
        
        if key == ord('q'):
            break

def split_and_upload(args):
    """
    1. Collects all data from local 'train' and 'eval' folders.
    2. Shuffles them.
    3. Splits 90/10 into train/eval.
    4. Moves files to correct folders.
    5. Regenerates labels.jsonl for both.
    6. Updates README.
    7. Uploads to HF.
    """
    print(f"Preparing to split dataset in {DATA_ROOT}...")
    
    # Ensure directories exist
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    if not os.path.exists(EVAL_DIR): os.makedirs(EVAL_DIR)
    
    train_meta = os.path.join(TRAIN_DIR, LABELS_FILE)
    eval_meta = os.path.join(EVAL_DIR, LABELS_FILE)

    all_samples = []

    # Helper: Load samples and tag their current location
    def load_samples(meta_path, source_split_dir, split_name):
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry['_current_split_dir'] = source_split_dir
                        entry['_current_split_name'] = split_name
                        all_samples.append(entry)

    load_samples(train_meta, TRAIN_DIR, "train")
    load_samples(eval_meta, EVAL_DIR, "eval")

    if not all_samples:
        print("No data found in local folders.")
        return

    print(f"Found {len(all_samples)} total samples. Shuffling and splitting 90/10...")
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train_set = all_samples[:split_idx]
    eval_set = all_samples[split_idx:]
    
    print(f"New distribution -> Train: {len(train_set)} | Eval: {len(eval_set)}")

    # Helper: Move files and write labels
    def process_split(sample_list, target_split_dir, target_meta_path):
        # Open in write mode to overwrite old labels
        with open(target_meta_path, 'w') as f:
            for entry in sample_list:
                current_split_dir = entry.pop('_current_split_dir')
                entry.pop('_current_split_name') # Remove tracking info

                image_paths = entry['image_paths']
                
                # Check and move images
                for cam_key, rel_path in image_paths.items():
                    src_path = os.path.join(current_split_dir, rel_path)
                    dst_path = os.path.join(target_split_dir, rel_path)
                    
                    # Ensure dest subdirectory exists (e.g. eval/obs.image.../)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    if src_path != dst_path:
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
                        elif os.path.exists(dst_path):
                            # File already there (e.g. shuffled back to same split), do nothing
                            pass
                        else:
                            print(f"Warning: Image missing at {src_path}")
                
                f.write(json.dumps(entry) + "\n")

    process_split(train_set, TRAIN_DIR, train_meta)
    process_split(eval_set, EVAL_DIR, eval_meta)

    # Update README
    readme_path = os.path.join(DATA_ROOT, "README.md")
    with open(readme_path, "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write("    path: train/labels.jsonl\n")
        f.write("  - split: test\n")
        f.write("    path: eval/labels.jsonl\n")
        f.write("---\n")

    upload_prompt(args)

def upload_prompt(args):
    if not os.path.exists(DATA_ROOT): return
    
    print("\n" + "="*30)
    print(f"Data organized in '{DATA_ROOT}'")
    confirm = input(f"Upload to {args.dataset_id}? (y/n): ").strip().lower()
    
    if confirm == 'y':
        api = HfApi()
        create_repo(args.dataset_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=DATA_ROOT,
            repo_id=args.dataset_id,
            repo_type="dataset"
        )
        print("Uploaded successfully.")

def eval_mode(args):
    # Setup Eval Data
    eval_set = SockTriangulationDataset(EVAL_DIR)
    
    if len(eval_set) == 0:
        print(f"No data found in {EVAL_DIR}. Run split?")
        eval_set = SockTriangulationDataset(TRAIN_DIR)
        if len(eval_set) == 0: return

    # Update model load to expect 4 cameras
    num_cameras = 4
    print(f"Loading model from {args.model_path}...")
    model = MultiViewDETR(num_cameras=num_cameras).to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    
    model.eval()
    projector = WorldToCameraProjector(CAM_CALIBRATIONS, DEVICE)

    print("\n--- Evaluation Controls ---")
    print("SPACE: Next random sample")
    print("Q:     Quit")
    
    cv2.namedWindow("Evaluation")
    
    while True:
        idx = np.random.randint(0, len(eval_set))
        img_stack, gt_objects = eval_set[idx]
        batch_imgs = img_stack.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_conf, pred_3d = model(batch_imgs)
        
        display_imgs = []
        for i in range(4): # 4 cameras
            img = img_stack[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            display_imgs.append(img)
            
        CONF_THRESH = 0.5
        valid_preds_mask = pred_conf[0, :, 0] > CONF_THRESH
        valid_pred_3d = pred_3d[0, valid_preds_mask]
        valid_confs = pred_conf[0, valid_preds_mask]
        
        for cam_idx in range(4):
            img = display_imgs[cam_idx]
            
            # Draw GT
            for obj_sightings in gt_objects:
                for sighting in obj_sightings:
                    if sighting['camera_index'] == cam_idx:
                        cv2.circle(img, (int(sighting['x']), int(sighting['y'])), 8, (0, 255, 0), 2)
            
            # Draw Preds
            if len(valid_pred_3d) > 0:
                proj_uv = projector.project(valid_pred_3d.unsqueeze(0), cam_idx).squeeze(0)
                for j in range(len(proj_uv)):
                    x, y = int(proj_uv[j, 0]), int(proj_uv[j, 1])
                    conf = valid_confs[j].item()
                    cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                    cv2.putText(img, f"{conf:.2f}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 2x2 Grid for Eval too
        row0 = np.hstack([display_imgs[0], display_imgs[1]])
        row1 = np.hstack([display_imgs[2], display_imgs[3]])
        tiled = np.vstack([row0, row1])
        
        cv2.imshow("Evaluation", tiled)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    train_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    eval_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    label_parser.add_argument("--source", choices=["lerobot", "live"], default="lerobot")

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval_mode(args)
    elif args.command == "label":
        label(args)
    elif args.command == "split":
        split_and_upload(args)