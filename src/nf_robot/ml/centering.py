import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import uuid
import numpy as np
import argparse
import random
import shutil
import math
import tempfile
from importlib.resources import files

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_REPO_ID = "naavox/square-centering-dataset"
DEFAULT_MODEL_PATH = files("nf_robot.ml").joinpath("models/square_centering.pth")
LOCAL_DATASET_ROOT = "square_centering_data"
UNPROCESSED_DIR = "square_centering_data_unlabeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture Config
IMG_RES = 384  # Square resolution required

# ==========================================
# MODEL DEFINITION
# ==========================================

class CenteringNet(nn.Module):
    """
    Predicts:
    1. Vector (x, y) to target center.
    2. Probability target exists (0.0 to 1.0).
    3. Probability gripper contains object (0.0 to 1.0).
    4. Gripping angle (0 to pi).
    """

    def __init__(self):
        super().__init__()
        
        # INPUT 3 channels (RGB) + 2 channels (X, Y coordinates) = 5
        self.enc1 = self.conv_block(5, 32)
        self.pool1 = nn.MaxPool2d(2) # 384 -> 192
        
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2) # 192 -> 96
        
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2) # 96 -> 48
        
        self.bottleneck = self.conv_block(128, 256)
        self.spatial_pool = nn.AdaptiveMaxPool2d((6, 6))

        flat_features = 256 * 6 * 6
        
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        # HEAD 1: Vector Regressor (x, y)
        self.head_vector = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) 
        )

        # HEAD 2: Target Valid Classifier
        self.head_valid = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # HEAD 3: Gripper Occupied Classifier
        self.head_gripper = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # HEAD 4: Angle Regressor (0 to pi)
        self.head_angle = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid() # Scale this output by pi in forward
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def append_coords(self, x):
        batch_size, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1)
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width)
        y_channel = y_coords.expand(batch_size, 1, height, width)
        x_channel = x_coords.expand(batch_size, 1, height, width)
        return torch.cat([x, x_channel, y_channel], dim=1)

    def forward(self, x):
        x = self.append_coords(x)
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        x = self.spatial_pool(x)
        
        feats = self.shared_fc(x)
        
        vector_out = self.head_vector(feats)
        valid_out = self.head_valid(feats)
        gripper_out = self.head_gripper(feats)
        # Scale sigmoid [0,1] to [0, pi]
        angle_out = self.head_angle(feats) * math.pi
        
        return vector_out, valid_out, gripper_out, angle_out

# ==========================================
# DATASET & UTILS
# ==========================================

class SockDataset(Dataset):
    def __init__(self, root_dir, training=True):
        self.data_dir = os.path.join(root_dir, "train" if training else "eval")
        self.metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
        
        if not os.path.exists(self.metadata_path):
            self.data_dir = os.path.join(root_dir, "train")
            self.metadata_path = os.path.join(self.data_dir, "metadata.jsonl")

        self.samples = []
        if os.path.exists(self.metadata_path):
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
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
            
        h, w = img.shape[:2]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        points = item.get("points", [])
        gripper_occ = 1.0 if item.get("gripper_occupied", False) else 0.0
        angle = item.get("angle", 0.0) # 0 to pi
        
        has_target = 0.0
        cx, cy = 0.0, 0.0 
        
        if len(points) > 0:
            has_target = 1.0
            pt = points[0]
            raw_x, raw_y = (pt[0], pt[1]) if isinstance(pt, (list, tuple)) else (pt.get('x', 0), pt.get('y', 0))
            
            cx = (raw_x - (w / 2)) / (w / 2)
            cy = (raw_y - (h / 2)) / (h / 2)
            
        targets = {
            "vector": torch.tensor([cx, cy], dtype=torch.float32),
            "has_target": torch.tensor([has_target], dtype=torch.float32),
            "gripper": torch.tensor([gripper_occ], dtype=torch.float32),
            "angle": torch.tensor([angle], dtype=torch.float32)
        }
            
        return img_tensor, targets

def draw_gripping_line(img, center, angle, color=(255, 0, 255), length=40):
    """
    Draws a line representing the gripper fingers crossing the object.
    The angle is relative to the vertical (y-axis).
    """
    dx = math.sin(angle) * (length / 2)
    dy = -math.cos(angle) * (length / 2)
    
    p1 = (int(center[0] - dx), int(center[1] - dy))
    p2 = (int(center[0] + dx), int(center[1] + dy))
    
    cv2.line(img, p1, p2, color, 2)
    cv2.circle(img, p1, 3, color, -1)
    cv2.circle(img, p2, 3, color, -1)

# ==========================================
# TRAINING LOOP
# ==========================================

def train(args):
    from huggingface_hub import snapshot_download
    print(f"Loading dataset from {args.dataset_id}...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")

    dataset = SockDataset(dataset_path, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    model = CenteringNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_mse = nn.MSELoss(reduction='none') 
    criterion_bce = nn.BCELoss()

    print(f"Training on {len(dataset)} samples...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, targets in dataloader:
            imgs = imgs.to(DEVICE)
            gt_vec = targets["vector"].to(DEVICE)
            gt_valid = targets["has_target"].to(DEVICE)
            gt_grip = targets["gripper"].to(DEVICE)
            gt_angle = targets["angle"].to(DEVICE)
            
            optimizer.zero_grad()
            pred_vec, pred_valid, pred_grip, pred_angle = model(imgs)
            
            loss_valid = criterion_bce(pred_valid, gt_valid)
            loss_grip = criterion_bce(pred_grip, gt_grip)
            
            # Mask for vector prediction: only if target exists
            valid_mask = gt_valid.squeeze()
            
            vec_mse = criterion_mse(pred_vec, gt_vec).mean(dim=1)
            loss_vec = (vec_mse * valid_mask).mean()
            
            # Mask for angle prediction: only if target exists AND gripper is empty.
            angle_mask = valid_mask * (1.0 - gt_grip.squeeze())
            angle_mse = criterion_mse(pred_angle, gt_angle).squeeze()
            loss_angle = (angle_mse * angle_mask).mean()
            
            loss = loss_vec + loss_valid + loss_grip + loss_angle
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Saved: {args.model_path}")

# ==========================================
# EVALUATION MODE
# ==========================================

def eval_mode(args):
    from huggingface_hub import snapshot_download
    model = CenteringNet().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    data_dir = os.path.join(dataset_path, "eval")
    if not os.path.exists(data_dir): data_dir = os.path.join(dataset_path, "train")
    
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    samples = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))

    while True:
        sample = random.choice(samples)
        img_input = cv2.imread(os.path.join(data_dir, sample["file_name"]))
        if img_input is None: continue

        h, w = img_input.shape[:2]
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_vec, pred_valid, pred_grip, pred_angle = model(img_tensor)

        vec = pred_vec[0].cpu().numpy()
        p_valid = pred_valid[0].item()
        p_grip = pred_grip[0].item()
        angle = pred_angle[0].item()

        display = img_input.copy()
        cx, cy = w // 2, h // 2
        tx, ty = cx + int(vec[0] * cx), cy + int(vec[1] * cy)

        if p_valid > 0.5:
            cv2.arrowedLine(display, (cx, cy), (tx, ty), (0, 255, 0), 2)
            draw_gripping_line(display, (tx, ty), angle)
        
        cv2.putText(display, f"Angle: {math.degrees(angle):.1f}deg", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow("Inference", display)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

# ==========================================
# LABELER
# ==========================================

current_clicks = []
current_image = None
current_gripper_state = False 
current_angle = 0.0 # Radians

def mouse_callback(event, x, y, flags, param):
    global current_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        current_clicks = [(x, y)]
        draw_label_interface()
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_clicks = []
        draw_label_interface()

def draw_label_interface():
    global current_image, current_clicks, current_gripper_state, current_angle
    if current_image is None: return
    display = current_image.copy()
    
    for pt in current_clicks:
        cv2.circle(display, pt, 5, (0, 255, 0), -1)
        draw_gripping_line(display, pt, current_angle, color=(0, 255, 0))

    col = (0, 255, 255) if current_gripper_state else (200, 200, 200)
    cv2.putText(display, f"Gripper: {'Full' if current_gripper_state else 'Empty'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(display, f"Angle: {math.degrees(current_angle):.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(display, "[A]/[D]: Rotate | [G]: Toggle Grip | [SPACE]: Save", (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow("Labeler", display)

def label_mode(args):
    global current_clicks, current_image, current_gripper_state, current_angle
    
    # We work directly within the local square_centering_data folder
    source_dir = LOCAL_DATASET_ROOT
    train_dir = os.path.join(source_dir, "train")
    metadata_path = os.path.join(train_dir, "metadata.jsonl")

    if not os.path.exists(metadata_path):
        print(f"Error: metadata.jsonl not found at {metadata_path}")
        return

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeler", IMG_RES*2, IMG_RES*2)
    cv2.setMouseCallback("Labeler", mouse_callback)

    existing_labels = []
    with open(metadata_path, 'r') as f:
        existing_labels = [json.loads(line) for line in f if line.strip()]

    queue = []
    if args.relabel:
        print(f"Relabeling local dataset in {train_dir}...")
        for entry in existing_labels:
            # Filter criteria: valid targets, no angle, no held objects
            has_points = len(entry.get("points", [])) > 0
            is_empty = not entry.get("gripper_occupied", False)
            missing_angle = "angle" not in entry
            
            if has_points and is_empty and missing_angle:
                queue.append(("labeled", entry))
        print(f"Found {len(queue)} samples needing updates.")
    else:
        if os.path.exists(UNPROCESSED_DIR):
            fns = [f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.jpg')]
            queue = [("unlabeled", fn) for fn in fns]

    if not queue:
        print("No work found in queue.")
        return

    while queue:
        work_type, data = queue.pop(0)
        
        if work_type == "unlabeled":
            full_path = os.path.join(UNPROCESSED_DIR, data)
            current_gripper_state = "_g1" in data
            current_image = cv2.imread(full_path)
            current_clicks = []
            current_angle = 0.0
            filename_to_save = f"{uuid.uuid4()}.jpg"
        else:
            entry = data
            full_path = os.path.join(train_dir, entry["file_name"])
            current_gripper_state = entry.get("gripper_occupied", False)
            current_image = cv2.imread(full_path)
            current_clicks = [(p[0], p[1]) if isinstance(p, list) else (p['x'], p['y']) for p in entry["points"]]
            current_angle = entry.get("angle", 0.0)
            filename_to_save = entry["file_name"]

        if current_image is None: 
            print(f"Warning: Could not read image {full_path}")
            continue
            
        draw_label_interface()
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): 
                return
            elif key == ord('a'):
                current_angle = (current_angle - 0.1) % math.pi
                draw_label_interface()
            elif key == ord('d'):
                current_angle = (current_angle + 0.1) % math.pi
                draw_label_interface()
            elif key == ord('g'): 
                current_gripper_state = not current_gripper_state
                draw_label_interface()
            elif key == ord(' '):
                # Update existing entry list in memory and write back to file
                new_entry = {
                    "file_name": filename_to_save,
                    "points": current_clicks,
                    "target_valid": len(current_clicks) > 0,
                    "gripper_occupied": current_gripper_state,
                    "angle": current_angle
                }
                
                if work_type == "unlabeled":
                    shutil.move(full_path, os.path.join(train_dir, filename_to_save))
                    existing_labels.append(new_entry)
                else:
                    for i, old_entry in enumerate(existing_labels):
                        if old_entry["file_name"] == filename_to_save:
                            existing_labels[i] = new_entry
                            break
                
                # Immediate write to ensure progress isn't lost if we quit
                with open(metadata_path, 'w') as f:
                    for e in existing_labels:
                        f.write(json.dumps(e) + "\n")
                break
    
    print("Labeling session finished.")

# ==========================================
# SPLITTER & UPLOADER
# ==========================================

def split_and_upload(args):
    """
    1. Collects all data from local 'train' and 'eval' folders.
    2. Shuffles them.
    3. Splits 90/10 into train/eval.
    4. Moves files to correct folders.
    5. Regenerates metadata.jsonl for both.
    6. Updates README.
    7. Uploads to HF.
    """
    print(f"Preparing to split dataset in {LOCAL_DATASET_ROOT}...")
    
    train_dir = os.path.join(LOCAL_DATASET_ROOT, "train")
    eval_dir = os.path.join(LOCAL_DATASET_ROOT, "eval")
    train_meta = os.path.join(train_dir, "metadata.jsonl")
    eval_meta = os.path.join(eval_dir, "metadata.jsonl")

    # Ensure directories exist
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    all_samples = []

    # Helper: Load samples and tag their current location
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

    # Helper: Move files and write metadata
    def process_split(sample_list, target_split, target_dir, target_meta_path):
        # Open in write mode to overwrite old metadata
        with open(target_meta_path, 'w') as f:
            for entry in sample_list:
                current_split = entry.pop('_current_split')
                fname = entry['file_name']
                
                # Move file if it's not in the right folder
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

    # Update README
    readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
    with open(readme_path, "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write("    path: train/metadata.jsonl\n")
        f.write("  - split: test\n")
        f.write("    path: eval/metadata.jsonl\n")
        f.write("---\n")

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
        api.upload_large_folder(
            folder_path=LOCAL_DATASET_ROOT,
            repo_id=args.dataset_id,
            repo_type="dataset"
        )
        print("Uploaded successfully.")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    train_p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch_size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-4)

    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    eval_p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    label_p = subparsers.add_parser("label")
    label_p.add_argument("--relabel", action="store_true", help="Relabel local square_centering_data missing angles")
    
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