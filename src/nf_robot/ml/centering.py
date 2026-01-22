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

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_REPO_ID = "naavox/square-centering-dataset"
DEFAULT_MODEL_PATH = "trainer/models/square_centering.pth"
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
    1. Vector (x, y) to target.
    2. Probability target exists (0.0 to 1.0).
    3. Probability gripper contains object (0.0 to 1.0).
    Input: 384x384 RGB Image.
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
        
        # Adapt to specific spatial size for the fully connected layer
        # For 384 input, passed through 3 pools (divide by 8) -> 48x48 features
        # We adaptive pool down to 6x6 to keep parameter count reasonable
        self.spatial_pool = nn.AdaptiveMaxPool2d((6, 6))

        # Flattened: 256 channels * 6 * 6 = 9216
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

        # HEAD 2: Target Valid Classifier (Binary)
        self.head_valid = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # HEAD 3: Gripper Occupied Classifier (Binary)
        self.head_gripper = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
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
        """Generates and appends (x, y) coordinate channels."""
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
        
        return vector_out, valid_out, gripper_out

# ==========================================
# DATASET & UTILS
# ==========================================

class SockDataset(Dataset):
    def __init__(self, root_dir, training=True):
        self.data_dir = os.path.join(root_dir, "train")
        self.metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
        self.training = training
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Could not find metadata at {self.metadata_path}")
            
        self.samples = []
        with open(self.metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        
        img = cv2.imread(img_path)
        
        # Robust handling for missing files
        if img is None:
            # Recursively try another random index
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
            
        h, w = img.shape[:2]
        
        # Strict Size Check - No Resizing
        if h != IMG_RES or w != IMG_RES:
            print(f"[DATASET WARNING] Skipping {item['file_name']}: Size {w}x{h} does not match required {IMG_RES}x{IMG_RES}.")
            # Recursively try another random index to maintain batch size
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
            
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Extract Targets
        points = item.get("points", [])
        gripper_occ = 1.0 if item.get("gripper_occupied", False) else 0.0
        
        has_target = 0.0
        cx, cy = 0.0, 0.0 # Default center if no target
        
        if len(points) > 0:
            has_target = 1.0
            pt = points[0]
            if isinstance(pt, (list, tuple)):
                raw_x, raw_y = pt[0], pt[1]
            else:
                raw_x, raw_y = pt.get('x', 0), pt.get('y', 0)
            
            # Normalize using the ACTUAL dimensions (which we verified match IMG_RES)
            # Center of image is (0, 0) -> norm -1 to 1
            norm_x = (raw_x - (w / 2)) / (w / 2)
            norm_y = (raw_y - (h / 2)) / (h / 2)
            cx, cy = norm_x, norm_y
            
        targets = {
            "vector": torch.tensor([cx, cy], dtype=torch.float32),
            "has_target": torch.tensor([has_target], dtype=torch.float32),
            "gripper": torch.tensor([gripper_occ], dtype=torch.float32)
        }
            
        return img_tensor, targets

def capture_gripper_image(ndimage, gripper_occupied=False):
    """
    Saves an image to the unprocessed directory. 
    Encodes gripper state in filename: {uuid}_g{1|0}.jpg
    """
    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)
    
    h, w = ndimage.shape[:2]
    if h != IMG_RES or w != IMG_RES:
        print(f"[CAPTURE WARNING] Image resolution is {w}x{h}, but network expects {IMG_RES}x{IMG_RES}.")
        
    state_str = "g1" if gripper_occupied else "g0"
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}_{state_str}.jpg"
    img_full_path = os.path.join(UNPROCESSED_DIR, img_filename)
    
    # Save (ensure RGB/BGR consistency)
    cv2.imwrite(img_full_path, ndimage)
    print(f"Captured: {img_filename} (Gripper: {gripper_occupied})")

# ==========================================
# TRAINING LOOP
# ==========================================

def train(args):
    from huggingface_hub import snapshot_download
    print(f"Loading dataset from {args.dataset_id}...")
    try:
        dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    dataset = SockDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    model = CenteringNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Losses
    criterion_mse = nn.MSELoss(reduction='none') # For masking
    criterion_bce = nn.BCELoss()

    print(f"Starting training on {len(dataset)} images for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for imgs, targets in dataloader:
            imgs = imgs.to(DEVICE)
            gt_vec = targets["vector"].to(DEVICE)
            gt_valid = targets["has_target"].to(DEVICE)
            gt_grip = targets["gripper"].to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_vec, pred_valid, pred_grip = model(imgs)
            
            # Target Valid Loss (Binary)
            loss_valid = criterion_bce(pred_valid, gt_valid)
            
            # Gripper Occupied Loss (Binary)
            loss_grip = criterion_bce(pred_grip, gt_grip)
            
            # Vector Loss (Masked - only if target exists)
            raw_mse = criterion_mse(pred_vec, gt_vec).mean(dim=1) # Average over x,y
            # gt_valid is (Batch, 1), squeeze to (Batch)
            masked_mse = (raw_mse * gt_valid.squeeze()).mean()
            
            # Combined Loss
            loss = masked_mse + loss_valid + loss_grip
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Total Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

# ==========================================
# EVALUATION LOOP
# ==========================================

def eval_mode(args):
    from huggingface_hub import snapshot_download
    print(f"Loading model from {args.model_path}...")
    model = CenteringNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except FileNotFoundError:
        print("Model file not found.")
        return
    model.eval()

    print(f"Downloading dataset {args.dataset_id} for samples...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    
    # Eval command now explicitly uses 'eval' folder
    data_dir = os.path.join(dataset_path, "eval")
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    if not os.path.exists(metadata_path):
        print(f"No eval metadata found at {metadata_path}. Did you run 'split' on the dataset?")
        return

    # Load metadata
    samples = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
            
    print(f"Loaded {len(samples)} evaluation samples.")
    print("Controls: [SPACE] Next, [Q] Quit")

    while True:
        sample = random.choice(samples)
        img_path = os.path.join(data_dir, sample["file_name"])
        img_input = cv2.imread(img_path)
        
        if img_input is None: 
            continue

        h, w = img_input.shape[:2]
        if h != IMG_RES or w != IMG_RES:
            print(f"[EVAL WARNING] Skipping {sample['file_name']}: Size {w}x{h} != {IMG_RES}x{IMG_RES}")
            continue

        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_vec, pred_valid, pred_grip = model(img_tensor)

        # Parse Predictions
        vec = pred_vec[0].cpu().numpy()
        p_valid = pred_valid[0].item()
        p_grip = pred_grip[0].item()

        # Visualization
        display = img_input.copy()
        cx, cy = w // 2, h // 2
        
        # Denormalize vector
        dx = int(vec[0] * (w/2))
        dy = int(vec[1] * (h/2))
        target_x, target_y = cx + dx, cy + dy

        # Draw UI
        color_valid = (0, 255, 0) if p_valid > 0.5 else (0, 0, 255)
        color_grip = (0, 255, 255) if p_grip > 0.5 else (100, 100, 100)

        # Crosshair
        cv2.line(display, (cx-10, cy), (cx+10, cy), (100,100,100), 1)
        cv2.line(display, (cx, cy-10), (cx, cy+10), (100,100,100), 1)

        # Prediction Arrow (only if valid)
        if p_valid > 0.5:
            cv2.arrowedLine(display, (cx, cy), (target_x, target_y), color_valid, 2, tipLength=0.1)
            cv2.circle(display, (target_x, target_y), 5, color_valid, -1)
        
        # Info Box
        cv2.rectangle(display, (5, 5), (220, 80), (0,0,0), -1)
        cv2.putText(display, f"Valid: {p_valid:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_valid, 2)
        cv2.putText(display, f"Grip:  {p_grip:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_grip, 2)
        cv2.putText(display, f"Vec: ({vec[0]:.2f}, {vec[1]:.2f})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Inference", display)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# ==========================================
# LABELER
# ==========================================

# Global state for labeler
current_clicks = []
current_image = None
current_gripper_state = False 

def mouse_callback(event, x, y, flags, param):
    global current_clicks, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        current_clicks = [(x, y)] # Single point mode usually, or append
        draw_label_interface()
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_clicks = []
        draw_label_interface()

def draw_label_interface():
    global current_image, current_clicks, current_gripper_state
    if current_image is None: return
    
    display = current_image.copy()
    
    for pt in current_clicks:
        cv2.circle(display, pt, 5, (0, 255, 0), -1)

    # HUD
    status = "Active" if current_gripper_state else "Empty"
    col = (0, 255, 255) if current_gripper_state else (200, 200, 200)
    
    cv2.putText(display, f"Gripper [g]: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(display, f"Points: {len(current_clicks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    help_txt = "[SPACE] Save | [n] Save No Target | [q] Quit"
    cv2.putText(display, help_txt, (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Labeler", display)

def label_mode(args):
    global current_clicks, current_image, current_gripper_state
    
    TRAIN_DIR = os.path.join(LOCAL_DATASET_ROOT, "train")
    METADATA_PATH = os.path.join(TRAIN_DIR, "metadata.jsonl")
    
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
    
    # Initialize README if missing
    readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train/metadata.jsonl\n---\n")

    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", mouse_callback)
    
    print("Labeler Started.")
    print("Files are loaded from:", UNPROCESSED_DIR)

    while True:
        # Get next file
        if not os.path.exists(UNPROCESSED_DIR):
            print(f"No unprocessed directory: {UNPROCESSED_DIR}")
            break
            
        files = [f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.jpg')]
        if not files:
            print("No more files to process.")
            upload_prompt(args)
            break
            
        fn = random.choice(files)
        full_path = os.path.join(UNPROCESSED_DIR, fn)
        
        # Decode gripper state from filename (uuid_g1.jpg or uuid_g0.jpg)
        current_gripper_state = "_g1" in fn
        
        current_image = cv2.imread(full_path)
        
        if current_image is None:
            continue
            
        h, w = current_image.shape[:2]
        if h != IMG_RES or w != IMG_RES:
            print(f"[LABEL WARNING] Skipping {fn}: Size {w}x{h} != {IMG_RES}x{IMG_RES}. Please delete or resize this file.")
            # Skip this iteration to find a new file. 
            # Note: In a real app we might want to move/delete bad files to stop them from reappearing.
            continue
            
        current_clicks = []
        
        draw_label_interface()
        
        save_it = False
        no_target = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                upload_prompt(args)
                return
            elif key == ord(' '):
                if len(current_clicks) > 0:
                    save_it = True
                    break
                else:
                    print("No points marked. Use 'n' for no target or click a point.")
                    save_it = True
                    no_target = True
                    break
            elif key == ord('n'):
                save_it = True
                no_target = True
                current_clicks = [] # Ensure empty
                break
            elif key == ord('g'):
                current_gripper_state = not current_gripper_state
                print(f"Gripper overridden to: {current_gripper_state}")
                draw_label_interface()

        if save_it:
            # Move image to dataset folder
            new_id = str(uuid.uuid4())
            new_fn = f"{new_id}.jpg"
            new_path = os.path.join(TRAIN_DIR, new_fn)
            shutil.move(full_path, new_path)
            
            # Write Metadata
            entry = {
                "file_name": new_fn,
                "points": current_clicks,
                "target_valid": not no_target,
                "gripper_occupied": current_gripper_state
            }
            with open(METADATA_PATH, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            
            status_msg = "Saved (No Target)" if no_target else f"Saved ({len(current_clicks)} pts)"
            print(f"{status_msg} | Grip: {current_gripper_state}")

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
        api.upload_folder(
            folder_path=LOCAL_DATASET_ROOT,
            repo_id=args.dataset_id,
            repo_type="dataset"
        )
        print("Uploaded successfully.")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CenteringNet System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    train_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    # Eval Command
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    eval_parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

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