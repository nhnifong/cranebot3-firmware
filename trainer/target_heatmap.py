import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
import argparse
from huggingface_hub import snapshot_download, HfApi, create_repo
import random
import uuid
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_REPO_ID = "naavox/target-heatmap-dataset"
DEFAULT_MODEL_PATH = "trainer/models/target_heatmap.pth"
LOCAL_DATASET_ROOT = "target_heatmap_data"
UNPROCESSED_DIR = "target_heatmap_data_unlabeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_RES = (640, 360)
MINIMUM_CONFIDENCE = 0.95 # during eval

# ==========================================
# MODEL DEFINITION
# ==========================================

class DobbyNet(nn.Module):
    """
    Learns a heatmap from images that have one or more labeled points
    The points are the locations of socks, hence the name.

    Input images are 640x360
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
        """
        Return an image and heatmap made from the stored points associated with the image
        """
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

    Args:
        heatmap (np.ndarray): 2D array of probabilities (0.0 to 1.0). Shape (H, W).
        top_n (int): Maximum number of targets to return.
        threshold (float): Minimum confidence value to consider a blob.

    Returns:
        np.ndarray(3,n): A list of (norm_x, norm_y, confidence), sorted by confidence.
                     Coordinates are normalized [0.0, 1.0].
    """
    # Threshold to find "hot" regions
    # Convert to uint8 mask (0 or 255)
    mask = (heatmap > threshold).astype(np.uint8) * 255

    # Find blobs (contours)
    # We use RETR_EXTERNAL because we only care about distinct outer blobs, not holes inside them.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        # Create a mask for just this blob to ensure we don't pick a peak outside it
        # (Simple bounding box ROI is usually sufficient and much faster)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the ROI from the heatmap
        roi = heatmap[y:y+h, x:x+w]
        
        # Find the max value and its location within this specific ROI
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        
        # Convert local ROI coordinates to global image coordinates
        global_x = x + max_loc[0]
        global_y = y + max_loc[1]
        
        candidates.append((global_x, global_y, max_val))

    # Sort by confidence (highest first)
    candidates.sort(key=lambda k: k[2], reverse=True)

    # Normalize coordinates and slice top N
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
    print(f"Downloading/Loading dataset from {args.dataset_id}...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    print(f"Dataset available at: {dataset_path}")

    dataset = DobbyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model = DobbyNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Use BCEWithLogitsLoss for pixel-wise heatmap regression
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

def eval_mode(args):
    print(f"Loading model from {args.model_path}...")
    model = DobbyNet().to(DEVICE)
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

    window_name = "Target Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size to double the processing resolution (1280x720)
    cv2.resizeWindow(window_name, IMAGE_RES[0] * 2, IMAGE_RES[1] * 2)

    while True:
        sample = random.choice(samples)
        img_path = os.path.join(data_dir, sample["file_name"])
        img_input = cv2.imread(img_path)
        
        if img_input is None: 
            continue

        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        batch = img_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            heatmap_out = model(batch)
        heatmap_np = heatmap_out.squeeze().cpu().numpy()
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_display = (img_np * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        heatmap_vis = (heatmap_np * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_display, 0.8, heatmap_color, 0.4, 0)

        targets = extract_targets_from_heatmap(heatmap_np)
        print(f'targets=\n{targets}')

        for x, y, confidence in targets:
            x = int(x * IMAGE_RES[0])
            y = int(y * IMAGE_RES[1])
            if confidence > 0.1: # Only draw if confidence is somewhat high
                box_size = 20
                top_left =     (x - box_size, y - box_size)
                bottom_right = (x + box_size, y + box_size)
                cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)
                conf_text = f"Max Conf: {confidence:.2f}"
                cv2.putText(overlay, conf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Target Heatmap", overlay)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# ==========================================
# LABELING TOOL
# ==========================================

# State
current_clicks = []
current_image = None

def mouse_callback(event, x, y, flags, param):
    global current_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point
        current_clicks.append((x, y))
        print(f"Added Point: {x}, {y}")
        # Redraw immediately for better responsiveness
        draw_interface()
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last point
        if current_clicks:
            removed = current_clicks.pop()
            print(f"Removed Point: {removed}")
            draw_interface()

def draw_interface():
    """Helper to redraw the image with points and HUD."""
    global current_image, current_clicks
    if current_image is None:
        return

    display = current_image.copy()

    # Draw all points
    for i, pt in enumerate(current_clicks):
        cv2.circle(display, pt, 5, (0, 255, 0), -1)

    # HUD
    status_text = f"Points: {len(current_clicks)}"
    cv2.putText(display, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # cv2.putText(display, "[Space] Save | [N] Skip | [Q] Quit & Upload", (10, 60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Labeler", display)

def label_mode(args):
    global current_clicks, current_image

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
    print("\n--- Instructions ---")
    print("1. Left-Click:   Mark a spot (x, y).")
    print("2. Right-Click:  Undo last mark.")
    print("3. SPACE:        Save to local JSONL and go to next.")
    print("4. 'n':          Skip to next random frame.")
    print("5. 'q':          Quit and ask to upload to Hugging Face.")

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
    
        current_image = cv2.imread(full_path)
        if current_image is None:
            continue
        # if current_image.shape != IMAGE_RES:
        #     raise ValueError(f"image shape {current_image.shape} from file {full_path} cannot be used in this dataset, require {IMAGE_RES}")

        current_clicks = [] 
        
        draw_interface()

        # Interaction Loop
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
            # Move image to dataset folder
            new_id = str(uuid.uuid4())
            new_fn = f"{new_id}.jpg"
            new_path = os.path.join(TRAIN_DIR, new_fn)
            shutil.move(full_path, new_path)
            
            # Write Metadata
            entry = {
                "file_name": fn,
                "points": current_clicks
            }
            with open(METADATA_PATH, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            
            print(f"Saved {len(current_clicks)} points -> {fn}")

def upload_prompt(args):
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
    train_parser.add_argument("--batch_size", type=int, default=28)
    train_parser.add_argument("--lr", type=float, default=1e-3)

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