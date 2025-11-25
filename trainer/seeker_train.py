import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
import argparse
from huggingface_hub import snapshot_download
from .seeker import SeekerNet

# Configuration Defaults
DEFAULT_REPO_ID = "naavox/gripper-spots-dataset"
DEFAULT_MODEL_PATH = "trainer/models/sock_gripper.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SockDataset(Dataset):
    def __init__(self, root_dir):
        self.data_dir = os.path.join(root_dir, "train")
        self.metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Could not find metadata at {self.metadata_path}")
            
        self.samples = []
        with open(self.metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # For regression, we strictly need a target. 
                    # If points are missing, we skip the sample or assume 0,0,0.
                    # Here we skip to ensure high-quality training data.
                    if data.get("points") and len(data["points"]) > 0:
                        self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            img_tensor: (3, H, W) normalized image
            vector_target: (3,) tensor representing (vx, vy, vz)
        """
        item = self.samples[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        h, w = img.shape[:2]
        
        # Process Image: (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Process Target Vector
        # We assume the first point is the target
        pt = item["points"][0]
        
        if isinstance(pt, (list, tuple)):
            cx, cy = pt[0], pt[1]
            cz = pt[2] if len(pt) > 2 else 0.0
        elif isinstance(pt, dict):
            cx = pt.get('x', 0)
            cy = pt.get('y', 0)
            cz = pt.get('z', 0.0) # Default to 0 if z is missing
        else:
            cx, cy, cz = w/2, h/2, 0.0

        # Normalize Coordinates to [-1, 1] range
        # (0, 0) becomes top-left (-1, -1), (w, h) becomes bottom-right (1, 1)
        # Center of image is (0, 0)
        norm_x = (cx - (w / 2)) / (w / 2)
        norm_y = (cy - (h / 2)) / (h / 2)
        
        # We assume Z is already normalized or small relative to pixel counts. 
        # If Z is in pixels, you might need to normalize it similarly.
        norm_z = float(cz)

        vector_target = torch.tensor([norm_x, norm_y, norm_z], dtype=torch.float32)
            
        return img_tensor, vector_target

def train(args):
    print(f"Downloading/Loading dataset from {args.dataset_id}...")
    try:
        dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print(f"Dataset available at: {dataset_path}")

    dataset = SockDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Initialize SeekerNet
    model = SeekerNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # MSELoss is standard for continuous vector regression
    criterion = nn.MSELoss() 

    print(f"Starting training on {len(dataset)} images for {args.epochs} epochs...")
    print(f"Device: {DEVICE}")

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(imgs)
            
            # Calculate regression loss
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, MSE Loss: {avg_loss:.6f}")

        # Optional: Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), args.model_path)

    torch.save(model.state_dict(), args.model_path)
    print(f"Final Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeekerNet on Gripper Spots Dataset")
    
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID, 
                        help=f"HuggingFace Dataset Repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f"Path to save the trained model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower is often better for regression)")

    args = parser.parse_args()
    train(args)