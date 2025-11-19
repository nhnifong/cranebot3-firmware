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
from .dobby import DobbyNet  

# Configuration Defaults
DEFAULT_REPO_ID = "naavox/laundry-spots-dataset"
DEFAULT_MODEL_PATH = "trainer/models/sock_tracker.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_blob(x_grid, y_grid, cx, cy, sigma=15):
    """Generates a Gaussian blob at (cx, cy)."""
    return np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))

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

def train(args):
    print(f"Downloading/Loading dataset from {args.dataset_id}...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    print(f"Dataset available at: {dataset_path}")

    dataset = SockDataset(dataset_path)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DobbyNet on Laundry Spots Dataset")
    
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID, 
                        help=f"HuggingFace Dataset Repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f"Path to save the trained model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train(args)