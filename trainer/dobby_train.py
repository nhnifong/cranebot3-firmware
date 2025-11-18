import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
import glob
from .dobby import DobbyNet

# --- Config ---
DATA_DIR = "dataset_heatmap"
MODEL_PATH = "trainer/models/sock_tracker.pth"
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_blob(x_grid, y_grid, cx, cy, sigma=15):
    return np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))

class SockDataset(Dataset):
    def __init__(self, root_dir):
        self.img_paths = glob.glob(os.path.join(root_dir, "images", "*.jpg"))
        self.label_dir = os.path.join(root_dir, "labels")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{file_id}.json")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        h, w = img.shape[:2]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        with open(label_path, 'r') as f:
            data = json.load(f)
        
        x_grid = np.arange(0, w, 1, float)
        y_grid = np.arange(0, h, 1, float)[:, np.newaxis]
        
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        points = data.get("points", [])
        if not points and "x" in data and "y" in data:
            points = [{"x": data["x"], "y": data["y"]}]
            
        for pt in points:
            blob = generate_blob(x_grid, y_grid, pt['x'], pt['y'])
            combined_heatmap = np.maximum(combined_heatmap, blob)
            
        heatmap_tensor = torch.from_numpy(combined_heatmap).float().unsqueeze(0) 
        return img_tensor, heatmap_tensor

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    dataset = SockDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DobbyNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- CHANGED: Loss Function ---
    # BCEWithLogitsLoss is better for pixel-wise classification than MSE
    # pos_weight can be used if class imbalance is still an issue, but usually this is enough.
    criterion = nn.BCEWithLogitsLoss() 

    print(f"Starting training on {len(dataset)} images...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, maps in dataloader:
            imgs, maps = imgs.to(DEVICE), maps.to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, maps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.5f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()