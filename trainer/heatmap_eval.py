import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import random
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_REPO_ID = "naavox/merged-5"
MODEL_PATH = "sock_tracker.pth"
CAMERA_KEY = "observation.images.anchor_camera_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- UPDATED Model Definition ---
class SimpleUNet(nn.Module):
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
        # We add Sigmoid here for inference/visualization
        return torch.sigmoid(self.final(d1))

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loading dataset {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(DATASET_REPO_ID)
    
    print("\n--- Controls ---")
    print("SPACE: Next random frame")
    print("Q:     Quit")

    while True:
        idx = random.randint(0, len(dataset) - 1)
        item = dataset[idx]
        
        if CAMERA_KEY not in item: continue
            
        img_tensor = item[CAMERA_KEY]
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

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap_np)
        
        if maxVal > 0.1: # Only draw if confidence is somewhat high
            box_size = 20
            top_left = (maxLoc[0] - box_size, maxLoc[1] - box_size)
            bottom_right = (maxLoc[0] + box_size, maxLoc[1] + box_size)
            cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)
            conf_text = f"Max Conf: {maxVal:.2f}"
            cv2.putText(overlay, conf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Sock Heatmap", overlay)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()