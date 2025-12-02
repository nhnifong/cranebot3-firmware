import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import json
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
# Assuming lerobot is installed
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- CONFIGURATION ---
DATASET_LABELS_PATH = "dataset_heatmap/labels.jsonl"
LEROBOT_DATASET_ID = "naavox/merged-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAM_CALIBRATIONS = [
    { # Camera 0 (Anchor 0)
        "matrix": np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], dtype=float),
        "dist": np.zeros(5),
        "rvec": np.array([[-0.918], [-2.217], [1.332]], dtype=float),
        "tvec": np.array([[2.678], [2.601], [2.455]], dtype=float),
    },
    { # Camera 1 (Anchor 1) - Placeholder
        "matrix": np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], dtype=float),
        "dist": np.zeros(5),
        "rvec": np.array([[0], [0], [0]], dtype=float), 
        "tvec": np.array([[0], [0], [0]], dtype=float),
    },
    { # Camera 2 (Gripper) - Placeholder
        "matrix": np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], dtype=float),
        "dist": np.zeros(5),
        "rvec": np.array([[0], [0], [0]], dtype=float), 
        "tvec": np.array([[0], [0], [0]], dtype=float),
    }
]

class MultiViewDETR(nn.Module):
    def __init__(self, num_cameras=3, num_queries=10, hidden_dim=256, nheads=8, num_layers=4):
        super().__init__()
        
        # 1. Backbone (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # 2. Positional Embeddings (Learnable)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.cam_embed = nn.Embedding(num_cameras, hidden_dim)

        # 3. Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        # 4. Object Queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # 5. Prediction Heads
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
        features = features.view(B, Num_Cams, -1, 256)
        
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
        src = features.view(B, -1, 256)
        
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
    def __init__(self, jsonl_path, lerobot_repo_id):
        self.data = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        
        self.dataset = LeRobotDataset(lerobot_repo_id)
        self.cam_keys = [
            "observation.images.anchor_camera_0",
            "observation.images.anchor_camera_1",
            "observation.images.gripper_camera"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_entry = self.data[idx]
        frame_idx = int(label_entry["frame_number"])
        
        # Load Images
        frame_data = self.dataset[frame_idx]
        images = []
        for key in self.cam_keys:
            if key in frame_data:
                images.append(frame_data[key])
            else:
                images.append(torch.zeros(3, 360, 640))
        img_stack = torch.stack(images)
        
        # Load Objects (List of lists of sightings)
        # Structure: [Obj1_Sightings, Obj2_Sightings...]
        # Sighting: {'camera_index': int, 'x': float, 'y': float}
        objects = label_entry["objects"]
        
        return img_stack, objects

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch] # List of lists of objects
    return images, targets

def train(args):
    train_set = SockTriangulationDataset(DATASET_LABELS_PATH, LEROBOT_DATASET_ID)
    loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    model = MultiViewDETR(num_cameras=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    projector = WorldToCameraProjector(CAM_CALIBRATIONS, DEVICE)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        for batch_imgs, batch_targets in loader:
            batch_imgs = batch_imgs.to(DEVICE)
            pred_conf, pred_3d = model(batch_imgs) # (B, Q, 1), (B, Q, 3)
            
            loss = 0
            
            for b in range(len(batch_imgs)):
                gt_objects = batch_targets[b] # List of objects for this frame
                num_gt = len(gt_objects)
                
                # --- COST MATRIX CALCULATION ---
                # We need to match each GT Object to a Predicted Query based on reprojection error.
                
                # Predictions: Q
                # Ground Truths: N
                # Cost matrix: (N, Q)
                
                if num_gt > 0:
                    cost_matrix = torch.zeros((num_gt, pred_3d.shape[1]), device=DEVICE)
                    
                    for i, obj_sightings in enumerate(gt_objects):
                        # Calculate reprojection error for this object against ALL queries
                        obj_loss = torch.zeros(pred_3d.shape[1], device=DEVICE)
                        
                        for sighting in obj_sightings:
                            cam_idx = sighting['camera_index']
                            gt_uv = torch.tensor([sighting['x'], sighting['y']], device=DEVICE)
                            
                            # Project ALL queries to this camera view
                            proj_uv = projector.project(pred_3d[b].unsqueeze(0), cam_idx).squeeze(0) # (Q, 2)
                            
                            # L1 distance
                            dist = torch.abs(proj_uv - gt_uv).sum(dim=1)
                            obj_loss += dist
                            
                        cost_matrix[i] = obj_loss

                    # Hungarian Matching
                    cost_cpu = cost_matrix.detach().cpu().numpy()
                    row_idx, col_idx = linear_sum_assignment(cost_cpu)
                    
                    # --- LOSS CALCULATION ---
                    # 1. Reprojection Loss (Matched pairs)
                    loss += cost_matrix[row_idx, col_idx].mean()
                    
                    # 2. Objectness Loss
                    target_conf = torch.zeros_like(pred_conf[b])
                    target_conf[col_idx] = 1.0
                    loss += nn.functional.binary_cross_entropy(pred_conf[b], target_conf)
                else:
                    # No objects in frame -> All confidence should be 0
                    loss += nn.functional.binary_cross_entropy(pred_conf[b], torch.zeros_like(pred_conf[b]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"multiview_detr_ep{epoch+1}.pth")

# --- LABELING MODE ---

def label(args):
    print("Loading dataset for labeling...")
    dataset = LeRobotDataset(LEROBOT_DATASET_ID)
    
    # Load existing labels to check for duplicates
    labeled_frames = set()
    if os.path.exists(DATASET_LABELS_PATH):
        with open(DATASET_LABELS_PATH, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    labeled_frames.add(int(entry['frame_number']))
                except: pass
    
    print(f"Found {len(labeled_frames)} already labeled frames.")
    print("Controls:")
    print("  Left Click: Add sighting for current object")
    print("  'm':        New Object (Change Color)")
    print("  SPACE:      Save & Next Frame")
    print("  's':        Skip Frame")
    print("  'q':        Quit")

    cam_keys = [
        "observation.images.anchor_camera_0",
        "observation.images.anchor_camera_1",
        "observation.images.gripper_camera"
    ]

    while True:
        # Pick random frame not yet labeled
        idx = np.random.randint(0, len(dataset))
        if idx in labeled_frames:
            continue
            
        item = dataset[idx]
        
        # Prepare images for tiling
        imgs_np = []
        for key in cam_keys:
            if key in item:
                # (C,H,W) -> (H,W,C)
                img = item[key].permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
            imgs_np.append(img)
            
        # Tile images horizontally
        tiled_img = np.hstack(imgs_np)
        H, W, _ = imgs_np[0].shape
        
        # Labeling State
        current_objects = [] # List of lists of sightings
        active_obj_idx = 0
        current_objects.append([]) # Start with object 0
        
        # Color palette for objects
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

        def mouse_cb(event, x, y, flags, param):
            nonlocal active_obj_idx
            if event == cv2.EVENT_LBUTTONDOWN:
                # Determine which camera was clicked
                cam_idx = x // W
                local_x = x % W
                local_y = y
                
                if cam_idx < len(cam_keys):
                    # Add sighting
                    sighting = {'camera_index': cam_idx, 'x': local_x, 'y': local_y}
                    current_objects[active_obj_idx].append(sighting)

        cv2.namedWindow("Labeler")
        cv2.setMouseCallback("Labeler", mouse_cb)
        
        while True:
            display = tiled_img.copy()
            
            # Draw all objects
            for obj_i, sightings in enumerate(current_objects):
                color = colors[obj_i % len(colors)]
                for s in sightings:
                    cx = s['x'] + (s['camera_index'] * W)
                    cy = s['y']
                    cv2.circle(display, (cx, cy), 5, color, -1)
                    # Draw line connecting previous sighting of same object if exists
                    if len(sightings) > 1 and s != sightings[0]:
                         # Simple visual connector for feedback
                         pass

            # UI Text
            cv2.putText(display, f"Active Object: {active_obj_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[active_obj_idx % len(colors)], 2)
            
            cv2.imshow("Labeler", display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('m'):
                active_obj_idx += 1
                current_objects.append([])
                print(f"Switched to Object {active_obj_idx}")
            elif key == ord('s'):
                print("Skipped.")
                break
            elif key == ord(' '):
                # Save
                # Filter out empty objects
                final_objects = [obj for obj in current_objects if len(obj) > 0]
                
                if len(final_objects) > 0:
                    entry = {
                        "frame_number": str(idx),
                        "objects": final_objects
                    }
                    
                    with open(DATASET_LABELS_PATH, 'a') as f:
                        f.write(json.dumps(entry) + "\n")
                    
                    print(f"Saved frame {idx} with {len(final_objects)} objects.")
                    labeled_frames.add(idx)
                else:
                    print("No objects labeled, not saving.")
                break
        
        if key == ord('q'):
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "label"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "label":
        label(args)
    else:
        print("Eval mode not implemented yet.")

if __name__ == "__main__":
    main()