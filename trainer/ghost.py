import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import random
import cv2
import argparse
import os

# --- Define constants ---
MODEL_PATH = "ghost_model_2.pth"
BATCH_SIZE = 76
LEARNING_RATE = 1e-4
TRAIN_STEPS = 50000
LOG_FREQ = 100
STATE_DIM = 10
NUM_KEYPOINTS = 32 
ACTION_DIM = 5

print("Loading dataset...")
dataset = LeRobotDataset("naavox/merged-5")

def get_batch(idx):
    """Gets images, state vector, and target vector."""
    item = dataset[idx]
    img0 = item["observation.images.anchor_camera_0"] 
    img1 = item["observation.images.anchor_camera_1"] 
    img2 = item["observation.images.gripper_camera"] 
    state = item["observation.state"]
    target = item["action"][:ACTION_DIM] 
    return img0, img1, img2, state, target

# --- The Spatial Softmax Layer ---
class SpatialSoftmax(nn.Module):
    def __init__(self, num_rows, num_cols, num_kp):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_kp = num_kp
        self.temperature = nn.Parameter(torch.ones(1))

        # Create grid of coordinates (-1 to 1)
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, num_cols),
            np.linspace(-1.0, 1.0, num_rows)
        )
        pos_x = torch.from_numpy(pos_x.reshape(num_rows * num_cols)).float()
        pos_y = torch.from_numpy(pos_y.reshape(num_rows * num_cols)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature_map):
        # feature_map: (Batch, num_kp, H, W)
        batch_size, _, H, W = feature_map.shape
        
        # Flatten spatial dims
        feature_map = feature_map.view(batch_size, self.num_kp, -1)
        
        # Softmax to get attention map (probability distribution of "where is the feature")
        softmax_attention = F.softmax(feature_map / self.temperature, dim=2)
        
        # Weighted Average of coordinates -> Expected (X, Y)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2, keepdim=True)
        
        # Result: (Batch, num_kp * 2) -> [x1, y1, x2, y2, ...]
        expected_xy = torch.cat([expected_x, expected_y], dim=2)
        return expected_xy.view(batch_size, -1)

class MultiCameraResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Backbone: STOP at the conv layers (remove FC and AvgPool)
        # This keeps the 2D spatial map
        self.vision_backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Bottleneck: Reduce 512 channels to 32 Keypoints
        # This forces the network to distill "features" into "locations"
        self.bottleneck = nn.Conv2d(512, NUM_KEYPOINTS, kernel_size=1)
        
        # Spatial Softmax
        # We initialize with a guess for 360x640 images passed through ResNet (downsample 32x)
        # 360/32 ~ 12, 640/32 ~ 20
        self.pool = SpatialSoftmax(12, 20, NUM_KEYPOINTS)
        
        # State Encoder (Proprioception)
        self.state_encoder = nn.Sequential(
            nn.BatchNorm1d(STATE_DIM),
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # ead
        # Input: (32 kp * 2 coords * 3 cameras) + 64 state features
        visual_feat_dim = NUM_KEYPOINTS * 2 * 3 
        input_dim = visual_feat_dim + 64
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            # --- MODIFIED: Output 5 features ---
            nn.Linear(256, ACTION_DIM) # Predict X, Y, Z, Winch, Finger
        )

    def forward_one_camera(self, img):
        # Extract features (B, 512, H, W)
        feats = self.vision_backbone(img)
        
        # Compress to Keypoints (B, 32, H, W)
        feats = self.bottleneck(feats)
        
        # Handle dynamic image sizes if necessary
        B, C, H, W = feats.shape
        if self.pool.num_rows != H or self.pool.num_cols != W:
            self.pool = SpatialSoftmax(H, W, C).to(feats.device)
            
        # Extract coordinates (B, 64)
        keypoints = self.pool(feats)
        return keypoints

    def forward(self, img0, img1, img2, state):
        # Process Vision per camera
        kp0 = self.forward_one_camera(img0)
        kp1 = self.forward_one_camera(img1)
        kp2 = self.forward_one_camera(img2)
        visual_feats = torch.cat([kp0, kp1, kp2], dim=1) 
        
        # Process State
        state_feats = self.state_encoder(state)
        
        # Fuse and Decide
        all_features = torch.cat([visual_feats, state_feats], dim=1)
        
        prediction = self.head(all_features)
        return prediction

def run_training(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    train_transforms = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomApply(nn.ModuleList([T.GaussianBlur(kernel_size=3)]), p=0.2),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    ])

    print(f"Starting Training for {TRAIN_STEPS} steps...")
    
    for i in range(1, TRAIN_STEPS + 1):
        idx_tensor = torch.randint(0, len(dataset), (BATCH_SIZE,))
        
        batch_img0 = []
        batch_img1 = []
        batch_img2 = []
        batch_state = []
        batch_targets = []
        
        for i_tensor in idx_tensor:
            img0, img1, img2, state, target = get_batch(i_tensor.item()) 
            batch_img0.append(img0)
            batch_img1.append(img1)
            batch_img2.append(img2)
            batch_state.append(state)
            batch_targets.append(target)

        imgs0 = torch.stack(batch_img0).cuda()
        imgs1 = torch.stack(batch_img1).cuda()
        imgs2 = torch.stack(batch_img2).cuda()
        states = torch.stack(batch_state).cuda()
        targets = torch.stack(batch_targets).cuda()

        # --- APPLY AUGMENTATIONS ---
        # We apply the same random transform to all 3 images in the batch
        imgs0 = train_transforms(imgs0)
        imgs1 = train_transforms(imgs1)
        imgs2 = train_transforms(imgs2)
        
        preds = model(imgs0, imgs1, imgs2, states)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % LOG_FREQ == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {i}/{TRAIN_STEPS}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
            
        if i == TRAIN_STEPS // 2:
             torch.save(model.state_dict(), "ghost_model_halfway.pth")

    print("Training finished.")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved to {MODEL_PATH}")

def run_evaluation(model):
    print("\n--- Starting Verification (Loop) ---")
    print("A 'verification_frame.jpg' will be saved for each frame.")
    print("Press ENTER to see the next random frame.")
    print("Press CTRL+C to stop.")
    
    model.eval()
    
    try:
        while True:
            random_idx = random.randint(0, len(dataset) - 1)
            
            with torch.no_grad():
                img0, img1, img2, state, ground_truth_target = get_batch(random_idx)
                
                img0_batch = img0.unsqueeze(0).cuda()
                img1_batch = img1.unsqueeze(0).cuda()
                img2_batch = img2.unsqueeze(0).cuda()
                state_batch = state.unsqueeze(0).cuda()

                prediction_tensor = model(img0_batch, img1_batch, img2_batch, state_batch)
                
                predicted_coords = prediction_tensor.squeeze().cpu().numpy()
                ground_truth_coords = ground_truth_target.numpy()

                print(f"\n--- Frame Index: {random_idx} ---")
                # --- MODIFIED: Print all 5 predicted features ---
                print("       [Gantry_X, Gantry_Y, Gantry_Z, Winch, Finger]")
                print(f"Pred:  {np.round(predicted_coords, 3)}")
                print(f"True:  {np.round(ground_truth_coords, 3)}")
                
                # Permute (C,H,W) to (H,W,C) for OpenCV
                image_to_show_rgb = (img0.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_to_show_bgr = cv2.cvtColor(image_to_show_rgb, cv2.COLOR_RGB2BGR)

                # --- MODIFIED: Simplified text for image ---
                pred_text = f"Pred Gantry: {np.round(predicted_coords[:3], 2)}"
                true_text = f"True Gantry: {np.round(ground_truth_coords[:3], 2)}"
                
                cv2.putText(image_to_show_bgr, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
                cv2.putText(image_to_show_bgr, true_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

                cv2.imwrite("verification_frame.jpg", image_to_show_bgr)
                print("Saved verification_frame.jpg. Check your file browser.")
                
                input("Press Enter for next frame, or Ctrl+C to exit...")
                    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting.")

def main():
    parser = argparse.ArgumentParser(description="Train and/or evaluate the multi-camera vision model.")
    parser.add_argument("--train", action="store_true", help="Run the training loop and save the model.")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation loop (loads model if not training).")
    args = parser.parse_args()

    if not args.train and not args.eval:
        print("Neither --train nor --eval specified. Exiting.")
        print("Use --train to train, --eval to evaluate, or both.")
        return

    model = MultiCameraResNet().cuda()

    if args.train:
        run_training(model)

    if args.eval:
        if not args.train:
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH))
                print(f"Loaded trained weights from {MODEL_PATH}")
            else:
                print(f"Warning: --eval set but no {MODEL_PATH} found. Running with untrained model.")
        
        run_evaluation(model)

if __name__ == "__main__":
    main()