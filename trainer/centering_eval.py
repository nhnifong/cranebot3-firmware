import torch
import cv2
import os
import json
import numpy as np
import argparse
import random
from huggingface_hub import snapshot_download
from .centering import CenteringNet

# Configuration Defaults
DEFAULT_REPO_ID = "naavox/gripper-spots-dataset"
DEFAULT_MODEL_PATH = "trainer/models/sock_gripper.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset_samples(root_dir):
    """Loads metadata samples from the downloaded dataset."""
    data_dir = os.path.join(root_dir, "train")
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find metadata at {metadata_path}")
        
    samples = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("points") and len(data["points"]) > 0:
                    samples.append(data)
    return samples, data_dir

def get_vector_from_sample(sample, w, h):
    """Extracts the normalized vector (x, y) from a sample."""
    pt = sample["points"][0]
    if isinstance(pt, (list, tuple)):
        cx, cy = pt[0], pt[1]
    elif isinstance(pt, dict):
        cx = pt.get('x', 0)
        cy = pt.get('y', 0)
    else:
        cx, cy = w/2, h/2

    # Normalize to [-1, 1]
    norm_x = (cx - (w / 2)) / (w / 2)
    norm_y = (cy - (h / 2)) / (h / 2)
    
    return torch.tensor([norm_x, norm_y], dtype=torch.float32), (cx, cy)

def denormalize_coords(norm_x, norm_y, w, h):
    """Converts model output [-1, 1] back to pixel coordinates."""
    cx = (norm_x * (w / 2)) + (w / 2)
    cy = (norm_y * (h / 2)) + (h / 2)
    return int(cx), int(cy)

def visualize(args):
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    model = CenteringNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return
    model.eval()

    # 2. Load Dataset
    print(f"Checking dataset {args.dataset_id}...")
    dataset_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    samples, data_dir = load_dataset_samples(dataset_path)
    print(f"Loaded {len(samples)} samples.")

    print("\nControls:")
    print("  [SPACE] : Next Random Image")
    print("  [Q]     : Quit")

    while True:
        # 3. Pick Random Image
        sample = random.choice(samples)
        img_path = os.path.join(data_dir, sample["file_name"])
        
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 4. Prepare Input
        # Must match training preprocessing exactly
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension

        # 5. Get Ground Truth
        gt_vector, (gt_cx, gt_cy) = get_vector_from_sample(sample, w, h)

        # 6. Run Inference
        with torch.no_grad():
            pred_vector = model(img_tensor).cpu().squeeze()

        # 7. Visualization Logic
        pred_x, pred_y, pred_z = pred_vector[0].item(), pred_vector[1].item(), pred_vector[2].item()
        
        # Convert prediction to pixels
        pred_cx, pred_cy = denormalize_coords(pred_x, pred_y, w, h)
        
        center_x, center_y = w // 2, h // 2

        # Draw Center Crosshair (Gray)
        cv2.line(img, (center_x - 10, center_y), (center_x + 10, center_y), (100, 100, 100), 1)
        cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 10), (100, 100, 100), 1)

        # Draw Ground Truth (Green Arrow & Circle)
        cv2.arrowedLine(img, (center_x, center_y), (int(gt_cx), int(gt_cy)), (0, 255, 0), 2, tipLength=0.1)
        cv2.circle(img, (int(gt_cx), int(gt_cy)), 5, (0, 255, 0), -1)

        # Draw Prediction (Red Arrow & Circle)
        cv2.arrowedLine(img, (center_x, center_y), (pred_cx, pred_cy), (0, 0, 255), 2, tipLength=0.1)
        cv2.circle(img, (pred_cx, pred_cy), 5, (0, 0, 255), -1)

        # Text Overlay
        info_text = f"Pred: ({pred_x:.2f}, {pred_y:.2f}, {pred_z:.2f})"
        gt_text =   f"True: ({gt_vector[0]:.2f}, {gt_vector[1]:.2f}, {gt_vector[2]:.2f})"
        
        # Add background for text readability
        cv2.rectangle(img, (5, 5), (300, 60), (0, 0, 0), -1)
        cv2.putText(img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, gt_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("CenteringNet Prediction", img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CenteringNet Predictions")
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    
    args = parser.parse_args()
    visualize(args)