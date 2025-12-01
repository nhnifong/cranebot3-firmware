import torch
import cv2
import numpy as np
import os
import random
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .dobby import DobbyNet
import argparse

DATASET_REPO_ID = "naavox/merged-5"
MODEL_PATH = "trainer/models/sock_tracker.pth"
CAMERA_KEY = "observation.images.anchor_camera_0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_RES = (640, 360)
MINIMUM_CONFIDENCE = 0.95

import cv2
import numpy as np

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

def main(uri):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = DobbyNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    if uri :
        print('connecting to stream')
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            exit()
    else:
        print(f"Loading dataset {DATASET_REPO_ID}...")
        dataset = LeRobotDataset(DATASET_REPO_ID)
    
    print("\n--- Controls ---")
    print("SPACE: Next random frame")
    print("Q:     Quit")

    # Initialize window with NORMAL flag to allow resizing
    window_name = "Sock Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size to double the processing resolution (1280x720)
    cv2.resizeWindow(window_name, IMAGE_RES[0] * 2, IMAGE_RES[1] * 2)

    while True:
        if uri:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, IMAGE_RES, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        else:
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
        # img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
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

        cv2.imshow("Sock Heatmap", overlay)
        
        key = cv2.waitKey(1 if uri else 0) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", help="When specified, connects to the given stream")
    args = parser.parse_args()

    main(uri=args.uri)