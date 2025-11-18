import cv2
import os
import uuid
import json
import numpy as np
import random
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
DATASET_REPO_ID = "naavox/merged-5"
SAVE_DIR = "dataset_heatmap"
# We default to the first anchor camera. Change to "observation.images.anchor_camera_1" 
# if you want to label the other view.
CAMERA_KEY = "observation.images.anchor_camera_1" 

os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

# State
current_clicks = []
current_image = None

def mouse_callback(event, x, y, flags, param):
    global current_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point
        current_clicks.append((x, y))
        print(f"Added Point: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last point
        if current_clicks:
            removed = current_clicks.pop()
            print(f"Removed Point: {removed}")

def main():
    print(f"Loading dataset {DATASET_REPO_ID}...")
    # We need videos downloaded to extract frames
    dataset = LeRobotDataset(DATASET_REPO_ID)
    
    cv2.namedWindow("Collector")
    cv2.setMouseCallback("Collector", mouse_callback)

    print("\n--- Instructions ---")
    print("1. Left-Click:   Mark a sock center.")
    print("2. Right-Click:  Undo last mark.")
    print("3. SPACE:        Save image + labels and go to next random frame.")
    print("4. 'n':          Skip to next random frame (don't save).")
    print("5. 'q':          Quit.")

    global current_clicks, current_image
    
    while True:
        # 1. Pick a random frame from the dataset
        idx = random.randint(0, len(dataset) - 1)
        item = dataset[idx]
        
        if CAMERA_KEY not in item:
            print(f"Warning: Key {CAMERA_KEY} not found in frame {idx}. Skipping.")
            continue
            
        # 2. Process Image
        # LeRobot returns (C, H, W) float tensor [0..1]
        img_tensor = item[CAMERA_KEY] 
        
        # Convert to numpy (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Scale to 0-255 uint8 if needed
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert RGB (LeRobot) to BGR (OpenCV)
        current_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        current_clicks = [] # Reset labels for new image
        
        # 3. Interaction Loop (Stay on this image until action taken)
        while True:
            display = current_image.copy()

            # Draw all points
            for i, pt in enumerate(current_clicks):
                # Draw crosshair/circle
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                # Optional: Number them
                # cv2.putText(display, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # HUD
            status_text = f"Points: {len(current_clicks)}"
            cv2.putText(display, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "[Space] Save | [N] Skip | [Q] Quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                cv2.destroyAllWindows()
                return # Exit program
                
            elif key == ord('n'):
                print("Skipped frame.")
                break # Break inner loop -> load new random frame
                
            elif key == ord(' '):
                if current_clicks:
                    # Generate filenames
                    file_id = str(uuid.uuid4())
                    img_filename = f"{file_id}.jpg"
                    label_filename = f"{file_id}.json"
                    
                    img_path = os.path.join(SAVE_DIR, "images", img_filename)
                    label_path = os.path.join(SAVE_DIR, "labels", label_filename)

                    # Save Clean Image (without markers)
                    cv2.imwrite(img_path, current_image)
                    
                    # Save Labels
                    # We save a list of points: [{"x": 100, "y": 200}, ...]
                    label_data = {
                        "points": [{"x": pt[0], "y": pt[1]} for pt in current_clicks],
                        "width": current_image.shape[1],
                        "height": current_image.shape[0],
                        "source_dataset": DATASET_REPO_ID,
                        "source_frame_idx": idx
                    }
                    
                    with open(label_path, 'w') as f:
                        json.dump(label_data, f, indent=2)
                    
                    print(f"Saved {len(current_clicks)} targets -> {file_id}")
                    break # Break inner loop -> load new random frame
                else:
                    print("No points selected! Press 'n' to skip if there are no socks.")

if __name__ == "__main__":
    main()