import cv2
import os
import uuid
import json
import numpy as np
import random
import shutil
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi, create_repo
from scipy.spatial.transform import Rotation
from cv_common import stabilize_frame, invert_pose, compose_poses
import model_constants

# Configuration
SOURCE_DATASET_ID = "naavox/merged-5"
TARGET_HF_REPO = "naavox/gripper-spots-dataset"  # Change this to your desired target repo name
LOCAL_DATASET_ROOT = "gripper_spots_data"        # Local folder name
CAMERA_KEY = "observation.images.gripper_camera"

# Paths for the structure: root -> train -> images + metadata.jsonl
TRAIN_DIR = os.path.join(LOCAL_DATASET_ROOT, "train")
METADATA_PATH = os.path.join(TRAIN_DIR, "metadata.jsonl")

# Ensure directories exist
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

# Create a basic README if it doesn't exist (Required for HF to recognize the structure)
readme_path = os.path.join(LOCAL_DATASET_ROOT, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write("    path: train/metadata.jsonl\n")
        f.write("---\n\n")
        f.write(f"# Laundry Spots Dataset\n\nGenerated from {SOURCE_DATASET_ID}.")

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

    cv2.imshow("Collector", display)

def main():
    print(f"Loading source dataset {SOURCE_DATASET_ID}...")
    dataset = LeRobotDataset(SOURCE_DATASET_ID)
    
    cv2.namedWindow("Collector")
    cv2.setMouseCallback("Collector", mouse_callback)

    print("\n--- Instructions ---")
    print("1. Left-Click:   Mark a spot (x, y).")
    print("2. Right-Click:  Undo last mark.")
    print("3. SPACE:        Save to local JSONL and go to next.")
    print("4. 'n':          Skip to next random frame.")
    print("5. 'q':          Quit and ask to upload to Hugging Face.")

    global current_clicks, current_image
    
    while True:
        # Pick a random frame
        idx = random.randint(0, len(dataset) - 1)
        item = dataset[idx]
        
        if CAMERA_KEY not in item:
            continue
            
        state = item['observation.state']
        pad_voltage = state[9]
        if pad_voltage > 0.4:
            continue # don't want to use images where a sock is being held.

        # Process Image (LeRobot Tensor -> OpenCV BGR)
        img_tensor = item[CAMERA_KEY] 
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # apply the iamge stabilization using the IMU value from the moment the image was collected
        gripper_rvec = state[5:8]
        imu_rvec = compose_poses([
            (gripper_rvec.numpy(), np.zeros(3)),
            invert_pose(model_constants.gripper_imu)
        ])[0]

        gripper_quat = Rotation.from_rotvec(imu_rvec).as_quat()
        img_np = stabilize_frame(img_np, gripper_quat)
        
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np * 255).astype(np.uint8)
        
        current_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        current_clicks = [] 
        
        draw_interface()

        # Interaction Loop
        break_outer = False
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting collection...")
                break_outer = True
                break 
                
            elif key == ord('n'):
                print("Skipped frame.")
                break 
                
            elif key == ord(' '):
                if current_clicks:
                    # Generate filename
                    file_id = str(uuid.uuid4())
                    img_filename = f"{file_id}.jpg"
                    img_full_path = os.path.join(TRAIN_DIR, img_filename)

                    # Save Image to train folder
                    cv2.imwrite(img_full_path, current_image)
                    
                    # Append to metadata.jsonl
                    # Format: {"file_name": "1.jpg", "points": [[x1,y1], [x2,y2]]}
                    metadata_entry = {
                        "file_name": img_filename,
                        "points": current_clicks
                    }
                    
                    with open(METADATA_PATH, 'a') as f:
                        f.write(json.dumps(metadata_entry) + "\n")
                    
                    print(f"Saved {len(current_clicks)} points -> {img_filename}")
                    break 
                else:
                    print("No points selected! Press 'n' to skip.")
                    
        if break_outer:
            break

    cv2.destroyAllWindows()
    
    # Upload Sequence
    if os.path.exists(METADATA_PATH):
        print("\n" + "="*30)
        print(f"Collection finished. Data stored in '{LOCAL_DATASET_ROOT}'")
        print(f"Target Repo: {TARGET_HF_REPO}")
        confirm = input("Do you want to upload this dataset to Hugging Face now? (y/n): ").strip().lower()
        
        if confirm == 'y':
            print("Authenticating and uploading...")
            api = HfApi()
            
            # Create repo if it doesn't exist
            create_repo(TARGET_HF_REPO, repo_type="dataset", exist_ok=True)
            
            # Upload the folder
            api.upload_folder(
                folder_path=LOCAL_DATASET_ROOT,
                repo_id=TARGET_HF_REPO,
                repo_type="dataset"
            )
            print(f"Successfully uploaded to https://huggingface.co/datasets/{TARGET_HF_REPO}")
        else:
            print("Skipping upload. You can upload later using the command below:")
            print(f"huggingface-cli upload {TARGET_HF_REPO} {LOCAL_DATASET_ROOT} --repo-type dataset")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()