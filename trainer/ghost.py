import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. Load your dataset (just one camera to test)
print("Loading dataset...")
dataset = LeRobotDataset("naavox/merged-3")
# We just want to see if we can map 'observation.images.anchor_camera_0' -> 'action' (position)

def get_batch(idx):
    item = dataset[idx]
    # Get image (C, H, W) and normalize roughly to [0,1]
    img = item["observation.images.anchor_camera_0"] 
    # Get target (x, y, z position) - adjust indices for your data
    target = item["action"][:3] 
    return img, target

# 2. Setup tiny model (ResNet18 + Linear Head)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 3) # Predict X, Y, Z
model = model.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("Starting Vision Sanity Check (Training)...")
# 3. Train for just 500 steps
for i in range(500):
    # Random sampling
    idx_tensor = torch.randint(0, len(dataset), (32,))
    
    # Efficiently unpack the batch
    batch_imgs = []
    batch_targets = []
    for i_tensor in idx_tensor:
        # THE FIX: .item() converts tensor(42) -> 42
        img, target = get_batch(i_tensor.item()) 
        batch_imgs.append(img)
        batch_targets.append(target)

    # Stack the lists into a single batch
    imgs = torch.stack(batch_imgs).cuda()
    targets = torch.stack(batch_targets).cuda()

    preds = model(imgs)
    loss = criterion(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("Training finished.")

# ---------------------------------------------------
# 4. INFERENCE & VERIFICATION
# ---------------------------------------------------
print("\n--- Starting Verification ---")
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# Pick one random image
random_idx = random.randint(0, len(dataset) - 1)
print(f"Picking random frame index: {random_idx}")

# Get the image and its true label
with torch.no_grad():
    image, ground_truth_target = get_batch(random_idx)
    
    # Add a batch dimension (B, C, H, W) and send to GPU
    image_batch = image.unsqueeze(0).cuda()

    # Get the model's prediction
    prediction_tensor = model(image_batch)
    
    # Move prediction to CPU and remove batch dimension
    predicted_coords = prediction_tensor.squeeze().cpu().numpy()
    ground_truth_coords = ground_truth_target.numpy()

    print(f"\nModel Prediction: {predicted_coords}")
    print(f"Ground Truth:     {ground_truth_coords}")
    
    # Display the image
    # We need to convert (C, H, W) to (H, W, C) for matplotlib
    # and convert from [0, 1] tensor to [0, 255] uint8 numpy array
    image_to_show = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    plt.imshow(image_to_show)
    plt.title(f"Frame {random_idx}\nPred: {np.round(predicted_coords, 2)}\nTrue: {np.round(ground_truth_coords, 2)}")
    plt.axis('off')  # Hide axes
    print("\nShowing image... Close the plot window to exit.")
    plt.show()