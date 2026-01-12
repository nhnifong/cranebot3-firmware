import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import cv2
import os

class UnsupervisedSorter:
    def __init__(self, device="cuda"):
        self.device = device
        
        # 1. The Embedder
        # We use ResNet18 as a generic feature extractor.
        # It's better than DobbySorter because it has seen the whole world (ImageNet),
        # so it knows about textures/shapes it hasn't explicitly trained on yet.
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base_model = models.resnet18(weights=weights)
        
        # Remove the final classification layer (fc)
        # Output shape: (Batch, 512, 1, 1) -> Flatten -> (Batch, 512)
        self.embedder = nn.Sequential(*list(base_model.children())[:-1])
        self.embedder.to(device)
        self.embedder.eval()
        
        # Preprocessing for ResNet (ImageNet stats)
        self.preprocess = weights.transforms()

        # Storage
        self.crop_registry = [] # List of raw crop images (for display)
        self.embeddings = []    # List of vectors
        self.labels = []        # Cluster IDs

    def add_crop(self, crop_tensor):
        """
        Ingest a 64x64 crop tensor (C, H, W).
        Computes embedding immediately but doesn't cluster yet.
        """
        # Store raw image for UI (convert to numpy uint8 HWC)
        img_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        self.crop_registry.append(img_np)

        # Compute Embedding
        # ResNet expects batches, so unsqueeze
        batch = crop_tensor.unsqueeze(0).to(self.device)
        
        # Apply ImageNet normalization (critical for pre-trained weights)
        # Note: crop_tensor should be 0-1 float
        batch = self.preprocess(batch)

        with torch.no_grad():
            # (1, 512, 1, 1) -> (1, 512)
            embedding = self.embedder(batch).squeeze().cpu().numpy()
            
        self.embeddings.append(embedding)

    def cluster(self, distance_threshold=15.0):
        """
        Groups the crops.
        distance_threshold: How 'different' objects must be to split into groups.
                            Lower = More small, specific groups.
                            Higher = Fewer, broad groups.
        """
        if not self.embeddings:
            return {}

        X = np.array(self.embeddings)
        
        # Optional: PCA to reduce noise before clustering
        # Reducing to ~50 dims often helps clustering distance metrics work better
        if X.shape[0] > 50:
            pca = PCA(n_components=50)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X

        # We use Agglomerative Clustering because we can set a distance threshold
        # instead of picking 'K' clusters.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='euclidean',
            linkage='ward' 
        )
        
        self.labels = clustering.fit_predict(X_reduced)
        
        # Organize results
        groups = {}
        for idx, label in enumerate(self.labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)
            
        print(f"Found {len(groups)} unique groups from {len(X)} items.")
        return groups

    def get_group_representative(self, group_indices):
        """
        Finds the 'center' image of the group to show the user.
        """
        # Get embeddings for this group
        group_vecs = np.array([self.embeddings[i] for i in group_indices])
        
        # Calculate mean vector
        mean_vec = np.mean(group_vecs, axis=0)
        
        # Find index of item closest to mean
        distances = np.linalg.norm(group_vecs - mean_vec, axis=1)
        center_idx_local = np.argmin(distances)
        center_idx_global = group_indices[center_idx_local]
        
        return self.crop_registry[center_idx_global]

# --- Example Usage Loop ---
if __name__ == "__main__":
    sorter = UnsupervisedSorter()
    
    # Simulate loading random crops
    print("Simulating data ingestion...")
    # In reality, you'd feed crops from your robot here
    fake_crop = torch.rand(3, 64, 64) 
    for _ in range(20):
        sorter.add_crop(fake_crop) # Just adding noise for demo
        
    # Cluster them
    groups = sorter.cluster()
    
    # UI Logic
    for group_id, indices in groups.items():
        print(f"--- Group {group_id} ({len(indices)} items) ---")
        
        # Show the user the representative image
        rep_image = sorter.get_group_representative(indices)
        
        # In your real code, you would cv2.imshow this image
        # cv2.imshow("Is this a sock?", cv2.cvtColor(rep_image, cv2.COLOR_RGB2BGR))
        # key = cv2.waitKey(0)
        
        print("User labels this group: 'Sock'")
        # Store logic: "If embedding matches Group X center, execute Sock Logic"