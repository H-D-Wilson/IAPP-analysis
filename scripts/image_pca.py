import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import transform, measure




# ===========================
# Improved Image Dataset Class
# ===========================
class ImageDataset:
    def __init__(self, image_folder, image_size=(128,128), background_threshold=10):
        self.image_folder = image_folder
        self.images = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
        self.image_size = image_size
        self.background_threshold = background_threshold
        self.transformation = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def preprocess_image(self, image_path):
        # Open image and convert to grayscale
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, dtype=np.float32)

        # --------------------------
        # 1. Background removal
        # --------------------------
        img_np[img_np < self.background_threshold] = 0

        # --------------------------
        # 2. Orientation alignment
        # --------------------------
        binary = img_np > 0
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        if props:
            angle = props[0].orientation  # radians
            img_np = transform.rotate(img_np, angle * 180 / np.pi, resize=False, preserve_range=True)

        # --------------------------
        # 3. Intensity normalization
        # --------------------------
        if img_np.max() > img_np.min():
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = np.zeros_like(img_np)

        # --------------------------
        # 4. Optional histogram equalization (comment in if desired)
        # --------------------------
        # img_np = (ImageOps.equalize(Image.fromarray((img_np*255).astype(np.uint8)))).astype(np.float32)/255.0

        # Convert back to PIL Image
        img_processed = Image.fromarray((img_np*255).astype(np.uint8))

        # Apply resize and ToTensor transformation
        tensor = self.transformation(img_processed)
        return tensor

    def get_flattened_image_data(self):
        tensors = []
        for image_file in self.images:
            image_path = os.path.join(self.image_folder, image_file)
            tensor = self.preprocess_image(image_path)
            tensors.append(tensor)
        images = torch.stack(tensors)
        return images.view(images.size(0), -1)

# ===========================
# Load images
# ===========================
# Paths are relative to the repo root, so they work on any machine
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="PCA, clustering and scatter plot of the cropped cross-section (top-left panel) images.")
parser.add_argument("--input-dir", type=Path,
                    default=REPO_ROOT / "output" / "cropped_panels",
                    help="Folder of cropped PNGs (default: output/cropped_panels)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry and Sequence Variant, used to label the scatter "
                         "points (default: data/Labels.csv)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "pca",
                    help="Where to save the scatter plot and cluster labels (default: output/pca)")
args = parser.parse_args()

image_folder = args.input_dir
output_folder = args.output_dir  # folder to save figures
os.makedirs(output_folder, exist_ok=True)

loader = ImageDataset(image_folder)
X = loader.get_flattened_image_data()
print(f"Shape of flattened image data: {X.shape}")  # n_samples x n_features

# ===========================
# PCA
# ===========================
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X.numpy())
print(f"Shape of PCA-transformed data: {X_pca.shape}")
# ===============================
# Extract PCA labels
# ===============================
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize

# Labels (must match image order)
labels = [f.replace(".png", "") for f in loader.images]

# Clean PCA data
X_pca = np.asarray(X_pca)
X_pca = np.nan_to_num(X_pca)

# Normalize (recommended for clustering)
X_pca_norm = normalize(X_pca)

# Build linkage matrix
Z = linkage(X_pca_norm, method='average')

# Extract clusters
pca_labels = fcluster(Z, t=4, criterion='maxclust')

# Save results
df_pca = pd.DataFrame({
    "EMDB_entry": labels,
    "image_pca_cluster": pca_labels
})

df_pca.to_csv(output_folder / "pca_cluster_labels.csv", index=False)

print("\nSaved PCA cluster labels")

# ===============================
#Extract PCA labels
# ===============================
from scipy.cluster.hierarchy import fcluster

Z = linkage(normalize(X_pca), method='average')
pca_labels = fcluster(Z, t=4, criterion='maxclust')

df_pca = pd.DataFrame({
    "EMDB_entry": np.array(labels),
    "image_pca_cluster": np.array(pca_labels)
})

label_values_dir = output_folder / "Label values"
label_values_dir.mkdir(parents=True, exist_ok=True)
df_pca.to_csv(label_values_dir / "pca_cluster_labels.csv", index=False)


print("\nSaved pca_cluster labels:")

# ---------------------------
# Point labels from Labels.csv (EMDB entry, plus the sequence variant if not WT)
# ---------------------------
data = pd.read_csv(args.labels_csv)

data.columns = (
    data.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

data["EMDB_entry"] = data["EMDB_entry"].astype(str)
data = data.set_index("EMDB_entry")

image_ids = [re.search(r"(\d+)", name).group(1) for name in labels]
data = data.loc[image_ids]   # match by EMDB_entry, in the same order as the PCA rows

labels = []
for i in range(len(data)):
    emdb = data.index[i]
    variant = str(data["Sequence_Variant"].iloc[i])

    if variant.upper() == "WT":
        labels.append(emdb)
    else:
        labels.append(f"{emdb} ({variant})")


# ---------------------------
# PC1 vs PC2 scatter plot
# ---------------------------
plt.figure(figsize=(8, 7))

# Scatter with cluster colouring
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pca_labels, cmap="tab10", s=50)

# Add labels to each point
for i, name in enumerate(labels):
    plt.text(
        X_pca[i, 0],
        X_pca[i, 1],
        name,
        fontsize=6,
        alpha=0.8
    )

# Axis labels (with variance explained)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)

# Title
plt.title("PCA of IAPP Cryo-EM cross-sections (First Two Components)", fontsize=13)

# Gridlines (clean + readable)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# Optional: add axis lines at origin (nice visual reference)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

# Tight layout
plt.tight_layout()

plt.savefig(os.path.join(output_folder, 'pca_scatter.png'), dpi=300)
plt.show()
