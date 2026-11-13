import argparse
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial import distance
from skimage import measure, transform
from skimage.feature import hog
from sklearn.preprocessing import normalize


# ======================================
# Step 1: Load + preprocess images
# ======================================
class ImagePreProcessing(torch.utils.data.Dataset):
    def __init__(self, image_folder, image_size=(379, 379), background_threshold=10):
        self.image_folder = image_folder
        self.images = sorted(
            [f for f in os.listdir(image_folder)
             if f.endswith(".png") and not f.startswith('.')]
        )
        self.image_size = image_size
        self.background_threshold = background_threshold

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, dtype=np.float32)

        img_np[img_np < self.background_threshold] = 0

        binary = img_np > 0
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)

        if props:
            angle = props[0].orientation
            img_np = transform.rotate(
                img_np,
                angle * 180 / np.pi,
                resize=False,
                preserve_range=True
            )

        if img_np.max() > img_np.min():
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        img = Image.fromarray((img_np * 255).astype(np.uint8))
        img = img.resize(self.image_size)

        return img

    def __getitem__(self, index):
        image_file = self.images[index]
        image_path = os.path.join(self.image_folder, image_file)
        return self.preprocess_image(image_path)

    def __len__(self):
        return len(self.images)


# ======================================
# PATH SETUP (relative to the repo root, so they work on any machine)
# ======================================
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="HOG features, hierarchical clustering and similarity network "
                "for the cropped cross-section (top-left panel) images.")
parser.add_argument("--input-dir", type=Path,
                    default=REPO_ROOT / "output" / "cropped_panels",
                    help="Folder of cropped PNGs (default: output/cropped_panels)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Sequence Variant, Polymorph "
                         "(default: data/Labels.csv)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "hog",
                    help="Where to save all outputs (default: output/hog)")
args = parser.parse_args()

hog_dir = args.output_dir
hog_dir.mkdir(parents=True, exist_ok=True)


# ======================================
# Load images
# ======================================
image_set = ImagePreProcessing(args.input_dir)
n = len(image_set)

print(f"Number of images loaded: {n}")

names = []

# ======================================
# Extract EMD numbers
# ======================================
emd_nums = []

for filename in image_set.images:
    match = re.search(r'(\d+)', filename)
    emd_number = match.group(1) if match else "unknown"
    emd_nums.append(emd_number)


# ======================================
# Step 2: Extract HOG features
# ======================================
fd_list = []

rows = n
columns = 2

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(8, 2 * n),
    constrained_layout=True
)

for i in range(n):
    image = image_set[i]
    image_np = np.array(image)

    fd, hog_image = hog(
        image_np,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True
    )

    fd_list.append(fd)

    axes[i, 0].imshow(image_np, cmap='gray')
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"{emd_nums[i]}")

    axes[i, 1].imshow(hog_image, cmap='gray')
    axes[i, 1].axis("off")

plt.savefig(hog_dir / "hog_feature_visualization.png", dpi=300)
plt.show()


# ======================================
# Step 3: Normalize HOG vectors
# ======================================
norm_fd_array = normalize(fd_list, norm='l2')


# ======================================
# Step 4: Distance matrices
# ======================================
condensed_distance_matrix = distance.pdist(
    norm_fd_array,
    metric="cosine"
)

square_matrix = distance.squareform(condensed_distance_matrix)

pd.DataFrame(square_matrix).to_csv(hog_dir / "normalized_distance_matrix.csv")
pd.DataFrame(condensed_distance_matrix).to_csv(hog_dir / "condensed_matrix.csv")


np.save(
    hog_dir / "square_matrix.csv",
square_matrix
),

np.save(
    hog_dir / "condensed_matrix.csv",
condensed_distance_matrix
),


# ======================================
# Step 5: Hierarchical clustering
# ======================================
linkage_matrix = linkage(
    condensed_distance_matrix,
    method='average'
)
np.save(
    hog_dir / "hog_linkage_matrix.npy",
    linkage_matrix
)

labels_list = emd_nums


# =========================
# LOAD METADATA
# =========================
data = pd.read_csv(args.labels_csv)

data.columns = (
    data.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

data["EMDB_entry"] = data["EMDB_entry"].astype(str)

data = data.set_index("EMDB_entry")
data = data.loc[emd_nums]


# =========================
# BUILD LABELS
# =========================
labels = []
for i in range(len(data)):
    emdb = data.index[i]
    variant = str(data["Sequence_Variant"].iloc[i])

    if variant.upper() == "WT":
        labels.append(emdb)
    else:
        labels.append(f"{emdb} ({variant})")


# =========================
# POLYMORPH COLOUR MAP
# =========================
polymorphs = data["Polymorph"].astype(str).values
unique_polys = np.unique(polymorphs)

import matplotlib as mpl
cmap = mpl.colormaps.get_cmap("turbo").resampled(len(unique_polys))

poly_color_map = {
    poly: cmap(i)
    for i, poly in enumerate(unique_polys)
}


# =========================
# PLOT DENDROGRAM
# =========================
plt.figure(figsize=(14, 8))
plt.subplots_adjust(right=0.80)
plt.subplots_adjust(bottom=0.2)

dend = dendrogram(
    linkage_matrix,
    labels=labels,
    leaf_rotation=75,
    leaf_font_size=10
)


# =========================
# 🔥 NEW: LEAF ORDER EXTRACTION
# =========================
leaf_order = dend['leaves']
ordered_ids = [emd_nums[i] for i in leaf_order]


# =========================
# COLOUR LABELS
# =========================
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()

for i, lbl in enumerate(xlbls):
    idx = leaf_order[i]
    poly = polymorphs[idx]
    lbl.set_color(poly_color_map[poly])


# =========================
# LEGEND
# =========================
import matplotlib.patches as mpatches

legend_handles = [
    mpatches.Patch(color=poly_color_map[p], label=p)
    for p in unique_polys
]

plt.legend(
    handles=legend_handles,
    title="Polymorph",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)


# ===============================
# Step 5b: Extract cluster labels
# ===============================
n_clusters = 4

cluster_labels = fcluster(
    linkage_matrix,
    t=n_clusters,
    criterion='maxclust'
)


# ===============================
# NEW: SAVE ORDER + CLUSTERS
# ===============================

cluster_df = pd.DataFrame({
    "EMDB_entry": emd_nums,
    "image_cluster": cluster_labels
})

# Labels in the same order as the rows of the linkage matrix (used by 13a_collage_combiner_cryoem.py)
cluster_df.to_csv(hog_dir / "hog_cluster_labels.csv", index=False)

order_df = pd.DataFrame({
    "order": range(len(ordered_ids)),
    "EMDB_entry": ordered_ids
})

final_df = order_df.merge(cluster_df, on="EMDB_entry")

final_df.to_csv(
    hog_dir / "hog_final_layout.csv",
    index=False
)

print("\nSaved final layout:")
print(final_df.head())

import pandas as pd

leaf_order = leaves_list(linkage_matrix)

ordered_emdb = [emd_nums[i] for i in leaf_order]

order_df = pd.DataFrame({
    "EMDB_entry": ordered_emdb,
    "order": np.arange(len(ordered_emdb))
})

order_df.to_csv(
    hog_dir / "hog_leaf_order.csv",
    index=False
)

plt.title("HOG Dendrogram")
plt.xlabel("Cryo-EM Cross Section")
plt.ylabel("Cosine Distance")

plt.savefig(hog_dir / "hog_dendrogram.png", dpi=300)
plt.show()


# ======================================
# Step 6: Network similarity graph
# ======================================
threshold = 0.5

labels = {i: emd_nums[i] for i in range(n)}

G = nx.Graph()
G.add_nodes_from(range(n))

for i in range(len(square_matrix)):
    for j in range(i + 1, len(square_matrix)):
        if square_matrix[i][j] < threshold:
            G.add_edge(i, j, weight=square_matrix[i][j])

fig_network, ax = plt.subplots(figsize=(8, 8))

pos = nx.spring_layout(G, k=1.1, seed=42)

nx.draw(
    G,
    pos,
    labels=labels,
    with_labels=True,
    node_size=600,
    font_size=7
)

ax.set_title("HOG Similarity Network")

fig_network.savefig(
    hog_dir / "hog_network.png",
    dpi=300
)

plt.show()