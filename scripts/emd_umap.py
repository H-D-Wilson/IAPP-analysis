import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import umap

# =========================
# SETTINGS
# =========================
# Paths are relative to the repo root, so they work on any machine
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="UMAP and clustering of the 3D density maps from the Sinkhorn distance matrix.")
parser.add_argument("--distance-matrix", type=Path,
                    default=REPO_ROOT / "output" / "emd_clustering" / "sinkhorn_distance_matrix.npy",
                    help="Sinkhorn distance matrix from 10_emd_density_clustering.py "
                         "(default: output/emd_clustering/sinkhorn_distance_matrix.npy)")
parser.add_argument("--resampled-dir", type=Path,
                    default=REPO_ROOT / "data" / "density_64",
                    help="Folder of resampled <EMDB_entry>.npy volumes, used for the names when the raw "
                         "maps are not available (default: data/density_64)")
parser.add_argument("--matrix-order-csv", type=Path,
                    default=REPO_ROOT / "output" / "emd_clustering" / "sinkhorn_matrix_order.csv",
                    help="Row order of the distance matrix, written by 10_emd_density_clustering.py; "
                         "used to check the names below (default: output/emd_clustering/sinkhorn_matrix_order.csv)")
parser.add_argument("--density-dir", type=Path,
                    default=REPO_ROOT / "data" / "RAW density",
                    help="Folder of <EMDB_entry>.mat files; only the filenames are used, to name "
                         "the matrix rows (default: data/RAW density)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Sequence Variant, Polymorph (default: data/Labels.csv)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "emd_umap",
                    help="Where to save outputs (default: output/emd_umap)")
args = parser.parse_args()

distance_matrix_path = args.distance_matrix

labels_csv_path = args.labels_csv

save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

n_neighbors = 10
min_dist = 0.1
n_clusters = 4

# =========================
# LOAD DISTANCE MATRIX
# =========================
D = np.load(distance_matrix_path)

print("Loaded distance matrix:", D.shape)

# =========================
# LOAD NAMES (IMPORTANT)
# =========================
# Must match order used when computing D
raw_files = (
    sorted([f for f in os.listdir(args.density_dir) if f.endswith(".mat")])
    if os.path.isdir(args.density_dir) else []
)
if raw_files:
    names = [f.replace(".mat", "") for f in raw_files]
else:
    # Raw maps not available: use the resampled volumes, which have the same names
    resampled_files = (
        sorted([f for f in os.listdir(args.resampled_dir) if f.endswith(".npy")])
        if os.path.isdir(args.resampled_dir) else []
    )
    if not resampled_files:
        raise SystemExit(f"No .mat files in {args.density_dir} and no .npy files in {args.resampled_dir}")
    names = [f.replace(".npy", "") for f in resampled_files]

# Check the names against the matrix order saved by script 10 (a mismatch would mislabel the points)
if args.matrix_order_csv.exists():
    saved_names = pd.read_csv(args.matrix_order_csv, dtype=str)["EMDB_entry"].tolist()
    if saved_names != names:
        raise SystemExit(
            f"The structure names do not match the row order of the distance matrix in "
            f"{args.matrix_order_csv.name}; rerun 10_emd_density_clustering.py or check the density folders."
        )
if len(names) != D.shape[0]:
    raise SystemExit(f"{len(names)} structures found but the distance matrix has {D.shape[0]} rows")

# =========================
# UMAP (PRECOMPUTED DISTANCE)
# =========================
D = np.log1p(D)

reducer = umap.UMAP(
    metric="precomputed",
    n_neighbors=5, # IMPORTANT for small datasets
   n_components=3,
    min_dist=0.1,      # spreads points out
    spread=1,
    random_state=42
)

embedding = reducer.fit_transform(D)

print("UMAP embedding shape:", embedding.shape)

# =========================
# CLUSTERING (ON EMBEDDING)
# =========================
cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
umap_labels = cluster_model.fit_predict(embedding) + 1   # +1 so clusters run 1-4, matching fcluster in the other scripts

# =========================
# SAVE LABELS
# =========================
df_umap = pd.DataFrame({
    "EMDB_entry": names,
    "umap_cluster": umap_labels
})

save_path = os.path.join(save_dir, "umap_cluster_labels.csv")
df_umap.to_csv(save_path, index=False)

print(f"Saved cluster labels: {save_path}")

# =========================
# LOAD METADATA
# =========================
data = pd.read_csv(labels_csv_path)

data.columns = (
    data.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

data["EMDB_entry"] = data["EMDB_entry"].astype(str)

# align with order
data = data.set_index("EMDB_entry")
data = data.loc[names]

# =========================
# BUILD LABELS (WT LOGIC)
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

cmap = plt.colormaps.get_cmap("tab10")

poly_color_map = {
    poly: cmap(i)
    for i, poly in enumerate(unique_polys)
}

point_colors = [poly_color_map[p] for p in polymorphs]

# =========================
# PLOT (MATCH YOUR PCA STYLE)
# =========================
plt.figure(figsize=(8, 7))

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=umap_labels,
    cmap="tab10",
    s=50
)

# labels
for i, name in enumerate(labels):
    plt.text(
        embedding[i, 0],
        embedding[i, 1],
        name,
        fontsize=6,
        alpha=0.8
    )

plt.xlabel("UMAP-1", fontsize=11)
plt.ylabel("UMAP-2", fontsize=11)

plt.title("UMAP of IAPP 3D Density Maps (Sinkhorn Distance)", fontsize=13)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.tight_layout()

plt.savefig(os.path.join(save_dir, "umap_scatter.png"), dpi=300)
plt.show()

# =========================
# OPTIONAL: PRINT CLUSTERS
# =========================
print("\nUMAP cluster labels:")
for name, label in zip(names, umap_labels):
    print(name, "->", label)
