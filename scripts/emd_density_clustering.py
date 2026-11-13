import argparse
import os
import time
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.ndimage import zoom
from scipy.spatial.distance import squareform
from tqdm import tqdm

# =========================
# SETTINGS
# =========================
# Paths are relative to the repo root, so they work on any machine
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Sinkhorn (optimal-transport) clustering of the 3D raw density maps.")
parser.add_argument("--density-dir", type=Path,
                    default=REPO_ROOT / "data" / "RAW density",
                    help="Folder of <EMDB_entry>.mat density files (default: data/RAW density)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Sequence Variant, Polymorph (default: data/Labels.csv)")
parser.add_argument("--image-dir", type=Path,
                    default=REPO_ROOT / "output" / "scalebar_images" / "cross_section",
                    help="Folder of scalebar_emd_<id>.png images shown under the dendrogram; "
                         "missing images are skipped (default: output/scalebar_images/cross_section, from 04b_add_scale_bars_cross_section.py)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "emd_clustering",
                    help="Where to save outputs (default: output/emd_clustering)")
parser.add_argument("--resampled-dir", type=Path,
                    default=REPO_ROOT / "data" / "density_64",
                    help="Where to save the resampled 64x64x64 density volumes, one <EMDB id>.npy each "
                         "(default: data/density_64)")
parser.add_argument("--no-save-resampled", action="store_true",
                    help="Do not save the resampled volumes")
parser.add_argument("--use-resampled", action="store_true",
                    help="Read the resampled volumes from --resampled-dir instead of the raw .mat maps "
                         "(used automatically when the raw folder is missing or empty)")
args = parser.parse_args()

folder = args.density_dir
resampled_dir = args.resampled_dir
if not args.no_save_resampled:
    resampled_dir.mkdir(parents=True, exist_ok=True)

reg = 0.10
target_shape = (64, 64, 64)

np.random.seed(42)

save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)   # created up front: the matrix order is saved inside the loading loop

# =========================
# LOAD + RESAMPLE VOXELS
# =========================
import scipy.io

def load_and_resample(filepath, target_shape=(64,64,64)):

    # =========================
    # TRY HDF5 (v7.3)
    # =========================
    try:
        with h5py.File(filepath, 'r') as f:
            raw = f['Raw_Density']
            v = np.array(raw['v']).squeeze()

    # =========================
    # FALLBACK (older MAT)
    # =========================
    except OSError:
        data = scipy.io.loadmat(filepath)

        # adjust this key if needed
        if "Raw_Density" in data:
            raw = data["Raw_Density"]

            # MATLAB struct extraction
            v = raw["v"][0,0]

        else:
            raise ValueError(f"No Raw_Density in {filepath}")

    # =========================
    # ENSURE NUMPY ARRAY
    # =========================
    v = np.array(v).squeeze()

    # =========================
    # ENSURE 3D
    # =========================
    if v.ndim == 1:
        size = int(round(v.size ** (1/3)))
        v = v.reshape((size, size, size))

    # =========================
    # RESAMPLE
    # =========================
    factors = (
        target_shape[0] / v.shape[0],
        target_shape[1] / v.shape[1],
        target_shape[2] / v.shape[2],
    )

    v = zoom(v, factors, order=1)

    # =========================
    # NORMALISE
    # =========================
    v = np.clip(v, 0, None)
    v = v / (np.sum(v) + 1e-12)

    return v
def voxel_to_point_cloud(v, n_samples=4000):
    """
    Convert voxel density into weighted point cloud
    """
    v = np.nan_to_num(v)  # safety


    grid = np.indices(v.shape).reshape(3, -1).T
    weights = v.flatten()

    # remove zero-mass voxels
    mask = weights > 1e-12
    grid = grid[mask]
    weights = weights[mask]

    # normalize weights
    weights = weights / (np.sum(weights) + 1e-12)

    # sample points based on density
    if len(weights) > n_samples:
        idx = np.random.choice(len(weights), size=n_samples, p=weights, replace=False)
    else:
        idx = np.random.choice(len(weights), size=n_samples, replace=True)

    pts = grid[idx]
    w = weights[idx]

    # renormalize weights
    w = w / (np.sum(w) + 1e-12)

    return pts.astype(np.float64), w.astype(np.float64)
# =========================
# PCA axis alignment
# =========================
def pca_align(points):
    centroid = np.mean(points, axis=0)
    X = points - centroid

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # enforce consistent orientation
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1

    X_aligned = X @ eigvecs

    return X_aligned


# =========================
# LOAD FILES
# =========================
raw_files = (
    sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
    if os.path.isdir(folder) else []
)
resampled_files = (
    sorted([f for f in os.listdir(resampled_dir) if f.endswith(".npy")])
    if resampled_dir.is_dir() else []
)

# Use the raw maps if they are available; otherwise (or with --use-resampled) use the
# resampled 64x64x64 volumes, which give the same results.
use_resampled = args.use_resampled or not raw_files
if use_resampled:
    if not resampled_files:
        raise SystemExit(f"No density maps found: no .mat files in {folder} and no .npy files in {resampled_dir}")
    files, source_dir = resampled_files, resampled_dir
    print(f"Using resampled volumes from {resampled_dir}")
else:
    files, source_dir = raw_files, folder
print(f"Found {len(files)} files")

point_sets = []
weight_sets = []
names = []

for f in files:
    path = os.path.join(source_dir, f)

    if use_resampled:
        v = np.load(path)
        if v.shape != tuple(target_shape):
            raise ValueError(f"{f} has shape {v.shape}, expected {target_shape}")
    else:
        v = load_and_resample(path, target_shape)

        # Save the resampled, normalised volume (small) so it can be shared without the large raw maps
        if not args.no_save_resampled:
            np.save(resampled_dir / f.replace(".mat", ".npy"), v)

    pts, w = voxel_to_point_cloud(v)

    pts = np.asarray(pts, dtype=np.float64)

    # =========================
    # PCA ALIGNMENT (ADD HERE)
    # =========================
    pts = pca_align(pts)

    variances = np.var(pts, axis=0)

    if not (variances[0] >= variances[1] >= variances[2]):
        print(f"WARNING: PCA issue in {f}")

    if variances[1] / (variances[0] + 1e-12) < 1e-3:
        print(f"WARNING: Degenerate fibril in {f}")
    pts = pts.astype(np.float64)

    # NORMALISE TO [0,1]
    pts /= (np.max(np.abs(pts)) + 1e-12)

    point_sets.append(pts)
    weight_sets.append(w)
    names.append(os.path.splitext(f)[0])
    matrix_order = pd.DataFrame({
        "matrix_index": np.arange(len(names)),
        "EMDB_entry": names
    })

    matrix_order.to_csv(
        os.path.join(
            save_dir,
            "sinkhorn_matrix_order.csv"
        ),
        index=False
    )

print(f"Loaded {len(point_sets)} structures (spatial OT ready)")

# =========================
# DISTANCE MATRIX (SINKHORN)
# =========================
n = len(point_sets)
D = np.zeros((n, n))

start = time.time()
pair_count = 0
total_pairs = n * (n - 1) // 2

print("Running spatial Sinkhorn...")

for i in tqdm(range(n)):

    A = point_sets[i]
    wa = weight_sets[i]

    for j in range(i + 1, n):

        B = point_sets[j]
        wb = weight_sets[j]

        # spatial cost matrix (efficient form)
        M = (
            np.sum(A**2, axis=1)[:, None] +
            np.sum(B**2, axis=1)[None, :] -
            2 * A @ B.T
        )
        M = np.clip(M, 0, 10)
        dist = ot.sinkhorn2(wa, wb, M, reg, method='sinkhorn_stabilized')

        if np.isnan(dist) or np.isinf(dist):
            print(f"WARNING: Bad value at ({i},{j}) -> set to 0")
            dist = 0.0

        D[i, j] = dist
        D[j, i] = dist

        pair_count += 1

    elapsed = time.time() - start
    eta = (elapsed / max(pair_count, 1)) * (total_pairs - pair_count)

    tqdm.write(f"Row {i+1}/{n} done | ETA {eta/60:.1f} min")

# =========================
# SAVE MATRIX
# =========================


# Save the order again beside the completed matrix
matrix_order.to_csv(
        os.path.join(
            save_dir,
            "sinkhorn_matrix_order.csv"
        ),
        index=False
    )


np.save(
    os.path.join(save_dir, "sinkhorn_distance_matrix.npy"),
    D
)

# =========================
# CLUSTERING
# =========================
condensed = squareform(D)
Z = linkage(condensed, method="average")

n_clusters = 4
emd_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

print("\nEMD cluster labels:")
for name, label in zip(names, emd_labels):
    print(name, "->", label)

df_labels = pd.DataFrame({
    "EMDB_entry": names,
    "emd_cluster": emd_labels
})

save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "emd_cluster_labels.csv")
df_labels.to_csv(save_path, index=False)

print(f"Saved: {save_path}")

# =========================
# LOAD METADATA
# =========================
data = pd.read_csv(args.labels_csv)

print("\nCOLUMNS IN CSV:")
for col in data.columns:
    print(f"'{col}'")

data.columns = (
    data.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

print("\nCLEANED COLUMNS:")
print(data.columns.tolist())

data["EMDB_entry"] = data["EMDB_entry"].astype(str)

# =========================
# ALIGN METADATA
# =========================
data = data.set_index("EMDB_entry")
data = data.loc[names]

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
# IMPORTS
# =========================
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageChops
import re

# =========================
# FIGURE LAYOUT
# =========================
fig = plt.figure(figsize=(36, 14))

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[5, 1]
)

# =========================
# TOP AXIS (DENDROGRAM)
# =========================
ax1 = fig.add_subplot(gs[0])

dend = dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90,
    leaf_font_size=8,
    ax=ax1
)
# =========================
# POLYMORPH COLOUR MAP
# =========================
polymorphs = data["Polymorph"].astype(str).values
unique_polys = np.unique(polymorphs)

cmap = plt.get_cmap("turbo", len(unique_polys))

poly_color_map = {
    poly: cmap(i)
    for i, poly in enumerate(unique_polys)
}
# =========================
# COLOUR LABELS
# =========================
leaf_order = dend['leaves']

xlbls = ax1.get_xmajorticklabels()

for i, lbl in enumerate(xlbls):

    idx = leaf_order[i]
    poly = polymorphs[idx]

    lbl.set_color(poly_color_map[poly])

    leaf_df = pd.DataFrame({
        "leaf_position": np.arange(len(leaf_order)),
        "matrix_index": leaf_order,
        "EMDB_entry": [names[i] for i in leaf_order],
        "emd_cluster": [emd_labels[i] for i in leaf_order]
    })

    leaf_df.to_csv(
        os.path.join(
            save_dir,
            "sinkhorn_leaf_order.csv"
        ),
        index=False
    )

# =========================
# IMAGE AXIS
# =========================
ax2 = fig.add_axes([0.05, 0.02, 0.90, 0.22])

ax2.axis("off")

# =========================
# IMAGE DIRECTORY
# =========================
image_dir = args.image_dir

# =========================
# ADD IMAGES
# =========================
x_positions = np.linspace(0.01, 0.99, len(leaf_order))

for x, idx in zip(x_positions, leaf_order):

    emdb_id = names[idx]

    img_path = os.path.join(
        image_dir,
        f"scalebar_emd_{emdb_id}.png"
    )

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")

    bg = Image.new(img.mode, img.size, (0,0,0))
    diff = ImageChops.difference(img, bg)

    bbox = diff.getbbox()

    if bbox:
        img = img.crop(bbox)

    img.thumbnail((160,160))

    imagebox = OffsetImage(
        img,
        zoom=0.20
    )

    ab = AnnotationBbox(
        imagebox,
        (x, 0.5),
        xycoords='axes fraction',
        frameon=False
    )

    ax2.add_artist(ab)

# =========================
# LEGEND
# =========================
import matplotlib.patches as mpatches

legend_handles = [
    mpatches.Patch(
        color=poly_color_map[p],
        label=p
    )
    for p in unique_polys
]

ax1.legend(
    handles=legend_handles,
    title="Polymorph",
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=9
)

# =========================
# TITLES
# =========================
ax1.set_title("IAPP Fibril Density Map Clustering")
ax1.set_xlabel("EMDB Entry")
ax1.set_ylabel("Sinkhorn Distance")

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, "EMD_3D_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
