import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster,leaves_list
from sklearn.preprocessing import StandardScaler, normalize
import os
import argparse
from pathlib import Path

# ===============================
# Step 1: Load data
# ===============================
# Paths are relative to the repo root, so they work on any machine
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Hierarchical clustering of the helical info from the Trace_y master CSV.")
parser.add_argument("--info-csv", type=Path,
                    default=REPO_ROOT / "data" / "Masterhelicalinfo.csv",
                    help="Trace_y master helical info CSV (default: data/Masterhelicalinfo.csv)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Sequence Variant, Polymorph (default: data/Labels.csv)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "helical_clustering",
                    help="Where to save outputs (default: output/helical_clustering)")
args = parser.parse_args()

file_path = args.info_csv

# Output folder (created up front, since the linkage matrix is saved before the labels step)
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_csv(file_path)

print("\nLoaded data:")
print(data.head())

# ===============================
# Step 2: Select features
# ===============================
feature_cols = [
    "Rise_nm",
    "Twist_deg",
    "Symmetry",
    "CsaEM",
    "Max_Height_nm",
    "Roundness_Index"
]

X = data[feature_cols]
labels = data["EMDB_entry"].astype(str)

# ===============================
# Step 3: Handle missing values
# ===============================
valid_mask = ~X.isnull().any(axis=1)

X = X[valid_mask]
labels = labels[valid_mask]

print(f"\nUsing {len(X)} valid entries after removing NaNs")

# ===============================
# Step 4: Standardisation
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional normalization
X_norm = normalize(X_scaled)

# ===============================
# Step 5: Hierarchical clustering
# ===============================
linkage_matrix = sch.linkage(X_norm, method='average', metric='euclidean')

pd.DataFrame(linkage_matrix).to_csv(
    save_dir / "Helical_linkage_matrix.csv",
    index=False
)
np.save(save_dir / "Helical_linkage_matrix.npy", linkage_matrix)   # loaded by 13b_collage_combiner_helical_info.py

# =========================
# LOAD METADATA
# =========================
import pandas as pd

data = pd.read_csv(args.labels_csv)
print("\nCOLUMNS IN CSV:")
for col in data.columns:
    print(f"'{col}'")

# Clean column names (VERY IMPORTANT)
data.columns = (
    data.columns
    .str.strip()                # remove leading/trailing spaces
    .str.replace(r"\s+", "_", regex=True)  # replace ANY whitespace with _
                  )

print("\nCLEANED COLUMNS:")
print(data.columns.tolist())

# Ensure EMDB entries are strings
data["EMDB_entry"] = data["EMDB_entry"].astype(str)

# =========================
# ALIGN METADATA WITH FILE ORDER
# =========================
# Your 'names' list must match EMDB_entry
# If names are like "33902", this works directly

data = data.set_index("EMDB_entry")
data = data.loc[labels.values]   # reorder to match clustering order


# ===============================
# Step 5b: Extract cluster labels
# ===============================
n_clusters = 4

cluster_labels = fcluster(
    linkage_matrix,
    t=n_clusters,
    criterion='maxclust'
)

cluster_df = pd.DataFrame({
    "EMDB_entry": labels.values,
    "helical_info_cluster": cluster_labels
})

# Ensure save directory exists
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "helical_cluster_labels.csv")
cluster_df.to_csv(csv_path, index=False)

print("\nSaved cluster labels:")
print(cluster_df.head())

leaf_order = leaves_list(linkage_matrix)

ordered_emdb = [labels.iloc[i] for i in leaf_order]

order_df = pd.DataFrame({
    "EMDB_entry": ordered_emdb,
    "order": np.arange(len(ordered_emdb))
})

order_df.to_csv(
    save_dir / "Helical_leaf_order.csv",
    index=False
)

# ===============================
# Step 6: Plot dendrogram
# ===============================
plt.figure(figsize=(14, 7))

sch.dendrogram(
    linkage_matrix,
    labels=labels.values,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Helical Info Clustering (IAPP)")
plt.xlabel("EMDB Entry")
plt.ylabel("Euclidean Distance")

plt.tight_layout()

plt.savefig(os.path.join(save_dir, "helical_dendrogram.png"), dpi=300)
plt.show()

   # reorder to match clustering order

# =========================
# BUILD LABELS (WT logic)
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
dend = sch.dendrogram(
   linkage_matrix,
    labels=labels,
    leaf_rotation=75,
    leaf_font_size=10
)

# =========================
# COLOUR LABELS BY POLYMORPH
# =========================
leaf_order = dend['leaves']

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

plt.title("Helical info clustering")
plt.xlabel("EMDB Entries")
plt.ylabel("Euclidean Distance")
plt.savefig(save_dir / "helical_dendrogram_polymorph.png", dpi=300)
plt.show()
plt.tight_layout()