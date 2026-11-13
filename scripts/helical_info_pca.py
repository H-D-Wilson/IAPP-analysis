import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

# ===============================
# Step 1: Load data
# ===============================
# Paths are relative to the repo root, so they work on any machine
REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="PCA and clustering of the helical info from the Trace_y master CSV.")
parser.add_argument("--info-csv", type=Path,
                    default=REPO_ROOT / "data" / "Masterhelicalinfo.csv",
                    help="Trace_y master helical info CSV (default: data/Masterhelicalinfo.csv)")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Sequence Variant (default: data/Labels.csv)")
parser.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "output" / "helical_pca",
                    help="Where to save outputs (default: output/helical_pca)")
args = parser.parse_args()

file_path = args.info_csv

labels_csv_path = args.labels_csv

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

# remove NaNs
valid_mask = ~X.isnull().any(axis=1)
X = X[valid_mask]
labels = labels[valid_mask]

print(f"\nUsing {len(X)} valid entries")

# ===============================
# Step 3: Standardise
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Step 4: PCA
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_ * 100

print("\nExplained variance:")
print(f"PC1: {explained_var[0]:.2f}%")
print(f"PC2: {explained_var[1]:.2f}%")

# ===============================
# Step 4b: CLUSTER PCA SPACE
# ===============================
X_pca = np.asarray(X_pca)
X_pca = np.nan_to_num(X_pca)

X_pca_norm = normalize(X_pca)

Z = linkage(X_pca_norm, method='average')

pca_labels = fcluster(Z, t=4, criterion='maxclust')

# ===============================
# SAVE CLUSTER LABELS (FIXED)
# ===============================
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

df_pca = pd.DataFrame({
    "EMDB_entry": labels.values,
    "helical_info_pca_cluster": pca_labels
})

save_path = os.path.join(save_dir, "pca_cluster_labels.csv")
df_pca.to_csv(save_path, index=False)

print(f"\nSaved PCA cluster labels → {save_path}")

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
data = data.loc[labels.values]   # match the PCA rows by EMDB_entry (not by position)

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
plt.title("PCA of IAPP Helical Info (First Two Components)", fontsize=13)

# Gridlines (clean + readable)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# Optional: add axis lines at origin (nice visual reference)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

# Tight layout
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'pca_scatter.png'), dpi=300)
plt.show()


# ===============================
# Step 6: Feature loadings
# ===============================
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=feature_cols
)

print("\nPCA Loadings:")
print(loadings)

# save loadings
loadings_path = os.path.join(save_dir, "helical_pca_loadings.csv")
loadings.to_csv(loadings_path)

print(f"Saved loadings → {loadings_path}")
