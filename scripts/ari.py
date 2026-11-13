import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# =========================================================
# PATHS
# =========================================================

# Paths are relative to the repo root, so they work on any machine.
# Defaults point at the outputs of the earlier scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

parser = argparse.ArgumentParser(
    description="Pairwise Adjusted Rand Index between the clusterings from all analysis scripts.")
parser.add_argument("--hog-csv", type=Path,
                    default=OUTPUT_DIR / "hog" / "hog_final_layout.csv",
                    help="HOG cluster labels from 06_hog.py (default: output/hog/hog_final_layout.csv)")
parser.add_argument("--sinkhorn-csv", type=Path,
                    default=OUTPUT_DIR / "emd_clustering" / "emd_cluster_labels.csv",
                    help="3D density cluster labels from 10_emd_density_clustering.py")
parser.add_argument("--helical-csv", type=Path,
                    default=OUTPUT_DIR / "helical_clustering" / "helical_cluster_labels.csv",
                    help="Helical info cluster labels from 09_helical_info_clustering.py")
parser.add_argument("--pca-csv", type=Path,
                    default=OUTPUT_DIR / "pca" / "Label values" / "pca_cluster_labels.csv",
                    help="Image PCA cluster labels from 07_pca_images.py")
parser.add_argument("--helical-pca-csv", type=Path,
                    default=OUTPUT_DIR / "helical_pca" / "pca_cluster_labels.csv",
                    help="Helical info PCA cluster labels from 08_helical_info_pca.py")
parser.add_argument("--umap-csv", type=Path,
                    default=OUTPUT_DIR / "emd_umap" / "umap_cluster_labels.csv",
                    help="3D density UMAP cluster labels from 11_emd_umap.py")
parser.add_argument("--output-dir", type=Path,
                    default=OUTPUT_DIR / "ari",
                    help="Where to save the ARI matrix and merged table (default: output/ari)")
args = parser.parse_args()

hog_csv = args.hog_csv
sinkhorn_csv = args.sinkhorn_csv
helical_csv = args.helical_csv
pca_csv = args.pca_csv
helical_pca_csv = args.helical_pca_csv
umap_csv = args.umap_csv

os.makedirs(args.output_dir, exist_ok=True)
save_path = args.output_dir / "ARI_pairwise_matrix.csv"

# =========================================================
# CHECK FILES EXIST
# =========================================================

input_paths = {
    "HOG": hog_csv,
    "Sinkhorn": sinkhorn_csv,
    "Helical": helical_csv,
    "PCA": pca_csv,
    "Helical PCA": helical_pca_csv,
    "UMAP": umap_csv,
}

for method_name, path in input_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{method_name} CSV was not found:\n{path}"
        )


# =========================================================
# LOAD DATA
# =========================================================

hog_df = pd.read_csv(hog_csv)
sinkhorn_df = pd.read_csv(sinkhorn_csv)
helical_df = pd.read_csv(helical_csv)
pca_df = pd.read_csv(pca_csv)
helical_pca_df = pd.read_csv(helical_pca_csv)
umap_df = pd.read_csv(umap_csv)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

dfs = [
    hog_df,
    sinkhorn_df,
    helical_df,
    pca_df,
    helical_pca_df,
    umap_df,
]

for df in dfs:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # =========================================================
    # STANDARDISE IDENTIFIER COLUMN NAMES
    # =========================================================

    for df_name, df in {
        "HOG": hog_df,
        "Sinkhorn": sinkhorn_df,
        "Helical": helical_df,
        "PCA": pca_df,
        "Helical PCA": helical_pca_df,
        "UMAP": umap_df,
    }.items():

        if "EMDB_entry" not in df.columns:
            if "sample" in df.columns:
                df.rename(
                    columns={"sample": "EMDB_entry"},
                    inplace=True
                )
            else:
                raise KeyError(
                    f"{df_name} file has no recognised identifier column. "
                    f"Available columns: {df.columns.tolist()}"
                )


# =========================================================
# CHECK EMDB COLUMN EXISTS
# =========================================================

for method_name, df in zip(input_paths.keys(), dfs):
    if "EMDB_entry" not in df.columns:
        raise KeyError(
            f"{method_name} file does not contain 'EMDB_entry'. "
            f"Available columns: {df.columns.tolist()}"
        )



# =========================================================
# BUILD DFS LIST
# =========================================================
for df in dfs:
    df["EMDB_entry"] = (
        df["EMDB_entry"]
        .astype(str)
        .str.strip()
        .str.replace(r"(?i)^emd[-_ ]*", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


# =========================================================
# SELECT AND RENAME CLUSTER COLUMNS
# =========================================================

hog_df = hog_df[
    ["EMDB_entry", "image_cluster"]
].rename(
    columns={
        "image_cluster": "Cryo-EM Cross-Section Cluster"
    }
)

sinkhorn_df = sinkhorn_df[
    ["EMDB_entry", "emd_cluster"]
].rename(
    columns={
        "emd_cluster": "3D Density Map Cluster"
    }
)

helical_df = helical_df[
    ["EMDB_entry", "helical_info_cluster"]
].rename(
    columns={
        "helical_info_cluster": "Helical Info Cluster"
    }
)

pca_df = pca_df[
    ["EMDB_entry", "image_pca_cluster"]
].rename(
    columns={
        "image_pca_cluster": "Cryo-EM Cross-Section PCA"
    }
)

helical_pca_df = helical_pca_df[
    ["EMDB_entry", "helical_info_pca_cluster"]
].rename(
    columns={
        "helical_info_pca_cluster": "Helical Info PCA"
    }
)

umap_df = umap_df[
    ["EMDB_entry", "umap_cluster"]
].rename(
    columns={
        "umap_cluster": "3D Density Map UMAP"
    }
)


# =========================================================
# CHECK FOR DUPLICATE EMDB ENTRIES
# =========================================================

prepared_dfs = {
    "HOG": hog_df,
    "Sinkhorn": sinkhorn_df,
    "Helical": helical_df,
    "PCA": pca_df,
    "Helical PCA": helical_pca_df,
    "UMAP": umap_df,
}

for method_name, df in prepared_dfs.items():
    duplicate_ids = df.loc[
        df["EMDB_entry"].duplicated(),
        "EMDB_entry"
    ].tolist()

    if duplicate_ids:
        raise ValueError(
            f"{method_name} contains duplicate EMDB entries: "
            f"{duplicate_ids}"
        )


# =========================================================
# CHECK INPUT DATA
# =========================================================

print("\nROW COUNTS:")

for method_name, df in prepared_dfs.items():
    print(f"{method_name}: {len(df)}")

print("\nEXAMPLE NORMALISED IDs:")

for method_name, df in prepared_dfs.items():
    print(
        f"{method_name}: "
        f"{df['EMDB_entry'].head().tolist()}"
    )


# =========================================================
# MERGE ALL METHODS
# =========================================================

merged = (
    hog_df
    .merge(
        sinkhorn_df,
        on="EMDB_entry",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        helical_df,
        on="EMDB_entry",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        pca_df,
        on="EMDB_entry",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        helical_pca_df,
        on="EMDB_entry",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        umap_df,
        on="EMDB_entry",
        how="inner",
        validate="one_to_one"
    )
)

print("\nMERGED SHAPE:")
print(merged.shape)

print("\nMERGED ENTRIES:")
print(merged.head())


# =========================================================
# STOP IF MERGE FAILED
# =========================================================

if merged.empty:
    raise ValueError(
        "The merged dataframe is empty. "
        "The EMDB identifiers still do not match across the CSV files."
    )

if len(merged) < len(hog_df):
    print(
        "\nWARNING:"
        f" Only {len(merged)} of {len(hog_df)} HOG entries "
        "were shared across every method."
    )


# =========================================================
# METHOD COLUMN NAMES
# =========================================================

methods = [
    "Cryo-EM Cross-Section Cluster",
    "3D Density Map Cluster",
    "Helical Info Cluster",
    "Cryo-EM Cross-Section PCA",
    "Helical Info PCA",
    "3D Density Map UMAP",
]


# =========================================================
# CHECK CLUSTER LABEL VALUES
# =========================================================

print("\nCLUSTER COUNTS:")

for method in methods:
    print(f"\n{method}")
    print(
        merged[method]
        .value_counts()
        .sort_index()
    )


# =========================================================
# COMPUTE PAIRWISE ARI MATRIX
# =========================================================

ari_matrix = pd.DataFrame(
    np.eye(len(methods)),
    index=methods,
    columns=methods,
    dtype=float,
)

for i, method1 in enumerate(methods):
    for j in range(i + 1, len(methods)):
        method2 = methods[j]

        ari = adjusted_rand_score(
            merged[method1].to_numpy(),
            merged[method2].to_numpy(),
        )

        ari_matrix.loc[method1, method2] = ari
        ari_matrix.loc[method2, method1] = ari


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n===================================")
print("PAIRWISE ADJUSTED RAND INDEX")
print("===================================\n")

print(ari_matrix.round(4))


# =========================================================
# SAVE RESULTS
# =========================================================

ari_matrix.to_csv(save_path)

print("\nSaved ARI matrix:")
print(save_path)


# =========================================================
# OPTIONAL: SAVE MERGED LABEL TABLE
# =========================================================

merged_save_path = args.output_dir / "ARI_merged_cluster_labels.csv"

merged.to_csv(
    merged_save_path,
    index=False,
)

print("\nSaved merged cluster-label table:")
print(merged_save_path)

