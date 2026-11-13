import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageChops


# =========================================================
# PATHS
# =========================================================

# Paths are relative to the repo root, so they work on any machine.
# Defaults point at the outputs of the earlier scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

parser = argparse.ArgumentParser(
    description="Dendrogram of the 3D density map (Sinkhorn) clustering with the Cryo-EM, "
                "simulated AFM and contact-point images aligned beneath each leaf.")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Polymorph, Sequence Variant (default: data/Labels.csv)")
parser.add_argument("--distance-matrix", type=Path,
                    default=OUTPUT_DIR / "emd_clustering" / "sinkhorn_distance_matrix.npy",
                    help="Sinkhorn distance matrix from 10_emd_density_clustering.py")
parser.add_argument("--matrix-order-csv", type=Path,
                    default=OUTPUT_DIR / "emd_clustering" / "sinkhorn_matrix_order.csv",
                    help="EMDB_entry in the row/column order of the distance matrix "
                         "(written by 10_emd_density_clustering.py)")
parser.add_argument("--leaf-order-csv", type=Path,
                    default=OUTPUT_DIR / "emd_clustering" / "sinkhorn_leaf_order.csv",
                    help="Saved leaf order from script 10; used only to check that the dendrogram "
                         "drawn here has the same leaf order")
parser.add_argument("--cryo-em-dir", type=Path,
                    default=OUTPUT_DIR / "cropped_panels",
                    help="Row 1 images, emd_<id>.png (default: output/cropped_panels)")
parser.add_argument("--afm-sim-dir", type=Path,
                    default=OUTPUT_DIR / "manual_cropped_batch" / "right",
                    help="Row 2 images, emd_<id>.png (default: output/manual_cropped_batch/right)")
parser.add_argument("--contact-point-dir", type=Path,
                    default=OUTPUT_DIR / "manual_cropped_batch" / "middle",
                    help="Row 3 images, emd_<id>.png (default: output/manual_cropped_batch/middle)")
parser.add_argument("--output-dir", type=Path,
                    default=OUTPUT_DIR / "figures",
                    help="Where to save the PNG and PDF (default: output/figures)")
args = parser.parse_args()

labels_csv = args.labels_csv

distance_matrix_path = args.distance_matrix

# This CSV must contain the EMDB entries in the exact row/column
# order used when the Sinkhorn distance matrix was created.
#
# Expected column:
# EMDB_entry
matrix_order_csv = args.matrix_order_csv

leaf_order_csv = args.leaf_order_csv

cryo_em_dir = args.cryo_em_dir

afm_sim_dir = args.afm_sim_dir

contact_point_dir = args.contact_point_dir

save_dir = args.output_dir

os.makedirs(save_dir, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalise_emdb_id(value):
    """
    Convert an EMDB identifier into the filename/metadata form used
    throughout the script.

    Examples:
        10671.0   -> "10671"
        "10671"   -> "10671"
        "EMD-10671" -> "10671"
        "emd_10671" -> "10671"
    """
    value = str(value).strip()

    value = (
        value
        .replace("EMD-", "")
        .replace("emd-", "")
        .replace("EMD_", "")
        .replace("emd_", "")
    )

    # Remove a decimal added by CSV parsing, e.g. "10671.0"
    if value.endswith(".0"):
        value = value[:-2]

    return value


def crop_background(image, background_colour=(0, 0, 0)):
    """
    Crop uniform black borders from an RGB PIL image.

    Change background_colour to (255, 255, 255) if the
    exported images have white backgrounds.
    """
    background = Image.new(
        image.mode,
        image.size,
        background_colour
    )

    difference = ImageChops.difference(
        image,
        background
    )

    bounding_box = difference.getbbox()

    if bounding_box:
        return image.crop(bounding_box)

    return image


def place_image(
    axis,
    image_path,
    x_position,
    zoom
):
    """
    Place one image at a specified dendrogram leaf position.
    """
    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        return

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        image = crop_background(
            image,
            background_colour=(0, 0, 0)
        )

        # Prevent unnecessarily large images consuming memory.
        image.thumbnail((700, 700))

        image_box = OffsetImage(
            np.asarray(image),
            zoom=zoom,
            interpolation="lanczos"
        )

        annotation = AnnotationBbox(
            image_box,
            (x_position, 0.5),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0
        )

        axis.add_artist(annotation)

    except (OSError, ValueError) as error:
        print(
            f"Could not load {image_path}: {error}"
        )


# =========================================================
# LOAD THE MATRIX OBSERVATION ORDER
# =========================================================
# A NumPy distance matrix stores only numbers. It does not
# store the EMDB identifiers associated with its rows/columns.
#
# Therefore, this order file is essential for attaching the
# correct labels, metadata and images to the dendrogram.

order_df = pd.read_csv(matrix_order_csv)

order_df.columns = (
    order_df.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

if "EMDB_entry" not in order_df.columns:
    raise KeyError(
        "The matrix-order CSV must contain a column named "
        "'EMDB_entry'. Available columns are: "
        + ", ".join(order_df.columns)
    )

input_ids = [
    normalise_emdb_id(value)
    for value in order_df["EMDB_entry"]
]

if len(input_ids) != len(set(input_ids)):
    duplicated_ids = (
        pd.Series(input_ids)[
            pd.Series(input_ids).duplicated(keep=False)
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "The matrix-order file contains duplicated EMDB entries: "
        + ", ".join(duplicated_ids)
    )

print(
    "Loaded matrix observation order:",
    len(input_ids),
    "structures"
)


# =========================================================
# LOAD EMDBCLUSTER / SINKHORN DISTANCE MATRIX
# =========================================================

distance_matrix = np.load(
    distance_matrix_path,
    allow_pickle=False
)

print(
    "Loaded Sinkhorn distance matrix:",
    distance_matrix.shape
)


# =========================================================
# VALIDATE DISTANCE MATRIX
# =========================================================

if distance_matrix.ndim != 2:
    raise ValueError(
        "The distance matrix must be a two-dimensional array. "
        f"Loaded array has {distance_matrix.ndim} dimensions."
    )

if distance_matrix.shape[0] != distance_matrix.shape[1]:
    raise ValueError(
        "The distance matrix must be square. "
        f"Loaded shape: {distance_matrix.shape}"
    )

if distance_matrix.shape[0] != len(input_ids):
    raise ValueError(
        "The number of EMDB identifiers does not match the "
        "distance-matrix dimensions. "
        f"The matrix contains {distance_matrix.shape[0]} observations, "
        f"but the order file contains {len(input_ids)} identifiers."
    )

if not np.all(np.isfinite(distance_matrix)):
    raise ValueError(
        "The distance matrix contains NaN or infinite values."
    )

# Work with a floating-point copy so that minor numerical
# corrections do not modify the original loaded object.
distance_matrix = np.asarray(
    distance_matrix,
    dtype=float
).copy()

# Tiny floating-point asymmetries can be introduced during
# numerical processing. Reject substantial asymmetry, but
# average negligible differences.
maximum_asymmetry = np.max(
    np.abs(distance_matrix - distance_matrix.T)
)

if maximum_asymmetry > 1e-8:
    raise ValueError(
        "The distance matrix is not symmetric. "
        f"Maximum absolute asymmetry: {maximum_asymmetry:.3e}"
    )

distance_matrix = (
    distance_matrix + distance_matrix.T
) / 2.0

# Distances should be zero on the diagonal.
maximum_diagonal_value = np.max(
    np.abs(np.diag(distance_matrix))
)

if maximum_diagonal_value > 1e-8:
    raise ValueError(
        "The diagonal of the distance matrix is not zero. "
        f"Maximum absolute diagonal value: "
        f"{maximum_diagonal_value:.3e}"
    )

np.fill_diagonal(
    distance_matrix,
    0.0
)

# Permit only negligible floating-point values below zero.
minimum_distance = np.min(distance_matrix)

if minimum_distance < -1e-10:
    raise ValueError(
        "The distance matrix contains negative values. "
        f"Minimum distance: {minimum_distance}"
    )

distance_matrix[
    distance_matrix < 0
] = 0.0

print(
    "Distance-matrix validation passed."
)


# =========================================================
# CONVERT TO SCIPY CONDENSED FORMAT
# =========================================================

condensed_distances = squareform(
    distance_matrix,
    checks=True
)

print(
    "Generated condensed distance vector:",
    condensed_distances.shape
)


# =========================================================
# RECONSTRUCT HIERARCHICAL CLUSTERING
# =========================================================

Z = linkage(
    condensed_distances,
    method="average"
)

print(
    "Generated linkage matrix:",
    Z.shape
)


# =========================================================
# IMAGE ROW SETTINGS
# =========================================================
# Edit only the filename functions if your images use
# different filename patterns.

image_rows = [
    {
        "directory": cryo_em_dir,
        "filename": lambda emdb: f"emd_{emdb}.png",
    },
    {
        "directory": afm_sim_dir,
        "filename": lambda emdb: f"emd_{emdb}.png",
    },
    {
        "directory": contact_point_dir,
        "filename": lambda emdb: f"emd_{emdb}.png",
    },
]


# =========================================================
# LOAD METADATA
# =========================================================

metadata = pd.read_csv(labels_csv)

metadata.columns = (
    metadata.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

required_metadata_columns = {
    "EMDB_entry",
    "Polymorph",
    "Sequence_Variant"
}

missing_columns = (
    required_metadata_columns
    - set(metadata.columns)
)

if missing_columns:
    raise KeyError(
        "Labels.csv is missing the following required columns: "
        + ", ".join(sorted(missing_columns))
    )

metadata["EMDB_entry"] = [
    normalise_emdb_id(value)
    for value in metadata["EMDB_entry"]
]

if metadata["EMDB_entry"].duplicated().any():
    duplicated_metadata_ids = (
        metadata.loc[
            metadata["EMDB_entry"].duplicated(
                keep=False
            ),
            "EMDB_entry"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Labels.csv contains duplicated EMDB entries: "
        + ", ".join(duplicated_metadata_ids)
    )

metadata = metadata.set_index(
    "EMDB_entry"
)

missing_metadata = [
    emdb
    for emdb in input_ids
    if emdb not in metadata.index
]

if missing_metadata:
    raise KeyError(
        "Metadata are missing for the following EMDB entries: "
        + ", ".join(missing_metadata)
    )

# This is the critical alignment step:
# metadata row 0 now corresponds to distance-matrix row 0,
# metadata row 1 corresponds to distance-matrix row 1, etc.
metadata = metadata.loc[
    input_ids
].copy()

print(
    "Metadata aligned to distance-matrix order:",
    metadata.shape
)


# =========================================================
# BUILD SHORT DENDROGRAM LABELS
# =========================================================
# Only display the EMDB entry number.
# Polymorph identity is conveyed by label colour.

labels = input_ids.copy()


# =========================================================
# POLYMORPH COLOUR MAP
# =========================================================

polymorphs = (
    metadata["Polymorph"]
    .astype(str)
    .str.strip()
    .to_numpy()
)

unique_polymorphs = pd.unique(
    polymorphs
)

colour_map = plt.colormaps["turbo"]

polymorph_colour_map = {
    polymorph: colour_map(
        index / max(
            len(unique_polymorphs) - 1,
            1
        )
    )
    for index, polymorph
    in enumerate(unique_polymorphs)
}


# =========================================================
# FIGURE LAYOUT
# =========================================================

# A3 landscape dimensions in inches.
figure_width = 16.54
figure_height = 11.69

fig = plt.figure(
    figsize=(figure_width, figure_height),
    constrained_layout=False
)

# Visual parameters retained exactly from the current script.
grid = fig.add_gridspec(
    nrows=5,
    ncols=1,
    height_ratios=[
        5.5,
        1.15,
        1.05,
        1.05,
        1.05
    ],
    left=0.065,
    right=0.985,
    bottom=0.035,
    top=0.845,
    hspace=0.0
)

ax_dendrogram = fig.add_subplot(
    grid[0]
)

ax_labels = fig.add_subplot(
    grid[1],
    sharex=ax_dendrogram
)

image_axes = [
    fig.add_subplot(
        grid[row_number],
        sharex=ax_dendrogram
    )
    for row_number in range(2, 5)
]


# =========================================================
# DRAW DENDROGRAM
# =========================================================

dendrogram_result = dendrogram(
    Z,
    labels=None,
    orientation="top",
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=None,
    above_threshold_color="black",
    ax=ax_dendrogram
)

leaf_indices = dendrogram_result[
    "leaves"
]

# SciPy places leaves at 5, 15, 25, 35, ...
leaf_x_positions = (
    np.arange(
        len(leaf_indices)
    ) * 10 + 5
)

ordered_emdb_ids = [
    input_ids[index]
    for index in leaf_indices
]

# Check the leaf order against the one saved by 10_emd_density_clustering.py
if leaf_order_csv.exists():
    saved_leaf_order = [
        normalise_emdb_id(value)
        for value in pd.read_csv(leaf_order_csv)["EMDB_entry"]
    ]
    if saved_leaf_order == ordered_emdb_ids:
        print(f"Leaf order matches {leaf_order_csv.name}")
    else:
        print(f"WARNING: leaf order differs from {leaf_order_csv.name}")
else:
    print(f"Leaf order file not found, skipping check: {leaf_order_csv}")


# =========================================================
# ADD EMDB + SEQUENCE VARIANT LABELS
# =========================================================

ax_labels.set_xlim(
    ax_dendrogram.get_xlim()
)

ax_labels.set_ylim(
    0,
    1
)

ax_labels.set_xticks([])
ax_labels.set_yticks([])

for spine in ax_labels.spines.values():
    spine.set_visible(False)

for x_position, observation_index in zip(
    leaf_x_positions,
    leaf_indices
):
    emdb_id = input_ids[
        observation_index
    ]

    polymorph = polymorphs[
        observation_index
    ]

    # Use the EMDB ID explicitly rather than relying only
    # on positional indexing.
    sequence_variant = str(
        metadata.loc[
            emdb_id,
            "Sequence_Variant"
        ]
    ).strip()

    # Keep WT compact, but retain mutation tags for variants.
    if sequence_variant.upper() == "WT":
        display_label = emdb_id
    else:
        display_label = (
            f"{emdb_id} ({sequence_variant})"
        )

    ax_labels.text(
        x_position,
        0.52,
        display_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=polymorph_colour_map[
            polymorph
        ],
        clip_on=False
    )


# =========================================================
# FORMAT DENDROGRAM
# =========================================================

ax_dendrogram.set_title(
    "Density Map Clustering",
    fontsize=18,
    pad=55
)

ax_dendrogram.set_ylabel(
    "Sinkhorn Distance",
    fontsize=13
)

ax_dendrogram.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False
)

ax_dendrogram.tick_params(
    axis="y",
    labelsize=10
)

ax_dendrogram.spines[
    "top"
].set_visible(False)

ax_dendrogram.spines[
    "right"
].set_visible(False)

ax_dendrogram.spines[
    "bottom"
].set_visible(True)


# =========================================================
# ADD THREE COMPACT IMAGE ROWS
# =========================================================

# Visual parameter retained exactly from the current script.
image_zoom = 0.1

dendrogram_xlim = (
    ax_dendrogram.get_xlim()
)

for row_axis, row_configuration in zip(
    image_axes,
    image_rows
):
    row_axis.set_xlim(
        dendrogram_xlim
    )

    row_axis.set_ylim(
        0,
        1
    )

    row_axis.set_xticks([])
    row_axis.set_yticks([])

    row_axis.set_facecolor(
        "white"
    )

    for spine in row_axis.spines.values():
        spine.set_visible(False)

    for x_position, emdb_id in zip(
        leaf_x_positions,
        ordered_emdb_ids
    ):
        filename = row_configuration[
            "filename"
        ](emdb_id)

        image_path = os.path.join(
            row_configuration[
                "directory"
            ],
            filename
        )

        place_image(
            axis=row_axis,
            image_path=image_path,
            x_position=x_position,
            zoom=image_zoom
        )


# =========================================================
# POLYMORPH LEGEND
# =========================================================

legend_handles = [
    mpatches.Patch(
        facecolor=polymorph_colour_map[
            polymorph
        ],
        edgecolor="none",
        label=polymorph
    )
    for polymorph in unique_polymorphs
]

# Visual parameters retained exactly from the current script.
fig.legend(
    handles=legend_handles,
    title="Polymorph",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=min(
        len(legend_handles),
        8
    ),
    fontsize=8,
    title_fontsize=9,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.2
)


# =========================================================
# SAVE
# =========================================================

png_path = os.path.join(
    save_dir,
    "density_map_cluster_multiscale_landscape.png"
)

pdf_path = os.path.join(
    save_dir,
    "density_map_cluster_multiscale_landscape.pdf"
)

fig.savefig(
    png_path,
    dpi=600,
    facecolor="white"
)

fig.savefig(
    pdf_path,
    facecolor="white"
)

print(
    "Saved PNG:",
    png_path
)

print(
    "Saved PDF:",
    pdf_path
)

plt.show()