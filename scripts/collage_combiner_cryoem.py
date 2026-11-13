import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.cluster.hierarchy import dendrogram
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
    description="Dendrogram of the HOG (Cryo-EM cross-section) clustering with the Cryo-EM, simulated AFM and contact-point images aligned beneath each leaf.")
parser.add_argument("--labels-csv", type=Path,
                    default=REPO_ROOT / "data" / "Labels.csv",
                    help="Metadata CSV with EMDB_entry, Polymorph, Sequence Variant (default: data/Labels.csv)")
parser.add_argument("--order-csv", type=Path,
                    default=OUTPUT_DIR / "hog" / "hog_cluster_labels.csv",
                    help="EMDB_entry column listing the structures in the order of the rows of the "
                         "linkage matrix (default: output/hog/hog_cluster_labels.csv, written by 06_hog.py)")
parser.add_argument("--linkage", type=Path,
                    default=OUTPUT_DIR / "hog" / "hog_linkage_matrix.npy",
                    help="Linkage matrix saved with np.save")
parser.add_argument("--leaf-order-csv", type=Path,
                    default=OUTPUT_DIR / "hog" / "hog_leaf_order.csv",
                    help="Saved leaf order from the clustering script; used only to check that the "
                         "dendrogram drawn here has the same leaf order")
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

order_csv = args.order_csv

linkage_path = args.linkage

leaf_order_csv = args.leaf_order_csv

cryo_em_dir = args.cryo_em_dir

afm_sim_dir = args.afm_sim_dir

contact_point_dir = args.contact_point_dir

save_dir = args.output_dir

os.makedirs(save_dir, exist_ok=True)


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
# HELPER FUNCTIONS
# =========================================================

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
        image = Image.open(image_path).convert("RGB")

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
# LOAD LINKAGE MATRIX
# =========================================================

Z = np.load(linkage_path)


print("Loaded linkage matrix:", Z.shape)

expected_structure_count = Z.shape[0] + 1


# =========================================================
# LOAD STRUCTURE IDs (in linkage-matrix row order)
# =========================================================

order_df = pd.read_csv(order_csv)

input_ids = (
    order_df["EMDB_entry"]
    .astype(str)
    .tolist()
)

print("Loaded structure identifiers:", len(input_ids))

if len(input_ids) != expected_structure_count:
    raise ValueError(
        "The number of EMDB identifiers does not match the "
        "number of observations represented by the linkage matrix. "
        f"The linkage matrix contains {expected_structure_count} "
        f"structures, while {len(input_ids)} identifiers were loaded."
    )


# =========================================================
# LOAD METADATA
# =========================================================

metadata = pd.read_csv(labels_csv)

metadata.columns = (
    metadata.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

metadata["EMDB_entry"] = (
    metadata["EMDB_entry"]
    .astype(str)
)

metadata = metadata.set_index("EMDB_entry")

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

# Keep metadata in the same order as the observations
# used to construct the linkage matrix.
metadata = metadata.loc[input_ids]


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
    .to_numpy()
)

unique_polymorphs = pd.unique(polymorphs)

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

# The dendrogram remains dominant.
# Image rows are compact and tightly stacked.
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

ax_dendrogram = fig.add_subplot(grid[0])

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

leaf_indices = dendrogram_result["leaves"]

# SciPy places leaves at 5, 15, 25, 35, ...
leaf_x_positions = (
    np.arange(len(leaf_indices)) * 10 + 5
)

ordered_emdb_ids = [
    input_ids[index]
    for index in leaf_indices
]

# Check the leaf order against the one saved by the clustering script
if leaf_order_csv.exists():
    saved_leaf_order = (
        pd.read_csv(leaf_order_csv)["EMDB_entry"]
        .astype(str)
        .tolist()
    )
    if saved_leaf_order == ordered_emdb_ids:
        print(f"Leaf order matches {leaf_order_csv.name}")
    else:
        print(f"WARNING: leaf order differs from {leaf_order_csv.name}")
else:
    print(f"Leaf order file not found, skipping check: {leaf_order_csv}")

# =========================================================
# ADD EMDB + SEQUENCE VARIANT LABELS
# =========================================================

ax_labels.set_xlim(ax_dendrogram.get_xlim())
ax_labels.set_ylim(0, 1)

ax_labels.set_xticks([])
ax_labels.set_yticks([])

for spine in ax_labels.spines.values():
    spine.set_visible(False)

for x_position, observation_index in zip(
    leaf_x_positions,
    leaf_indices
):
    emdb_id = input_ids[observation_index]

    polymorph = polymorphs[observation_index]

    sequence_variant = str(
        metadata.iloc[observation_index]["Sequence_Variant"]
    ).strip()

    # Keep WT compact, but retain mutation tags for variants
    if sequence_variant.upper() == "WT":
        display_label = emdb_id
    else:
        display_label = f"{emdb_id} ({sequence_variant})"

    ax_labels.text(
        x_position,
        0.52,
        display_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=polymorph_colour_map[polymorph],
        clip_on=False
    )

# =========================================================
# FORMAT DENDROGRAM
# =========================================================

ax_dendrogram.set_title(
    "Cryo-EM Cross-Section Clustering",
    fontsize=18,
    pad=55
)

ax_dendrogram.set_ylabel(
    "Cosine Distance",
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

ax_dendrogram.spines["top"].set_visible(
    False
)

ax_dendrogram.spines["right"].set_visible(
    False
)

# Keep the original axis appearance but remove the
# unnecessary x tick marks.
ax_dendrogram.spines["bottom"].set_visible(
    True
)


# =========================================================
# ADD THREE COMPACT IMAGE ROWS
# =========================================================

# Increase or decrease this value depending on image overlap.
image_zoom = 0.1

dendrogram_xlim = ax_dendrogram.get_xlim()

for row_axis, row_configuration in zip(
    image_axes,
    image_rows
):
    row_axis.set_xlim(dendrogram_xlim)
    row_axis.set_ylim(0, 1)

    row_axis.set_xticks([])
    row_axis.set_yticks([])

    row_axis.set_facecolor("white")

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
            row_configuration["directory"],
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

# Place the legend below the title and above the dendrogram.
fig.legend(
    handles=legend_handles,
    title="Polymorph",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=min(len(legend_handles), 8),
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
    "HOG_cluster_multiscale_landscape.png"
)

pdf_path = os.path.join(
    save_dir,
    "HOG_cluster_multiscale_landscape.pdf"
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

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)

plt.show()