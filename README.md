# IAPP fibril structure clustering and dimensionality reduction

This is my Python code used in my thesis titled  "Computational quantitative analysis of structural polymorphism in Islet Amyloid Polypeptide". The analysis code compares 31 Islet Amyloid Polypeptide (IAPP) cryo-EM structures from the Electron Microscopy Data Bank (EMDB); each structure is inspected at 3 different scales of morphology data these undergo clustering as well as dimensionality reduction. The results of these are then compared in terms of agreement:

| Representation | Clustering | 
|---|---|
| Cryo-EM cross-section images | HOG features, hierarchical clustering; PCA |
| Helical info (rise, twist, symmetry, cross-sectional area, height, roundness) | PCA; hierarchical clustering |
| 3D density maps | Sinkhorn (optimal-transport) distance, hierarchical clustering; UMAP |

The agreement between all six clusterings is measured with the Adjusted Rand Index (`ari.py`)script, and the final dendrogram figures with structure images were made with the 'collage_combiner_x.py' scripts.

The inputs (figures, density maps and helical tables) were generated with **Trace_y**, a MATLAB program for amyloid filament tracing. The MATLAB step is not part of this repository; the Trace_y outputs needed to run everything here are in `data/` (see [`data/README.md`](data/README.md)). So reproducing the data produced should not be difficult. As said if you wish to feed your own data produced from trace_y into this pipline the 'data/README.md' provides extra details.

## Repository layout

```
.
├── data/                        input data (see data/README.md)
│   ├── multipanel_figures/      Trace_y multi-panel PNGs
│   ├── RAW density/             3D density maps, <EMDB id>.mat
│   ├── Labels.csv               polymorph / variant / origin / pH per structure
│   ├── Masterhelicalinfo.csv    helical info table
│   └── Pitchtable.csv           helical pitch per structure
├── scripts/                     the pipeline
├── output/                      everything the scripts generate (git-ignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The Python version I recommend for using this pipeline is 3.11 or above (the scripts in this case were used and made in version 3.12). `torch` is only used for image tensors, so a CPU-only install is enough.

Worth mentioning that because of the **repository root**, all of the functions can be run relative to the repository, so the scripts can work wherever they are placed; as a result, reproducibility should be smooth. Also important to note that every script has its own little  `--help` option listing its inputs and outputs, and each input/output folder can be overridden on the command line.

## Pipeline

```
data/multipanel_figures ─► crop panels ─► (manual crops where Auto crop fails)
                                   │
                                   ├─► add scale bars ─► collages
                                   │
data/Labels.csv ───────────────────┼─► HOG ─────────┐
                                   ├─► image PCA ───┤
data/Masterhelicalinfo.csv ────────┼─► info PCA ────┼─► ARI comparison
                                   ├─► info cluster ┤
data/raw density ──────────────────┼─► Sinkhorn ─┬──┤
                                   │                └─► UMAP
                                   └─► collage combiner scripts ─► combined dendrogram figures
```

### 1. Image preparation

| Step | Script | What it does |
|---|---|---|
| 01 | `auto_crop_panels.py` | This script detects and crops the top-left (cross-section) panel of each Trace_y figure by default but can be adjusted to also auto-crop the top right, bottom right and bottom left panels if desired; it also labels the cropped panels with their respective EMDB ID, and makes a preliminary collage to check the results across all of the cropped images and prints the panels it could not crop. |
| 02 | `manual_crop.py` | This script manually crops one panel of one figure (`--panel left/middle/right` or `--ratio`). Use for failures from step 01. The resulting crop is put in output/manual_cropped_batch/<panel>/, so the automatic crops are never overwritten. You can add --replace-auto to deliberately replace a failed cross-section crop in output/cropped_panels/. |
| 03 | `manual_crop_batch.py` | This script is the same as the manual crop script but is applied to every figure. Output goes to `output/manual_cropped_batch/<panel>/`. |
| 04a | `scalebar_adder_longitudinal.py` | This script adds scale bars to crops of the simulated-AFM (one-pitch longitudinal) panel; specifically, it calculates the length of these bars using each structure's pitch from `Pitchtable.csv`. |
| 04b | `scalebar_adder_cross_section.py` | This script adds scale bars to the cross-section crops (Cryo-EM images and contact point images). |
| 05 | `collage_maker.py` | This script takes the images from an image folder and combines them into a figure-ready collage. |

```bash
python scripts/auto_crop_panels.py
python scripts/manual_crop.py emd_18827.png --panel middle     # for any failures
python scripts/manual_crop_batch.py --panel right              # needed by 04a and 13x
python scripts/manual_crop_batch.py --panel middle             # needed by 13x
python scripts/scalebar_adder_longitudinal.py
python scripts/scalebar_adder_cross_section.py
python scripts/collage_maker.py
```

Steps 06 and 07 read `output/cropped_panels/` from step 01 (and 02 if used). The 13 figures read it too, and also need the `right`, `middle` and `left` batches for those scripts to work.

### 2. Clustering

| Step | Script | Input | Main outputs (in `output/…`) |
|---|---|---|---|
| 06 | `hog.py` | cross-section crops, `Labels.csv` | `hog/`: dendrogram, similarity network, `hog_final_layout.csv`, `hog_cluster_labels.csv`, `hog_linkage_matrix.npy` |
| 07 | `image_pca.py` | cross-section crops, `Labels.csv` | `pca/`: `pca_scatter.png`, `pca_cluster_labels.csv` |
| 08 | `helical_info_pca.py` | `Masterhelicalinfo.csv`, `Labels.csv` | `helical_pca/`: `pca_scatter.png`, `pca_cluster_labels.csv`, `helical_pca_loadings.csv` |
| 09 | `helical_info_clustering.py` | `Masterhelicalinfo.csv`, `Labels.csv` | `helical_clustering/`: dendrograms, `helical_cluster_labels.csv`, linkage matrix (`.csv`, `.npy`), leaf order |
| 10 | `emd_density_clustering.py` | `data/RAW density/*.mat or density_64`, `Labels.csv` | `emd_clustering/`: `EMD_3D_cluster.png`, `emd_cluster_labels.csv`, `sinkhorn_distance_matrix.npy` |
| 11 | `emd_umap.py` | output of step 10 | `emd_umap/`: `umap_scatter.png`, `umap_cluster_labels.csv` |

```bash
python scripts/hog.py
python scripts/image_pca.py
python scripts/helical_info_pca.py
python scripts/helical_info_clustering.py
python scripts/emd_density_clustering.py      # slow: pairwise Sinkhorn distances, prints an ETA
python scripts/emd_umap.py
```

The scripts in this section of the pipeline produce cluster tables that share the intrinsic column `EMDB_entry` and number clusters starting at 1. Each script has its own cluster column: `image_cluster` (06), `image_pca_cluster` (07), `helical_info_pca_cluster` (08), `helical_info_cluster` (09), `emd_cluster` (10), `umap_cluster` (11). These cluster tables are used in the next phase of pipeline part 3.

### 3. Comparison and final figures

| Step | Script | What it does | Output (in `output/…`) |
|---|---|---|---|
| 12 | `ari.py` | Pairwise Adjusted Rand Index between the six clusterings (steps 06–11). | `ari/ARI_pairwise_matrix.csv`, `ari/ARI_merged_cluster_labels.csv` |
| 13a | `collage_combiner_cryoem.py` | HOG dendrogram with Cryo-EM, simulated AFM and contact-point images under each leaf. | `figures/HOG_cluster_multiscale_landscape.{png,pdf}` |
| 13b | `collage_combiner_helical_info.py` | Same figure for the helical info clustering. | `figures/helical_info_cluster_multiscale_landscape.{png,pdf}` |
| 13c | `collage_combiner_density_map.py` | Same figure for the density-map clustering. | `figures/density_map_cluster_multiscale_landscape.{png,pdf}` |

```bash
python scripts/ari.py
python scripts/collage_combiner_cryoem.py
python scripts/collage_combiner_helical_info.py
python scripts/collage_combiner_density_map.py
```

The 13 scripts print `Leaf order matches …`; when the dendrograms are drawn, they have the same leaf order as the standalone counterparts saved by the clustering scripts. They save as PNGs at a dpi of 600 on A3, so the figures are very large in the thesis; they had to be rotated horizontally to fit on a page.

## Assumptions worth knowing

- **Scale bars, longitudinal panel (04a):** The scale used by the scale bars is derived from each structure's own pitch (`ceil(1 px/nm × pitch)` data pixels across the panel). The scale is only accurate if the cropped image stretches across exactly that panel width;  so it's important to check a few results against known pitches to see if it is truly representative.
- **Scale bars, cross-section panel (04b):** The script assumes a panel's width is 20 nm (±10 nm; this is Trace_y's default for this parameter). Because of this, it underestimates the scale for structures a little where the max cross-sectional radius is above 10 nm.
- **Scale bars only suit their own panel:** For instance, 04a on cross-section crops or 04b on longitudinal crops creates a scale bar that is incorrect since both calculate the scales of these bars differently that is suited for the dimensions of the script's specific image input.
- **Filenames carry the EMDB ID:** The filename of each input corresponds to the EMDB ID of that specific structure that the files represent; for instance, the multi-panel figures like `emd_18827.png` and density maps like `18827.mat highlight this; specific scripts require these files to obey this nomenclature, so they must be left named as is`.
- **HOG and image PCA also use the "EMD-xxxxx" label** that step 01 uses on each crop, since they read the crops directly, which also have the same naming convention.

## Licence and citation

Code license: MIT. (see more in LICENSE)
Data license: CC0 (see more in data/SOURCES.md)
If you use this work in yours, please cite the repository (Wilson, 2026) and also Trace_y (Xue, 2025)
and the EMDB (wwPDB Consortium, 2024) for good measure; see data/SOURCES.md for the full series of references.
