# Data

This README is specifically to detail all of the inputs in this pipeline. These include the multi-panel figures, density maps and helical tables, which are outputs of **Trace_y**, a program designed for the tracing of amyloid filamnets (MATLAB). It was used to process public cryo-EM density maps of Islet Amyloid Polypeptide  (IAPP) fibrils from the Electron Microscopy Data Bank (EMDB) (https://www.ebi.ac.uk/emdb/). `Labels.csv` holds the per-structure metadata relevant to identifying the structure type.

```
data/
├── multipanel_figures/      one Trace_y figure per structure from AFM simulation (PNG)
├── RAW density/             one 3D density map per structure (<EMDB id>.mat)
├── Labels.csv               Derived from structure metadata found in EMDB
├── Masterhelicalinfo.csv    numeric data produced from AFM simulation within Trace_y 
└── Pitchtable.csv           Produced using the pitch values  and corresponding EMDB ID from masterhelicalinfo.csv 
```

The 31 structures are known by their EMDB number (for example, the `EMDB_entry` column, e.g. `15731`). The 31 IDs appear in all three CSVs in the same order, and the scripts match their corresponding feature values based on that ID.

## multipanel_figures/

Trace_y produces these multi-panel figure PNGs; each structure has one, and the image is named in a manner where the **first run of numbers is that structure's EMDB ID ** (for example `emd_18827.png`). Scripts 01–03 need the images to be called as such to crop these images and label the cropped images with their respective EMDB IDs:

| Panel | Used for |
|---|---|
| top-left: cross-section from the axes-aligned EM map ("Cryo-EM") | HOG (06), image PCA (07), cross-section scale bars (04b) |
| top-right: simulated AFM image of one pitch-long filament | longitudinal scale bars (04a) |
| middle: contact points | cross-section scale bars (04b) and combined figures (13a–13c) |

The manual crop scripts manual_crop.py and manual_crop_batch.py locate panels using a fractional preset selection method (`left`, `middle`, `right`), rather than automatic panel detection, so these figures being cropped need the same layout for this to work. Manual scripts are there mainly to be used when automatic scripts fail to crop specific images or when they fail to crop a panel altogether (e.g. middle panels).

## RAW density/

The raw density maps of the IAPP structures take the form of a MATLAB file; each one represents one structure, and its name corresponds to the structure it represents `<EMDB id>.mat` (for example `15731.mat`). Each contains a MATLAB struct `Raw_Density` where `v` is the 3D density array. Both MATLAB v7.3 (HDF5) and older `.mat` formats are read as some specifically required HDF5. The density maps are used by the  emd_density_clustering.py and emd_umap.py scripts, which take the EMDB ID from the filename to identify them.

The Raw density folder is 1.2 GB because of this; it cannot be uploaded to GitHub due to being too big. As an alternative, I have uploaded the folder to Zenodo (https://doi.org/10.5281/zenodo.24114920). By following the DOI, the original Raw density folder can be downloaded as a zip file and then simply unzipped to use it. Using this folder is not strictly necessary for using the pipeline, as I have also included the resampled density maps that are used by the emd_density_clustering.py and emd_umap.py scripts after processing the raw density maps; these are smaller in size and thus are included in the repository as `data/density_64` and should yield the same results.

## Labels.csv

Per-structure metadata, one row per structure. It's used in steps 6, 7, 8, 9, 10, 11 and 13a–13c. Its purpose is to add informative labels to the figures produced so that the interpretation of the figures and their implications is easier to deduce. They include the following, as can be seen below

| Column | Description | Values |
|---|---|---|
| `EMDB_entry` | EMDB entry number | e.g. `15731` |
| `Polymorph` | Polymorph name; used to colour labels and legends | `Polymorph 1`, `Polymorph 2`, `Polymorph 2PF`, `Polymorph 2PF-C`, `Polymorph 2PF-L`, `Polymorph 2PF-P`, `Polymorph 3`, `Polymorph 3PF`, `Polymorph 3PF-CU`, `Polymorph 4PF-CU`, `Polymorph 4PF-LJ`, `Polymorph 4PF-LU`, `LL`, `LLU`, `LLUU`, `SF`, `SS`, `TF` |
| `Sequence_Variant` | Peptide sequence variant; `WT` is shown without a suffix in plot labels | `WT`, `A25P`, `S20G`, `S28P`, `S29P` |
| `Origin` | Source of the fibrils | `Homo sapiens`, `Synthetic` |
| `PH` | pH of the sample | `6.0`, `6.8`, `7.0`, `7.4` |

## Masterhelicalinfo.csv

This is the main CSV; it contains the Helical feature numeric values per structure. The dataset is an amalgamation of the helicalinfo struct, which is a mixed dataset (double and character data) generated after AFM simulation from Trace_y. Read by scripts helical_info_pca and helical_info_clustering, which use the 6 columns marked below with a tick (✔) (rows with a missing value in any of them were dropped). IMPORTANT TO NOTE: file must be **UTF-8**.

| Column | Description | Used |
|---|---|---|
| `EMDB_entry` | EMDB entry number | ✔ |
| `Rise_nm` | Helical rise (nm) | ✔ |
| `Twist_deg` | Helical twist (degrees) | ✔ |
| `Symmetry` | Symmetry of the structure | ✔ |
| `Pitch_nm` | Helical pitch (nm) | |
| `COD_minus_Pitch` | | |
| `DPF` | | |
| `CsaEM` | Cross-sectional area from the EM map | ✔ |
| `Min_Height_nm` | Minimum height (nm) | |
| `Max_Height_nm` | Maximum height (nm) | ✔ |
| `Mean_Height_nm` | Mean height (nm) | |
| `Roundness_Index` | Roundness of the cross-section | ✔ |
| `Sequence_Variant` | Sequence variant (same as in `Labels.csv`) | |

## Pitchtable.csv

| Column | Description |
|---|---|
| `EMDB_entry` | EMDB entry number |
| `Pitch_nm` | Helical pitch (nm) |

The Pitchtable.csv is specifically used by the scalebar_adder_longitudinal script to put accurate scalebars on the longitudinal AFM simulation panel on the upper right of multi-panel figures. The values are the same as the `Pitch_nm` and respective 'EMDB_entry' columns of `Masterhelicalinfo.csv` since this table was made from it. Similar to the'Masterhelicalinfo.csv`, this file also starts with a UTF-8 byte-order mark, which is handled.

 ## Sources and licences

The 31 structures come from EMDB, which is released under a Creative Commons (CC0) licence. The exact entries, their associated publications, the licences of the files & references (Trace_y and EMDB) can be found in the SOURCES.md file.
