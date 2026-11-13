import argparse
import math
import re
import platform
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# -----------------------------
# PATHS (relative to the repo root, so they work on any machine)
# -----------------------------
# NOTE: the scale calculation below is written for crops of the TOP-RIGHT
# "Simulated AFM image of one pitch long filament" panel specifically,
# NOT the cross-section panel used in Scale_Bar_adder.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# Choose which cropped-image folder to use with --input-dir, e.g.
#   python scripts/04a_add_scale_bars_longitudinal.py --input-dir output/manual_cropped_batch/right
#   python scripts/04a_add_scale_bars_longitudinal.py --input-dir output/cropped_panels
parser = argparse.ArgumentParser(description="Add scale bars to cropped panel images.")
parser.add_argument("--input-dir", type=Path,
                    default=OUTPUT_DIR / "manual_cropped_batch" / "right",
                    help="Folder of cropped PNGs (default: output/manual_cropped_batch/right)")
parser.add_argument("--output-dir", type=Path,
                    default=OUTPUT_DIR / "scalebar_images" / "longitudinal",
                    help="Where to save the scale-barred images (default: output/scalebar_images/longitudinal)")
args = parser.parse_args()

cropped_path = args.input_dir
save_dir = args.output_dir
pitch_csv_path = DATA_DIR / "Pitchtable.csv"                   # Trace_y output: columns EMDB_entry, Pitch_nm
master_info_csv_path = DATA_DIR / "Masterhelicalinfo.csv"  # Trace_y output (not used by this script)

# Font: Arial on Windows/macOS, DejaVu Sans elsewhere (falls back to Pillow's default if missing)
if platform.system() == "Windows":
    font_path = r"C:\Windows\Fonts\Arial.ttf"
elif platform.system() == "Darwin":
    font_path = "/Library/Fonts/Arial.ttf"
else:
    font_path = "DejaVuSans.ttf"

save_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CONSTANTS
# -----------------------------
# This panel's width in DATA pixels is pitch-dependent:
#   imgSize_1pitch = ceil(pixel_density * pitch)   -- from CompileHelicalFilamentInfo.m
# pixel_density is confirmed as 1 px/nm from the Trace_y figure captions.
# Since the panel is rendered at a roughly fixed on-screen/exported width
# regardless of how many data pixels it actually contains, the true nm-per
# rendered-pixel ratio must be calculated per structure using each
# structure's own pitch, together with that image's actual cropped width.
PIXEL_DENSITY = 1     # px/nm
SCALE_BAR_NM = 20     # longer bar than the cross-section panel, since this
                       # view spans a much larger physical range (one full
                       # helical pitch, ~49-168 nm across the dataset).
                       # Adjust if 20 nm still looks too small/large.

# -----------------------------
# LOAD PITCH TABLE (once, before the loop)
# -----------------------------
pitch_df = pd.read_csv(pitch_csv_path, encoding="utf-8-sig")
print(pitch_df.columns.tolist())
pitch_lookup = dict(zip(pitch_df["EMDB_entry"].astype(str), pitch_df["Pitch_nm"]))

# -----------------------------
# DIAGNOSTIC CHECK
# -----------------------------
print(f"Looking in: {cropped_path}")
print(f"Exists: {cropped_path.exists()}")
found_files = list(cropped_path.glob("*.png"))
print(f"Found {len(found_files)} png files")
if found_files:
    print(f"Example: {found_files[0].name}")

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for img_path in found_files:

    img_name = img_path.name

    # Extract EMD number (matches e.g. "emd_10670" or just "10670")
    match = re.search(r'(\d+)', img_name)
    if not match:
        print(f"Cannot extract EMD number from {img_name}")
        continue

    emd_number = match.group(1)

    if emd_number not in pitch_lookup:
        print(f"No pitch found for EMD {emd_number}")
        continue

    pitch = pitch_lookup[emd_number]
    img_size_1pitch = math.ceil(PIXEL_DENSITY * pitch)

    # Load cropped image
    cropped_img = Image.open(img_path)
    cropped_width = cropped_img.size[0]  # rendered panel width in px

    # Derive true scale for this specific structure
    pixels_per_nm = cropped_width / img_size_1pitch
    scale_bar_length_px = SCALE_BAR_NM * pixels_per_nm

    # Draw scale bar
    draw = ImageDraw.Draw(cropped_img)

    margin = 20
    bar_height = 5

    x_start = cropped_img.size[0] - margin - scale_bar_length_px
    y_start = cropped_img.size[1] - margin - bar_height
    x_end = cropped_img.size[0] - margin
    y_end = cropped_img.size[1] - margin

    draw.rectangle([x_start, y_start, x_end, y_end], fill="white")

    text = f"{SCALE_BAR_NM} nm"
    try:
        font = ImageFont.truetype(font_path, 18)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    x_text = x_start + (scale_bar_length_px - text_width) / 2
    y_text = y_start - 20

    draw.text((x_text, y_text), text, fill="white", font=font)

    # Save
    save_path = save_dir / f"scalebar_{img_name}"
    cropped_img.save(save_path)
    print(f"Saved: {save_path.name} (pixels_per_nm={pixels_per_nm:.3f}, pitch={pitch:.1f}nm)")


print("Done.")
