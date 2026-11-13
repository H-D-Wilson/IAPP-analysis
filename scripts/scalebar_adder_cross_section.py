import argparse
import re
import platform
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# PATHS (relative to the repo root, so they work on any machine)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

precrop_path = DATA_DIR / "multipanel_figures"   # original Trace_y figures (not used by this script)

# Choose which cropped-image folder to use with --input-dir, e.g.
#   python scripts/04b_add_scale_bars_cross_section.py
#   python scripts/04b_add_scale_bars_cross_section.py --input-dir output/cropped_panels
parser = argparse.ArgumentParser(description="Add scale bars to cropped cross-section panel images.")
parser.add_argument("--input-dir", type=Path,
                    default=OUTPUT_DIR / "cropped_panels",
                    help="Folder of cropped PNGs (default: output/cropped_panels)")
parser.add_argument("--output-dir", type=Path,
                    default=OUTPUT_DIR / "scalebar_images" / "cross_section",
                    help="Where to save the scale-barred images (default: output/scalebar_images/cross_section)")
args = parser.parse_args()

cropped_path = args.input_dir
save_dir = args.output_dir

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
# These crops come from the TOP-LEFT "Cross-section from axes-aligned EM map"
# panel, not the AFM simulation panel. That panel's MATLAB code plots data in
# true nm but sets xlim to [-xz_lims xz_lims]/10 -- an extra, inconsistent
# division by 10 not present in the equivalent bottom-left panel using the
# same xz_lims variable. The true displayed width is therefore xz_lims nm on
# each side, not the mislabelled "1" shown on the axis ticks.
#
# xz_lims defaults to 10 (i.e. +/-10 nm, 20 nm total) unless a structure's
# csr_max exceeds 10 nm, in which case xz_lims = ceil(csr_max/5)*5. Without
# per-structure csr_max values on hand, we use the default floor value as a
# single dataset-wide constant. This will be exact for most structures and
# will modestly underestimate true scale only for structures with an
# unusually large cross-sectional radius (csr_max > 10 nm) -- worth noting
# as an assumption in the methods write-up.
TRUE_PANEL_WIDTH_NM = 20   # +/-10 nm, matching xz_lims default/floor value
SCALE_BAR_NM = 10

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
# PROCESS IMAGES (single loop)
# -----------------------------
for img_path in found_files:

    img_name = img_path.name

    # Extract EMD number (matches e.g. "emd_10670" or just "10670")
    match = re.search(r'(\d+)', img_name)
    if not match:
        print(f"Cannot extract EMD number from {img_name}")
        continue

    emd_number = match.group(1)

    # Load cropped image
    cropped_img = Image.open(img_path)
    cropped_width = cropped_img.size[0]  # rendered panel width in px

    # Derive scale using the fixed cross-section panel width (see note above)
    pixels_per_nm = cropped_width / TRUE_PANEL_WIDTH_NM
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

    text = "10 nm"
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
    print(f"Saved: {save_path.name} (pixels_per_nm={pixels_per_nm:.3f})")

print("Done.")
