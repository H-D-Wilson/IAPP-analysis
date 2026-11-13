"""
Step 1b: Manually crop a chosen panel from a Trace_y multi-panel figure.

Use this when 01_crop_panels.py fails to detect a panel (e.g. middle panels).
The crop is defined as proportions of the image (left, top, right, bottom),
so it works for figures of any resolution.

Usage (from the repo root):
    python scripts/02_manual_crop.py emd_18827.png
    python scripts/02_manual_crop.py emd_18827.png --panel middle
    python scripts/02_manual_crop.py emd_18827.png --ratio 0.46 0.51 0.53 0.7
    python scripts/02_manual_crop.py path/to/figure.png --output-dir my_output

If the image path is not found as given, it is looked up in
data/multipanel_figures/.

Output: output/cropped_panels/emd_<id>.png
        (same folder and naming as 01_crop_panels.py, so a manual crop replaces
        a failed or unwanted automatic crop and is included in the collage)
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# PATH SETUP (relative to repo root)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "multipanel_figures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "cropped_panels"

TARGET_SIZE = (400, 400)
FONT_SIZE = 18
FONT_CANDIDATES = ["Arial.ttf", "arial.ttf", "DejaVuSans.ttf"]

# Crop boxes as fractions of the image: (left, top, right, bottom)
PANEL_PRESETS = {
    "left": (0.12, 0.13, 0.17, 0.28),
    "middle": (0.46, 0.51, 0.53, 0.70),
    "right": (0.82, 0.10, 0.90, 0.30),
}
MARGIN = 0.03  # 3% buffer added around the crop box


def load_font(size=FONT_SIZE):
    """Try common fonts on macOS/Windows/Linux, fall back to Pillow's default."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_image_path(image_arg, input_dir):
    path = Path(image_arg)
    if path.exists():
        return path
    candidate = input_dir / image_arg
    if candidate.exists():
        return candidate
    raise SystemExit(f"Image not found: {image_arg} (also looked in {input_dir})")


def crop_panel(img, ratio, margin=MARGIN):
    width, height = img.size
    left = max(0, int((ratio[0] - margin) * width))
    top = max(0, int((ratio[1] - margin) * height))
    right = min(width, int((ratio[2] + margin) * width))
    bottom = min(height, int((ratio[3] + margin) * height))
    return img.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Multi-panel figure (path, or filename in the input dir)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--panel", choices=PANEL_PRESETS, default="right",
                       help="Preset crop box (default: right)")
    group.add_argument("--ratio", type=float, nargs=4,
                       metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                       help="Custom crop box as fractions of the image, e.g. 0.46 0.51 0.53 0.7")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="Open the result when finished")
    args = parser.parse_args()

    ratio = tuple(args.ratio) if args.ratio else PANEL_PRESETS[args.panel]
    image_path = resolve_image_path(args.image, args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    print(f"Processing {image_path}, size: {img.size}, crop ratio: {ratio}")

    cropped_img = crop_panel(img, ratio).resize(TARGET_SIZE)

    # Extract EMD number from filename
    match = re.search(r"(\d+)", image_path.name)
    emdb_number = match.group(1) if match else "Unknown"

    # Add label
    draw = ImageDraw.Draw(cropped_img)
    draw.text((10, 10), f"EMD-{emdb_number}", fill="white", font=load_font())

    save_path = args.output_dir / f"emd_{emdb_number}.png"
    cropped_img.save(save_path)
    print(f"Saved cropped image to {save_path}")

    if args.show:
        cropped_img.show()


if __name__ == "__main__":
    main()
