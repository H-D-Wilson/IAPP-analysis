"""
Step 1c: Manually crop the SAME panel from EVERY Trace_y multi-panel figure.

Use this when 01_crop_panels.py fails on a large number of figures (e.g. it
cannot detect middle panels at all). For a single figure, use
02_manual_crop.py instead.

Usage (from the repo root):
    python scripts/03_manual_crop_batch.py
    python scripts/03_manual_crop_batch.py --panel left
    python scripts/03_manual_crop_batch.py --ratio 0.46 0.51 0.53 0.7
    python scripts/03_manual_crop_batch.py --input-dir my_figures

Input : data/multipanel_figures/*.png
Output: output/manual_cropped_batch/<panel>/emd_<id>.png

Output goes to its own folder (not output/cropped_panels/) so it never
overwrites automatic crops that were fine. Copy the crops you want into
output/cropped_panels/ afterwards. Each panel gets its own subfolder, so
batches for different panels do not overwrite each other.
"""

import argparse
import glob
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# PATH SETUP (relative to repo root)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "multipanel_figures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "manual_cropped_batch"

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--panel", choices=PANEL_PRESETS, default="middle",
                       help="Preset crop box (default: middle)")
    group.add_argument("--ratio", type=float, nargs=4,
                       metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                       help="Custom crop box as fractions of the image")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the output folder (default: output/manual_cropped_batch/<panel>)")
    args = parser.parse_args()

    ratio = tuple(args.ratio) if args.ratio else PANEL_PRESETS[args.panel]
    label = "custom" if args.ratio else args.panel
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(p) for p in sorted(glob.glob(str(args.input_dir / "*.png")))]
    if not image_paths:
        raise SystemExit(f"No PNG files found in {args.input_dir}")

    print(f"Crop ratio: {ratio}\nOutput folder: {output_dir}\n")

    font = load_font()
    failed = []

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            print(f"Processing {image_path.name}, size: {img.size}")

            cropped_img = crop_panel(img, ratio).resize(TARGET_SIZE)

            # Extract EMD number from filename
            match = re.search(r"(\d+)", image_path.name)
            emdb_number = match.group(1) if match else "Unknown"

            draw = ImageDraw.Draw(cropped_img)
            draw.text((10, 10), f"EMD-{emdb_number}", fill="white", font=font)

            save_path = output_dir / f"emd_{emdb_number}.png"
            cropped_img.save(save_path)
            print(f"  Saved -> {save_path}")

        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            failed.append(image_path)

    print(f"\nDone: {len(image_paths) - len(failed)} of {len(image_paths)} images cropped.")
    for f in failed:
        print(f"  Failed: {f}")


if __name__ == "__main__":
    main()
