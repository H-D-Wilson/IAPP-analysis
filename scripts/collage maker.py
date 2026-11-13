"""
Step 5: Build the final-figure collage from a folder of images.

Unlike the quick-check collage that 01_crop_panels.py makes (to verify the
crops look right), this is the collage intended for the final figure: every
image is resized to a uniform thumbnail and tiled into a grid.

Usage (from the repo root):
    python scripts/05_make_collage.py
    python scripts/05_make_collage.py --cols 4 --thumb-size 400
    python scripts/05_make_collage.py --input-dir path/to/images --output final.png

Input : output/scalebar_images/longitudinal/*.png  (or any folder via --input-dir,
        e.g. output/scalebar_images/cross_section)
Output: output/collages/final_collage.png
"""

import argparse
import math
from pathlib import Path

from PIL import Image

# -----------------------------
# PATH SETUP (relative to repo root)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "output" / "scalebar_images" / "longitudinal"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "collages" / "final_collage.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Folder of PNG images to tile")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output file path")
    parser.add_argument("--cols", type=int, default=3, help="Number of grid columns")
    parser.add_argument("--thumb-size", type=int, default=300,
                        help="Thumbnail width/height in pixels (images are resized to a square)")
    args = parser.parse_args()

    # Sorted for a reproducible order; skip the output file in case it lives in the input folder
    paths = sorted(p for p in args.input_dir.glob("*.png")
                   if p.resolve() != args.output.resolve())
    if not paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    thumb_size = (args.thumb_size, args.thumb_size)
    imgs = [Image.open(p).convert("RGB").resize(thumb_size) for p in paths]

    rows = math.ceil(len(imgs) / args.cols)
    collage = Image.new("RGB", (args.cols * thumb_size[0], rows * thumb_size[1]), color="black")

    for i, img in enumerate(imgs):
        x = (i % args.cols) * thumb_size[0]
        y = (i // args.cols) * thumb_size[1]
        collage.paste(img, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    collage.save(args.output)
    print(f"Collage of {len(imgs)} images ({args.cols} cols x {rows} rows) saved to {args.output}")


if __name__ == "__main__":
    main()
