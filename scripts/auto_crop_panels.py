"""
Step 1: Crop the top-left (CryoEM) panel from each Trace_y multi-panel figure,
label it with its EMD ID, and stitch the crops into a single collage.

Usage (from the repo root):
    python scripts/01_crop_panels.py
    python scripts/01_crop_panels.py --input-dir my_figures --output-dir my_output --cols 6

Input : data/multipanel_figures/*.png   (filenames must contain the EMD number)
Output: output/cropped_panels/emd_<id>.png
        output/collages/collage.png
"""

import argparse
import glob
import math
import os
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# PATH SETUP (relative to repo root, so it works on any machine)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "multipanel_figures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

TARGET_SIZE = (400, 400)
BORDER_CROP = (10, 10, 390, 390)  # removes panel borders; tune if needed
FONT_SIZE = 18
FONT_CANDIDATES = ["Arial.ttf", "arial.ttf", "DejaVuSans.ttf"]


# -----------------------------
# HELPERS
# -----------------------------
def load_font(size=FONT_SIZE):
    """Try common fonts on macOS/Windows/Linux, fall back to Pillow's default."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_top_left_panel(image_path, debug=False):
    """Detect the top-left rectangular panel and return (PIL image, (x, y, w, h))."""
    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = cv_img.shape[:2]

    # Focus on top-left quadrant (CryoEM panel)
    roi = cv_img[0:h // 2, 0:w // 2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            if ww * hh > (w * h * 0.01):
                candidates.append((x, y, ww, hh))
                if debug:
                    cv2.rectangle(roi, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

    if debug:
        cv2.imshow("Detection", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not candidates:
        return None, None

    # Sort top-left first
    candidates = sorted(candidates, key=lambda r: (r[1], r[0]))
    x, y, ww, hh = candidates[0]

    cropped = cv2.cvtColor(roi[y:y + hh, x:x + ww], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped), (x, y, ww, hh)


def process_images(image_paths, crop_dir, debug=False):
    """Crop, resize, label and save each panel. Returns a list of failures."""
    manual_crop = []
    font = load_font()

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            cropped_img, _ = find_top_left_panel(img_path, debug=debug)

            if cropped_img is None:
                print("  No panel detected")
                manual_crop.append(img_path)
                continue

            cropped_img = cropped_img.resize(TARGET_SIZE).crop(BORDER_CROP)

            # Extract EMD number from filename
            match = re.search(r"(\d+)", os.path.basename(img_path))
            emd_id = match.group(1) if match else "unknown"

            draw = ImageDraw.Draw(cropped_img)
            draw.text((10, 10), f"EMD-{emd_id}", fill="white", font=font)

            save_path = crop_dir / f"emd_{emd_id}.png"
            cropped_img.save(save_path)
            print(f"  Saved -> {save_path}")

        except Exception as e:
            print(f"  Error: {e}")
            manual_crop.append(img_path)

    return manual_crop


def make_collage(crop_dir, collage_dir, cols=8, show=False):
    images = [Image.open(p) for p in sorted(glob.glob(str(crop_dir / "*.png")))]

    if not images:
        print("No images for collage")
        return

    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    collage = Image.new("RGB", (cols * w, rows * h))

    for i, img in enumerate(images):
        collage.paste(img, ((i % cols) * w, (i // cols) * h))

    save_path = collage_dir / "collage.png"
    collage.save(save_path)
    print(f"Collage saved -> {save_path}")

    if show:
        collage.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Folder of Trace_y multi-panel PNGs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder for cropped panels and collage")
    parser.add_argument("--cols", type=int, default=8, help="Collage columns")
    parser.add_argument("--debug", action="store_true",
                        help="Show detected panel boxes (opens a window per image)")
    parser.add_argument("--show", action="store_true",
                        help="Open the collage when finished")
    args = parser.parse_args()

    crop_dir = args.output_dir / "cropped_panels"
    collage_dir = args.output_dir / "collages"
    crop_dir.mkdir(parents=True, exist_ok=True)
    collage_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob.glob(str(args.input_dir / "*.png")))
    if not image_paths:
        raise SystemExit(f"No PNG files found in {args.input_dir}")

    manual_crop = process_images(image_paths, crop_dir, debug=args.debug)
    make_collage(crop_dir, collage_dir, cols=args.cols, show=args.show)

    if manual_crop:
        print("\nImages needing manual cropping:")
        for m in manual_crop:
            print(m)
    else:
        print("\nAll images processed successfully")


if __name__ == "__main__":
    main()
