#!/usr/bin/env python3
"""Convert YOLO-format annotations into cropped plate images + labels for OCR training.

Assumptions:
- YOLO label files are plain text with lines: <class> <xc> <yc> <w> <h> (normalized)
- Input images are in `images_dir`; labels with same basename (but .txt) in `labels_dir`.
- Output: crop images under out_dir/images/ and a labels.txt with lines:
    relative/path/to/crop.png\tPLATE_TEXT

This script will only work when your label files contain the plate text in a companion CSV or the YOLO label is accompanied by a text file with the transcription.
If your dataset stores plate text elsewhere, adapt the `get_plate_text` function.
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out-crops", required=True)
    p.add_argument("--transcriptions-csv", default=None,
                   help="Optional CSV mapping image basename -> plate text (basename,plate_text)")
    p.add_argument("--min-area", type=int, default=20)
    return p.parse_args()


def load_transcriptions(csv_path):
    mapping = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
    return mapping


def get_plate_text(image_basename, mapping):
    # Fallback: mapping by basename, else use basename as label (not ideal)
    return mapping.get(image_basename, image_basename)


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_crops)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    mapping = load_transcriptions(args.transcriptions_csv)

    labels_out = []

    for img_path in images_dir.glob("**/*.*"):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            # skip images without labels
            continue
        # Read YOLO labels: choose the first annotation (assume single plate per image)
        with open(lbl_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if not lines:
            continue
        # We expect the plate transcription in a separate mapping. Many datasets encode bbox only.
        # Use first bbox for cropping
        first = lines[0].split()
        if len(first) < 5:
            continue
        try:
            cls, xc, yc, w, h = first[:5]
            xc = float(xc); yc = float(yc); w = float(w); h = float(h)
        except Exception:
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)
        # Clamp
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        if (x2 - x1) * (y2 - y1) < args.min_area:
            continue
        crop = img.crop((x1, y1, x2, y2))
        # Name the crop by original basename
        out_name = f"{img_path.stem}_{x1}_{y1}_{x2}_{y2}.png"
        out_path = out_images / out_name
        crop.save(out_path)
        plate_text = get_plate_text(img_path.stem, mapping)
        labels_out.append(f"{out_path.relative_to(out_dir)\t}{plate_text}")

    # Write labels.txt
    with open(out_dir / "labels.txt", "w", encoding="utf-8") as fh:
        for ln in labels_out:
            fh.write(ln + "\n")

    print(f"Wrote {len(labels_out)} crops to {out_images} and labels.txt")


if __name__ == '__main__':
    main()
