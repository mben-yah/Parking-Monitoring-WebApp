# -*- coding: utf-8 -*-
"""
crop_plates.py
──────────────
Runs YOLO on every image in a folder (or a single image), draws the
bounding-box annotation on the full image, crops the plate region,
then saves both to dedicated output folders:

    output_dir/
        annotated/   <- full image with bbox + label drawn on it
        crops/       <- tight crop of the plate region only

Usage
─────
    # Process all images in a folder
    python crop_plates.py --input "C:\\path\\to\\images" --output "C:\\path\\to\\out"

    # Single image
    python crop_plates.py --input "C:\\path\\to\\plate.jpg"

    # Use a specific weights file
    python crop_plates.py --input "..." --weights "runs/detect/train9/weights/best.pt"
"""

from __future__ import annotations

import argparse
import sys
import warnings
import io
from pathlib import Path

warnings.filterwarnings("ignore")

# Force UTF-8 console output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

# ─── Default paths ────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DEFAULT_WEIGHTS  = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "arabic_train" / "weights" / "best.pt"
FALLBACK_WEIGHTS = BASE_DIR / "runs" / "detect" / "train9" / "weights" / "best.pt"
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ─── Visual style ─────────────────────────────────────────────────────────────
BOX_COLOR   = (0, 229, 255)   # cyan
TEXT_COLOR  = (0, 229, 255)
BOX_THICK   = 3
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.9
FONT_THICK  = 2
CONF_THRESH = 0.10            # low to catch anything
PADDING     = 6               # px padding around crop


def _resolve_weights(weights_arg: str | None) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Weights not found: {p}")
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    if FALLBACK_WEIGHTS.exists():
        print(f"[WARN] Arabic weights not found, using fallback: {FALLBACK_WEIGHTS}")
        return FALLBACK_WEIGHTS
    raise FileNotFoundError(
        "No weights found. Pass --weights explicitly or train first.\n"
        f"  Looked at: {DEFAULT_WEIGHTS}\n"
        f"  Fallback:  {FALLBACK_WEIGHTS}"
    )


def process_image(
    img_path: Path,
    model: YOLO,
    out_annotated: Path,
    out_crops: Path,
) -> dict:
    """
    Run YOLO on one image, save annotated full image and plate crop.

    Returns
    -------
    dict with keys: filename, detected, confidence, bbox, annotated_path, crop_path
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [SKIP] Cannot read image: {img_path.name}")
        return {"filename": img_path.name, "detected": False}

    img_h, img_w = img_bgr.shape[:2]

    # ── YOLO inference ────────────────────────────────────────────────────────
    results = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)
    boxes   = results[0].boxes

    annotated = img_bgr.copy()

    if boxes is None or len(boxes) == 0:
        print(f"  [--] {img_path.name} — no plate detected")
        # Save annotated as-is (no box drawn) so we still have the original
        ann_path = out_annotated / f"{img_path.stem}_annotated{img_path.suffix}"
        cv2.imwrite(str(ann_path), annotated)
        return {"filename": img_path.name, "detected": False, "annotated_path": str(ann_path)}

    # ── Pick the highest-confidence box ───────────────────────────────────────
    confs    = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    conf     = float(confs[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # Clamp to image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)

    print(f"  [OK] {img_path.name} — conf={conf:.3f}  bbox=[{x1},{y1},{x2},{y2}]")

    # ── Draw annotation on full image ─────────────────────────────────────────
    # Bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

    # Label background + text
    label      = f"plate {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    lbl_y1     = max(y1 - th - 10, 0)
    lbl_y2     = lbl_y1 + th + 8
    cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 8, lbl_y2), BOX_COLOR, -1)
    cv2.putText(annotated, label, (x1 + 4, lbl_y2 - 4),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)

    # ── Save annotated full image ─────────────────────────────────────────────
    ann_path = out_annotated / f"{img_path.stem}_annotated{img_path.suffix}"
    cv2.imwrite(str(ann_path), annotated)

    # ── Crop the plate region (with small padding) ────────────────────────────
    cx1 = max(x1 - PADDING, 0)
    cy1 = max(y1 - PADDING, 0)
    cx2 = min(x2 + PADDING, img_w)
    cy2 = min(y2 + PADDING, img_h)
    crop = img_bgr[cy1:cy2, cx1:cx2]

    crop_path = out_crops / f"{img_path.stem}_crop{img_path.suffix}"
    cv2.imwrite(str(crop_path), crop)

    return {
        "filename":       img_path.name,
        "detected":       True,
        "confidence":     conf,
        "bbox":           [int(x1), int(y1), int(x2), int(y2)],
        "annotated_path": str(ann_path),
        "crop_path":      str(crop_path),
    }


def run(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    weights:    str | None = None,
) -> list[dict]:
    """
    Main entry point. Accepts a single image or a directory of images.
    """
    input_path = Path(input_path)

    # Collect images
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"No images found in {input_path}")
            return []
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
        images = [input_path]
    else:
        raise ValueError(f"input_path must be an image file or a directory: {input_path}")

    # Output dirs
    base_out   = Path(output_dir) if output_dir else input_path.parent / "plate_output"
    out_annotated = base_out / "annotated"
    out_crops     = base_out / "crops"
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    w     = _resolve_weights(weights)
    model = YOLO(str(w))
    print(f"\nWeights : {w}")
    print(f"Images  : {len(images)}")
    print(f"Output  : {base_out}")
    print(f"  annotated/ -> full images with drawn bbox")
    print(f"  crops/     -> tight plate crops")
    print("-" * 55)

    records = []
    detected = 0
    for img_path in images:
        rec = process_image(img_path, model, out_annotated, out_crops)
        records.append(rec)
        if rec.get("detected"):
            detected += 1

    print("-" * 55)
    print(f"Done. {detected}/{len(images)} plates detected.")
    print(f"Annotated images : {out_annotated}")
    print(f"Plate crops      : {out_crops}")

    return records


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO plate detector — annotate and crop")
    parser.add_argument("--input",   "-i", required=True,
                        help="Path to an image file or a folder of images")
    parser.add_argument("--output",  "-o", default=None,
                        help="Output folder (default: <input_dir>/plate_output/)")
    parser.add_argument("--weights", "-w", default=None,
                        help="Path to YOLO .pt weights file (auto-detected if omitted)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output,
        weights=args.weights,
    )
