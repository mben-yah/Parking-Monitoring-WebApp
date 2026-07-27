# -*- coding: utf-8 -*-
"""
augment_dataset.py
──────────────────
Augments Dataset/images/train with:
  1. Zoom-in  — random center crop (scale 0.70–0.92) resized back
  2. Hue shift — random HSV hue rotation (±40 deg)

Also copies the crops from results/english/plate_output_v2/crops/
and generates full-image labels (0 0.5 0.5 1.0 1.0) for each.

Output layout
─────────────
Dataset_Augmented/
  images/
    train/   ← originals + zoom variants + hue variants + crop images
    val/     ← copied as-is from Dataset/images/val
  labels/
    train/   ← original labels + adjusted labels + crop labels
    val/     ← copied as-is from Dataset/labels/val
"""
from __future__ import annotations
import sys, io, random, shutil
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
SRC_IMG  = BASE / "Dataset" / "images" / "train"
SRC_LBL  = BASE / "Dataset" / "labels" / "train"
SRC_VAL_IMG = BASE / "Dataset" / "images" / "val"
SRC_VAL_LBL = BASE / "Dataset" / "labels" / "val"
CROPS    = BASE / "results" / "english" / "plate_output_v2" / "crops"

DST      = BASE / "Dataset_Augmented"
DST_TIMG = DST / "images" / "train"
DST_TLBL = DST / "labels" / "train"
DST_VIMG = DST / "images" / "val"
DST_VLBL = DST / "labels" / "val"

# ─── Augmentation params ──────────────────────────────────────────────────────
ZOOM_MIN, ZOOM_MAX = 0.70, 0.92    # crop this fraction of each side
HUE_SHIFT_MAX      = 40            # max hue shift in degrees (OpenCV: 0-179)
SEED               = 42

random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_labels(lbl_path: Path) -> list[list[float]]:
    """Read YOLO label file → list of [cls, cx, cy, w, h]."""
    if not lbl_path.exists():
        return []
    rows = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append([float(p) for p in parts])
    return rows


def write_labels(lbl_path: Path, rows: list[list[float]]):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        for r in rows:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ─── Zoom augmentation ────────────────────────────────────────────────────────

def zoom_augment(img_bgr: np.ndarray, labels: list[list[float]], scale: float):
    """
    Center-crop `scale` fraction of the image, resize back to original size.
    Adjusts all bounding boxes accordingly.
    Returns (img_out, labels_out) or None if all boxes fall outside crop.
    """
    H, W = img_bgr.shape[:2]

    # Crop window (centered)
    crop_w = int(W * scale)
    crop_h = int(H * scale)
    x_off  = (W - crop_w) // 2
    y_off  = (H - crop_h) // 2

    cropped = img_bgr[y_off:y_off + crop_h, x_off:x_off + crop_w]
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    new_labels = []
    for cls, cx, cy, bw, bh in labels:
        # Convert to pixel coords in original image
        cx_p = cx * W;  cy_p = cy * H
        bw_p = bw * W;  bh_p = bh * H
        x1_p = cx_p - bw_p / 2;  y1_p = cy_p - bh_p / 2
        x2_p = cx_p + bw_p / 2;  y2_p = cy_p + bh_p / 2

        # Shift to crop-local coords
        x1_c = x1_p - x_off;  y1_c = y1_p - y_off
        x2_c = x2_p - x_off;  y2_c = y2_p - y_off

        # Clamp to crop boundaries
        x1_c = clamp(x1_c, 0, crop_w)
        y1_c = clamp(y1_c, 0, crop_h)
        x2_c = clamp(x2_c, 0, crop_w)
        y2_c = clamp(y2_c, 0, crop_h)

        if x2_c <= x1_c or y2_c <= y1_c:
            continue  # box entirely outside crop

        # Renormalize to crop dims (which fill the whole output after resize)
        cx_n = clamp((x1_c + x2_c) / 2 / crop_w)
        cy_n = clamp((y1_c + y2_c) / 2 / crop_h)
        bw_n = clamp((x2_c - x1_c) / crop_w)
        bh_n = clamp((y2_c - y1_c) / crop_h)
        new_labels.append([cls, cx_n, cy_n, bw_n, bh_n])

    if not new_labels:
        return None  # all boxes lost — skip this augmentation

    return resized, new_labels


# ─── Hue augmentation ─────────────────────────────────────────────────────────

def hue_augment(img_bgr: np.ndarray, shift: int):
    """Shift the hue channel by `shift` degrees (OpenCV H range: 0-179)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    for d in [DST_TIMG, DST_TLBL, DST_VIMG, DST_VLBL]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in SRC_IMG.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"Source images : {len(images)}")

    orig_count  = 0
    zoom_count  = 0
    hue_count   = 0
    skip_count  = 0

    for img_path in images:
        stem    = img_path.stem
        suffix  = img_path.suffix
        lbl_src = SRC_LBL / f"{stem}.txt"
        labels  = read_labels(lbl_src)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            skip_count += 1
            continue

        # 1. Copy original
        shutil.copy2(img_path, DST_TIMG / img_path.name)
        write_labels(DST_TLBL / f"{stem}.txt", labels)
        orig_count += 1

        if not labels:
            continue   # no box to adjust — skip augmentations

        # 2. Zoom-in variant
        scale = random.uniform(ZOOM_MIN, ZOOM_MAX)
        result = zoom_augment(img, labels, scale)
        if result is not None:
            z_img, z_lbl = result
            out_name = f"{stem}_zoom{suffix}"
            cv2.imwrite(str(DST_TIMG / out_name), z_img)
            write_labels(DST_TLBL / f"{stem}_zoom.txt", z_lbl)
            zoom_count += 1

        # 3. Hue-shift variant
        shift = random.randint(15, HUE_SHIFT_MAX) * random.choice([-1, 1])
        h_img = hue_augment(img, shift)
        out_name = f"{stem}_hue{suffix}"
        cv2.imwrite(str(DST_TIMG / out_name), h_img)
        write_labels(DST_TLBL / f"{stem}_hue.txt", labels)   # bbox unchanged
        hue_count += 1

    print(f"  Originals copied : {orig_count}")
    print(f"  Zoom variants    : {zoom_count}")
    print(f"  Hue  variants    : {hue_count}")
    print(f"  Skipped          : {skip_count}")

    # ── Add crops (full-image label: plate fills the frame) ────────────────────
    crop_count = 0
    if CROPS.exists():
        crop_imgs = sorted(p for p in CROPS.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\nCrops source     : {len(crop_imgs)} images from {CROPS}")
        for cp in crop_imgs:
            dst_name = f"crop_{cp.name}"
            shutil.copy2(cp, DST_TIMG / dst_name)
            # Full-image label: class=0, cx=0.5, cy=0.5, w=1.0, h=1.0
            lbl_name = f"crop_{cp.stem}.txt"
            write_labels(DST_TLBL / lbl_name, [[0, 0.5, 0.5, 1.0, 1.0]])
            crop_count += 1
        print(f"  Crop images added: {crop_count}")
    else:
        print(f"\n[WARN] Crops folder not found: {CROPS}")

    # ── Copy val split unchanged ────────────────────────────────────────────────
    val_imgs = sorted(p for p in SRC_VAL_IMG.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    val_count = 0
    for vp in val_imgs:
        shutil.copy2(vp, DST_VIMG / vp.name)
        lbl_src = SRC_VAL_LBL / f"{vp.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, DST_VLBL / f"{vp.stem}.txt")
        val_count += 1
    print(f"\nVal images copied: {val_count}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_train = len(list(DST_TIMG.iterdir()))
    print(f"\nDataset_Augmented/images/train : {total_train} images total")
    print(f"Dataset_Augmented/images/val   : {val_count} images")
    print(f"\nOutput: {DST}")


if __name__ == "__main__":
    run()
