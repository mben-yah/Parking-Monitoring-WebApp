# -*- coding: utf-8 -*-
"""
augment_arabic.py
─────────────────
Augments Ar_Dataset_Split/images/train with:
  1. Zoom-in  — 30% center crop (scale = 0.70) resized back
  2. Hue shift — random ±40° HSV hue rotation

Output: Ar_Dataset_Augmented/
"""
from __future__ import annotations
import sys, io, random, shutil
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

BASE       = Path(__file__).parent
SRC_IMG    = BASE / "Ar_Dataset_Split" / "images" / "train"
SRC_LBL    = BASE / "Ar_Dataset_Split" / "labels" / "train"
SRC_V_IMG  = BASE / "Ar_Dataset_Split" / "images" / "val"
SRC_V_LBL  = BASE / "Ar_Dataset_Split" / "labels" / "val"

DST        = BASE / "Ar_Dataset_Augmented"
DST_TI     = DST / "images" / "train"
DST_TL     = DST / "labels" / "train"
DST_VI     = DST / "images" / "val"
DST_VL     = DST / "labels" / "val"

# 30 % zoom-in = keep 70 % of the image
ZOOM_SCALE = 0.70
HUE_MAX    = 40
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
SEED       = 42
random.seed(SEED); np.random.seed(SEED)


def read_labels(p: Path) -> list[list[float]]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        pts = line.strip().split()
        if len(pts) == 5:
            rows.append([float(x) for x in pts])
    return rows

def write_labels(p: Path, rows: list[list[float]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def zoom_aug(img: np.ndarray, labels: list[list[float]], scale: float):
    H, W  = img.shape[:2]
    cw, ch = int(W * scale), int(H * scale)
    xo, yo = (W - cw) // 2, (H - ch) // 2
    cropped = img[yo:yo+ch, xo:xo+cw]
    resized = cv2.resize(cropped, (W, H))
    new_lbl = []
    for cls, cx, cy, bw, bh in labels:
        cx_p = cx*W; cy_p = cy*H; bw_p = bw*W; bh_p = bh*H
        x1 = clamp(cx_p - bw_p/2 - xo, 0, cw)
        y1 = clamp(cy_p - bh_p/2 - yo, 0, ch)
        x2 = clamp(cx_p + bw_p/2 - xo, 0, cw)
        y2 = clamp(cy_p + bh_p/2 - yo, 0, ch)
        if x2 <= x1 or y2 <= y1:
            continue
        new_lbl.append([cls,
            clamp((x1+x2)/2/cw), clamp((y1+y2)/2/ch),
            clamp((x2-x1)/cw),   clamp((y2-y1)/ch)])
    return (resized, new_lbl) if new_lbl else None

def hue_aug(img: np.ndarray, shift: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def run():
    for d in [DST_TI, DST_TL, DST_VI, DST_VL]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in SRC_IMG.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Source images : {len(images)}")
    orig = zoom = hue = skip = 0

    for img_path in images:
        stem   = img_path.stem
        suffix = img_path.suffix
        labels = read_labels(SRC_LBL / f"{stem}.txt")
        img    = cv2.imread(str(img_path))
        if img is None:
            skip += 1; continue

        # 1. Original
        shutil.copy2(img_path, DST_TI / img_path.name)
        write_labels(DST_TL / f"{stem}.txt", labels)
        orig += 1

        if not labels:
            continue

        # 2. Zoom (fixed 30 %)
        r = zoom_aug(img, labels, ZOOM_SCALE)
        if r:
            zi, zl = r
            cv2.imwrite(str(DST_TI / f"{stem}_zoom{suffix}"), zi)
            write_labels(DST_TL / f"{stem}_zoom.txt", zl)
            zoom += 1

        # 3. Hue shift
        shift = random.randint(15, HUE_MAX) * random.choice([-1, 1])
        cv2.imwrite(str(DST_TI / f"{stem}_hue{suffix}"), hue_aug(img, shift))
        write_labels(DST_TL / f"{stem}_hue.txt", labels)
        hue += 1

    # Copy val unchanged
    val_imgs = sorted(p for p in SRC_V_IMG.iterdir() if p.suffix.lower() in IMG_EXTS)
    for vp in val_imgs:
        shutil.copy2(vp, DST_VI / vp.name)
        lbl = SRC_V_LBL / f"{vp.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, DST_VL / f"{vp.stem}.txt")

    total = len(list(DST_TI.iterdir()))
    print(f"  Originals : {orig}")
    print(f"  Zoom      : {zoom}")
    print(f"  Hue       : {hue}")
    print(f"  Skipped   : {skip}")
    print(f"  Val imgs  : {len(val_imgs)}")
    print(f"\nTotal train : {total} images → {DST}")

if __name__ == "__main__":
    run()
