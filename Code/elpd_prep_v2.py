# -*- coding: utf-8 -*-
"""
elpd_prep_v2.py
───────────────
End-to-end pipeline:
  1. COCO → YOLO conversion  (archive\COCO)
  2. Train / Val split       (85 / 15)
  3. Augmentation            (crop · scale · translation — NO colour changes)
  4. YOLOv8 fine-tune        (from english_train33 best.pt, 40 epochs)

Output:  Code\ELPD_Commercial_v2\
Logs:    Code\logs\elpd_v2_training.log
"""

import json, shutil, random, time, math, sys, io
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
COCO_ROOT   = Path(r"C:\Users\Mohamed Walid\Desktop\archive\COCO")
COCO_JSON   = COCO_ROOT / "annotations" / "instances_Train.json"
COCO_IMGS   = COCO_ROOT / "images" / "Train"

CODE        = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
OUT_ROOT    = CODE / "ELPD_Commercial_v2"
LOG_DIR     = CODE / "logs"
LOG_FILE    = LOG_DIR / "elpd_v2_training.log"

TRAIN_IMG   = OUT_ROOT / "images" / "train"
TRAIN_LBL   = OUT_ROOT / "labels" / "train"
VAL_IMG     = OUT_ROOT / "images" / "val"
VAL_LBL     = OUT_ROOT / "labels" / "val"

BEST_PT     = CODE / "runs" / "detect" / "runs" / "detect" / "english_train33" / "weights" / "best.pt"
YAML_PATH   = CODE / "elpd_commercial_v2.yaml"
RUN_PROJECT = CODE / "runs" / "detect" / "runs" / "detect"
RUN_NAME    = "elpd_commercial_train2"

VAL_RATIO   = 0.15
EPOCHS      = 40
BATCH       = 16
IMGSZ       = 640

LOG_DIR.mkdir(exist_ok=True)

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}]  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Check if dataset is already built & populated
dataset_exists = (
    TRAIN_IMG.exists() 
    and len(list(TRAIN_IMG.glob("*.png")) + list(TRAIN_IMG.glob("*.jpg"))) > 1000
)

if dataset_exists:
    log("=" * 70)
    log("Existing dataset found (ELPD_Commercial_v2). Skipping conversion & augmentation.")
else:
    # ══════════════════════════════════════════════════════════════════════════════
    # 1. COCO → YOLO conversion
    # ══════════════════════════════════════════════════════════════════════════════
    log("=" * 70)
    log("STEP 1 — COCO → YOLO conversion")

    with open(COCO_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_img  = {img["id"]: img for img in coco["images"]}
    img_anns   = {}
    for a in coco["annotations"]:
        img_anns.setdefault(a["image_id"], []).append(a)

    on_disk = {p.name for p in COCO_IMGS.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    valid   = [
        iid for iid, img in id_to_img.items()
        if img["file_name"] in on_disk and iid in img_anns
    ]
    log(f"Total in JSON: {len(coco['images'])}  |  On disk: {len(on_disk)}  |  Valid annotated: {len(valid)}")

    def bbox_to_yolo(bbox, img_w, img_h):
        x, y, w, h = bbox
        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return (
            max(0.0, min(1.0, xc)),
            max(0.0, min(1.0, yc)),
            max(0.0, min(1.0, nw)),
            max(0.0, min(1.0, nh)),
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # 2. Train / Val split
    # ══════════════════════════════════════════════════════════════════════════════
    log("STEP 2 — Train / Val split")

    random.shuffle(valid)
    n_val   = max(1, int(len(valid) * VAL_RATIO))
    val_ids = set(valid[:n_val])
    trn_ids = set(valid[n_val:])
    log(f"Train: {len(trn_ids)}  |  Val: {len(val_ids)}")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        d.mkdir(parents=True)

    def write_sample(iid, img_dir, lbl_dir):
        info  = id_to_img[iid]
        fname = info["file_name"]
        iw, ih = info["width"], info["height"]
        src   = COCO_IMGS / fname
        shutil.copy(src, img_dir / fname)
        lines = []
        for a in img_anns[iid]:
            xc, yc, nw, nh = bbox_to_yolo(a["bbox"], iw, ih)
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        stem = Path(fname).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

    for iid in trn_ids: write_sample(iid, TRAIN_IMG, TRAIN_LBL)
    for iid in val_ids:  write_sample(iid, VAL_IMG,   VAL_LBL)
    log(f"Written train: {len(list(TRAIN_IMG.iterdir()))} images")
    log(f"Written val  : {len(list(VAL_IMG.iterdir()))} images")

    # ══════════════════════════════════════════════════════════════════════════════
    # 3. Augmentation — crop · scale · translation  (NO colour changes)
    # ══════════════════════════════════════════════════════════════════════════════
    log("STEP 3 — Augmentation (crop · scale · translation)")

    def read_yolo_labels(lbl_path):
        labels = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    labels.append((int(parts[0]), *map(float, parts[1:])))
        return labels

    def write_yolo_labels(lbl_path, labels):
        lbl_path.write_text(
            "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels)
        )

    def clip_bbox(xc, yc, w, h):
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w  = max(0.01, min(1.0, w))
        h  = max(0.01, min(1.0, h))
        return xc, yc, w, h

    def aug_crop(img, labels):
        H, W = img.shape[:2]
        ratio = random.uniform(0.70, 0.90)
        crop_w = int(W * ratio)
        crop_h = int(H * ratio)
        x0 = random.randint(0, W - crop_w)
        y0 = random.randint(0, H - crop_h)
        x1, y1 = x0 + crop_w, y0 + crop_h

        cropped = img[y0:y1, x0:x1]
        out     = cv2.resize(cropped, (W, H))

        new_labels = []
        for c, xc, yc, bw, bh in labels:
            bx0 = (xc - bw / 2) * W
            by0 = (yc - bh / 2) * H
            bx1 = (xc + bw / 2) * W
            by1 = (yc + bh / 2) * H
            ibx0 = max(bx0, x0); iby0 = max(by0, y0)
            ibx1 = min(bx1, x1); iby1 = min(by1, y1)

            if ibx1 <= ibx0 or iby1 <= iby0:
                continue

            nxc = ((ibx0 + ibx1) / 2 - x0) / crop_w
            nyc = ((iby0 + iby1) / 2 - y0) / crop_h
            nw  = (ibx1 - ibx0) / crop_w
            nh  = (iby1 - iby0) / crop_h
            new_labels.append((c, *clip_bbox(nxc, nyc, nw, nh)))

        return out, new_labels if new_labels else None

    def aug_scale(img, labels):
        H, W = img.shape[:2]
        factor = random.uniform(0.75, 1.25)
        nW, nH = int(W * factor), int(H * factor)
        resized = cv2.resize(img, (nW, nH))

        out = np.full((H, W, 3), 114, dtype=np.uint8)
        paste_x = (W - nW) // 2
        paste_y = (H - nH) // 2

        src_x0 = max(0, -paste_x)
        src_y0 = max(0, -paste_y)
        src_x1 = src_x0 + min(nW, W - max(0, paste_x))
        src_y1 = src_y0 + min(nH, H - max(0, paste_y))

        dst_x0 = max(0, paste_x)
        dst_y0 = max(0, paste_y)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        out[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]

        new_labels = []
        for c, xc, yc, bw, bh in labels:
            rxc = xc * nW + paste_x
            ryc = yc * nH + paste_y
            rbw = bw * nW
            rbh = bh * nH

            bx0 = max(0, rxc - rbw / 2)
            by0 = max(0, ryc - rbh / 2)
            bx1 = min(W, rxc + rbw / 2)
            by1 = min(H, ryc + rbh / 2)

            if bx1 <= bx0 or by1 <= by0:
                continue

            nxc = (bx0 + bx1) / 2 / W
            nyc = (by0 + by1) / 2 / H
            nw  = (bx1 - bx0) / W
            nh  = (by1 - by0) / H
            new_labels.append((c, *clip_bbox(nxc, nyc, nw, nh)))

        return out, new_labels if new_labels else None

    def aug_translate(img, labels):
        H, W = img.shape[:2]
        tx = int(random.uniform(-0.10, 0.10) * W)
        ty = int(random.uniform(-0.10, 0.10) * H)

        M   = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(img, M, (W, H),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

        new_labels = []
        for c, xc, yc, bw, bh in labels:
            nxc = xc + tx / W
            nyc = yc + ty / H

            if not (0 < nxc < 1 and 0 < nyc < 1):
                continue

            nxc = max(bw / 2, min(1 - bw / 2, nxc))
            nyc = max(bh / 2, min(1 - bh / 2, nyc))
            new_labels.append((c, *clip_bbox(nxc, nyc, bw, bh)))

        return out, new_labels if new_labels else None

    AUGMENTORS = [
        ("crop",      aug_crop),
        ("scale",     aug_scale),
        ("translate", aug_translate),
    ]

    orig_images = list(TRAIN_IMG.glob("*.png")) + list(TRAIN_IMG.glob("*.jpg"))
    added = 0
    skipped = 0

    for img_path in orig_images:
        stem   = img_path.stem
        lbl_path = TRAIN_LBL / f"{stem}.txt"
        labels = read_yolo_labels(lbl_path)

        if not labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for suffix, fn in AUGMENTORS:
            aug_img, aug_lbl = fn(img, labels)
            if aug_img is None or not aug_lbl:
                skipped += 1
                continue

            out_name = f"{stem}_{suffix}.png"
            cv2.imwrite(str(TRAIN_IMG / out_name), aug_img)
            write_yolo_labels(TRAIN_LBL / f"{stem}_{suffix}.txt", aug_lbl)
            added += 1

    total_train = len(list(TRAIN_IMG.iterdir()))
    total_val   = len(list(VAL_IMG.iterdir()))
    log(f"Augmentation done — added {added} images (skipped {skipped})")
    log(f"Final train: {total_train}  |  val: {total_val}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Write YAML
# ══════════════════════════════════════════════════════════════════════════════
log("STEP 4 — Writing dataset YAML")

yaml_content = f"""# ELPD Commercial v2 — crop/scale/translation augmentation
path: {OUT_ROOT.as_posix()}
train: images/train
val:   images/val
nc: 1
names:
  0: license_plate
"""
YAML_PATH.write_text(yaml_content, encoding="utf-8")
log(f"YAML: {YAML_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. YOLOv8 training
# ══════════════════════════════════════════════════════════════════════════════
log("STEP 5 — YOLOv8 training")
log(f"Epochs   : {EPOCHS}  |  Batch: {BATCH}  |  Imgsz: {IMGSZ}")
log("=" * 70)

from ultralytics import YOLO

resume_pt = RUN_PROJECT / RUN_NAME / "weights" / "last.pt"

import json as _json
epoch_start = time.time()

def on_epoch_end(trainer):
    epoch   = trainer.epoch + 1
    total   = trainer.epochs
    now     = time.time()
    elapsed = now - epoch_start
    avg_s   = elapsed / epoch
    rem     = avg_s * (total - epoch)
    eta     = datetime.now() + timedelta(seconds=rem)

    m       = trainer.metrics or {}
    mAP50   = m.get("metrics/mAP50(B)",    "—")
    mAP5095 = m.get("metrics/mAP50-95(B)", "—")
    prec    = m.get("metrics/precision(B)", "—")
    rec     = m.get("metrics/recall(B)",    "—")

    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

    log(
        f"Epoch {epoch:>2}/{total}  |  "
        f"mAP50={fmt(mAP50)}  mAP50-95={fmt(mAP5095)}  "
        f"P={fmt(prec)}  R={fmt(rec)}  |  "
        f"Elapsed={str(timedelta(seconds=int(elapsed)))}  "
        f"ETA={eta.strftime('%H:%M:%S')}  (~{str(timedelta(seconds=int(rem)))} left)"
    )

    snap = {
        "epoch": epoch, "total": total,
        "mAP50":    round(float(mAP50), 4)   if isinstance(mAP50,   float) else None,
        "mAP50_95": round(float(mAP5095), 4) if isinstance(mAP5095, float) else None,
        "elapsed_s": round(elapsed, 1),
        "eta": eta.isoformat(),
    }
    (LOG_DIR / "elpd_v2_epoch_latest.json").write_text(
        _json.dumps(snap, indent=2), encoding="utf-8"
    )

if resume_pt.exists():
    log(f"Resuming training from checkpoint: {resume_pt}")
    model = YOLO(str(resume_pt))
    model.add_callback("on_train_epoch_end", on_epoch_end)
    results = model.train(resume=True)
else:
    log(f"Starting new training from base weights: {BEST_PT}")
    model = YOLO(str(BEST_PT))
    model.add_callback("on_train_epoch_end", on_epoch_end)
    results = model.train(
        data     = str(YAML_PATH),
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        project  = str(RUN_PROJECT),
        name     = RUN_NAME,
        exist_ok = True,
        verbose  = True,
        patience = 15,
    )

# ── Final summary ──────────────────────────────────────────────────────────────
total_time = time.time() - epoch_start
best_pt    = RUN_PROJECT / RUN_NAME / "weights" / "best.pt"
log("=" * 70)
log("TRAINING COMPLETE!")
log(f"Total time  : {str(timedelta(seconds=int(total_time)))}")
log(f"Best weights: {best_pt}")

(LOG_DIR / "elpd_v2_training_final.json").write_text(
    _json.dumps({
        "status": "complete",
        "total_time_s": round(total_time, 1),
        "best_weights": str(best_pt),
        "finished_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8"
)
log(f"Log: {LOG_FILE}")
