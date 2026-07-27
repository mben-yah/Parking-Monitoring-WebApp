# -*- coding: utf-8 -*-
"""
elpd_commercial_pipeline.py
────────────────────────────
Full pipeline for the commercial ELPD dataset (COCO format) at G:\\ELPD\\COCO

Stages:
  1. Parse COCO JSON → convert to YOLO format (xywh → normalized)
  2. Train/Val split (85/15) - only files actually on disk
  3. Data augmentation (brightness, contrast, blur, noise)
  4. Generate elpd_commercial.yaml
  5. Train YOLOv8 (fine-tune from english_train33 best.pt)
  6. Evaluate on val set
  7. Generate visualizations (ground truth grid + predictions grid)
"""

import os, sys, io, json, random, shutil, cv2, numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
ELPD_COCO   = Path(r"G:\ELPD\COCO")
ANN_JSON    = ELPD_COCO / "annotations" / "instances_Train.json"
IMG_SRC     = ELPD_COCO / "images" / "Train"

OUT_DATASET = BASE_DIR / "ELPD_Commercial"
YAML_PATH   = BASE_DIR / "elpd_commercial.yaml"
LOG_DIR     = BASE_DIR / "logs"
LOG_JSON    = LOG_DIR / "elpd_commercial_log.json"

# ── Helper ─────────────────────────────────────────────────────────────────────
def sep(msg=""):
    print(f"\n{'─'*60}\n  {msg}")


# ══════════════════════════════════════════════════════════════
# Stage 1 & 2: Parse COCO → YOLO + Train/Val split
# ══════════════════════════════════════════════════════════════
def prepare_dataset(val_ratio: float = 0.15, seed: int = 42):
    sep("Stage 1–2: COCO → YOLO format + Train/Val split")

    with open(ANN_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup tables
    id_to_fn   = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # Group annotations by image_id
    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    # Filter to images that physically exist on disk
    on_disk = set(p.name for p in IMG_SRC.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    valid_ids = [iid for iid, fn in id_to_fn.items()
                 if fn in on_disk and iid in img_anns]

    print(f"  Total images in JSON   : {len(id_to_fn)}")
    print(f"  Images on disk         : {len(on_disk)}")
    print(f"  Usable (img+ann exist) : {len(valid_ids)}")

    # Train / Val split
    random.seed(seed)
    random.shuffle(valid_ids)
    n_val   = max(1, int(len(valid_ids) * val_ratio))
    val_ids = set(valid_ids[:n_val])
    trn_ids = set(valid_ids[n_val:])

    # Prepare output dirs
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (OUT_DATASET / sub / split).mkdir(parents=True, exist_ok=True)

    def write_split(ids, split):
        written = 0
        for iid in ids:
            fn = id_to_fn[iid]
            w, h = id_to_dims[iid]
            src_img = IMG_SRC / fn
            dst_img = OUT_DATASET / "images" / split / fn
            dst_lbl = OUT_DATASET / "labels" / split / (Path(fn).stem + ".txt")

            shutil.copy2(src_img, dst_img)

            lines = []
            for ann in img_anns[iid]:
                bx, by, bw, bh = ann["bbox"]   # COCO: [x_min, y_min, w, h]
                xc = (bx + bw / 2) / w
                yc = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0,1]
                xc, yc, nw, nh = [max(0.0, min(1.0, v)) for v in (xc, yc, nw, nh)]
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            dst_lbl.write_text("\n".join(lines))
            written += 1
        return written

    n_trn = write_split(trn_ids, "train")
    n_val_w = write_split(val_ids, "val")
    print(f"  Written: {n_trn} train, {n_val_w} val")
    return n_trn, n_val_w


# ══════════════════════════════════════════════════════════════
# Stage 3: Data Augmentation
# ══════════════════════════════════════════════════════════════
def augment_dataset():
    sep("Stage 3: Data Augmentation")
    trn_img = OUT_DATASET / "images" / "train"
    trn_lbl = OUT_DATASET / "labels" / "train"

    orig_imgs = list(trn_img.glob("*.png")) + list(trn_img.glob("*.jpg"))
    aug_count = 0

    for img_p in orig_imgs:
        lbl_p = trn_lbl / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        lbl_text = lbl_p.read_text()

        # Apply 1 augmented variant per original
        aug = img.copy()
        # Brightness + contrast
        alpha = random.uniform(0.8, 1.25)
        beta  = random.randint(-30, 30)
        aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        # Random blur or noise
        choice = random.choice(["blur", "noise", "none"])
        if choice == "blur":
            k = random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (k, k), 0)
        elif choice == "noise":
            noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
            aug   = cv2.add(aug, noise)

        stem_aug  = img_p.stem + "_aug"
        aug_img_p = trn_img / (stem_aug + img_p.suffix)
        aug_lbl_p = trn_lbl / (stem_aug + ".txt")
        cv2.imwrite(str(aug_img_p), aug)
        aug_lbl_p.write_text(lbl_text)
        aug_count += 1

    total = len(list(trn_img.glob("*")))
    print(f"  {len(orig_imgs)} original + {aug_count} augmented = {total} total train images")
    return total


# ══════════════════════════════════════════════════════════════
# Stage 4: YAML
# ══════════════════════════════════════════════════════════════
def generate_yaml():
    sep("Stage 4: Generating YAML")
    content = f"""# ELPD Commercial Dataset (COCO → YOLO)
path: {OUT_DATASET.as_posix()}

train: images/train
val:   images/val

nc: 1
names:
  0: license_plate
"""
    YAML_PATH.write_text(content, encoding="utf-8")
    print(f"  YAML: {YAML_PATH}")


# ══════════════════════════════════════════════════════════════
# Stage 5: Training
# ══════════════════════════════════════════════════════════════
def train_model(epochs: int = 40):
    sep(f"Stage 5: YOLOv8 Training ({epochs} epochs)")
    from ultralytics import YOLO

    # Start from best existing English model
    base_pt = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "english_train33" / "weights" / "best.pt"
    if not base_pt.exists():
        base_pt = BASE_DIR / "yolov8n.pt"
    print(f"  Starting weights: {base_pt}")

    project = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    run_name = "english_train34"

    model = YOLO(str(base_pt))
    model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(project),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    weights = project / run_name / "weights" / "best.pt"
    print(f"  Trained weights: {weights}")
    return weights


# ══════════════════════════════════════════════════════════════
# Stage 6: Evaluation
# ══════════════════════════════════════════════════════════════
def evaluate(weights_path: Path, n_train: int, n_val: int):
    sep("Stage 6: Evaluation")
    from ultralytics import YOLO

    model    = YOLO(str(weights_path))
    val_imgs = list((OUT_DATASET / "images" / "val").glob("*.png")) + \
               list((OUT_DATASET / "images" / "val").glob("*.jpg"))

    hits, scores = 0, []
    for p in val_imgs[:120]:
        res = model.predict(str(p), conf=0.25, verbose=False)
        boxes = res[0].boxes if res[0].boxes is not None else []
        if len(boxes) > 0:
            hits += 1
            scores.append(float(boxes[0].conf[0]))

    det_rate = hits / max(len(val_imgs), 1) * 100
    avg_conf = float(np.mean(scores)) if scores else 0.0

    summary = {
        "dataset": "ELPD Commercial (COCO format)",
        "run":     "english_train34",
        "weights": str(weights_path),
        "n_train": n_train,
        "n_val":   n_val,
        "det_rate_pct": round(det_rate, 2),
        "avg_conf":     round(avg_conf, 4),
        "timestamp":    datetime.now().isoformat(),
    }

    LOG_DIR.mkdir(exist_ok=True)
    LOG_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Detection rate: {det_rate:.1f}%  |  Avg confidence: {avg_conf:.3f}")
    print(f"  Log: {LOG_JSON}")
    return summary


# ══════════════════════════════════════════════════════════════
# Stage 7: Visualizations
# ══════════════════════════════════════════════════════════════
def generate_visualizations(weights_path: Path):
    sep("Stage 7: Generating Visualizations")
    LOG_DIR.mkdir(exist_ok=True)

    # ── 1. Ground Truth Bounding Box Grid ──────────────────────
    trn_imgs = list((OUT_DATASET / "images" / "train").glob("*.png")) + \
               list((OUT_DATASET / "images" / "train").glob("*.jpg"))
    trn_imgs = [p for p in trn_imgs if "_aug" not in p.stem]  # originals only

    if trn_imgs:
        samples = random.sample(trn_imgs, min(9, len(trn_imgs)))
        cols, rows = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 11))
        fig.suptitle("ELPD Commercial — Ground Truth Bounding Box Annotations",
                     fontsize=14, fontweight="bold", color="#1a1a2e")
        fig.patch.set_facecolor("#f8f9fa")

        for ax, img_p in zip(axes.flat, samples):
            img = cv2.imread(str(img_p))
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            ax.imshow(img_rgb)
            ax.set_facecolor("#1a1a2e")

            lbl_p = OUT_DATASET / "labels" / "train" / (img_p.stem + ".txt")
            if lbl_p.exists():
                for line in lbl_p.read_text().splitlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = parts
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        rect = patches.Rectangle(
                            (x1, y1), bw * w, bh * h,
                            linewidth=2.5, edgecolor="#00e676", facecolor="none"
                        )
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5, "plate", color="#00e676",
                                fontsize=8, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.7))
            ax.set_title(img_p.name[:22], fontsize=8, color="#333")
            ax.axis("off")

        for ax in axes.flat[len(samples):]:
            ax.axis("off")

        plt.tight_layout()
        out = LOG_DIR / "elpd_commercial_gt_grid.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  GT grid: {out}")

    # ── 2. Model Prediction Grid ───────────────────────────────
    if weights_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(weights_path))

        val_imgs = list((OUT_DATASET / "images" / "val").glob("*.png")) + \
                   list((OUT_DATASET / "images" / "val").glob("*.jpg"))

        if val_imgs:
            samples = random.sample(val_imgs, min(9, len(val_imgs)))
            fig, axes = plt.subplots(3, 3, figsize=(15, 11))
            fig.suptitle("ELPD Commercial — Model Detection Predictions (english_train34)",
                         fontsize=14, fontweight="bold", color="#1a1a2e")
            fig.patch.set_facecolor("#f8f9fa")

            for ax, img_p in zip(axes.flat, samples):
                img = cv2.imread(str(img_p))
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                ax.imshow(img_rgb)
                ax.set_facecolor("#1a1a2e")

                res   = model.predict(str(img_p), conf=0.25, verbose=False)
                boxes = res[0].boxes if res[0].boxes is not None else []
                label = "No Detection"
                if len(boxes) > 0:
                    b = boxes[0]
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    conf = float(b.conf[0])
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2.5, edgecolor="#2196f3", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f"{conf:.0%}", color="#2196f3",
                            fontsize=8, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.7))
                    label = f"Conf {conf:.2f}"
                ax.set_title(f"{img_p.name[:18]}\n{label}", fontsize=8)
                ax.axis("off")

            for ax in axes.flat[len(samples):]:
                ax.axis("off")

            plt.tight_layout()
            out = LOG_DIR / "elpd_commercial_preds_grid.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Predictions grid: {out}")

    # ── 3. Dataset stats bar chart ─────────────────────────────
    n_trn = len(list((OUT_DATASET / "images" / "train").glob("*")))
    n_val = len(list((OUT_DATASET / "images" / "val").glob("*")))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Train (orig+aug)", "Validation"], [n_trn, n_val],
                  color=["#3b82f6", "#10b981"], edgecolor="#fff", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, [n_trn, n_val]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(v), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title("ELPD Commercial — Dataset Split Overview", fontweight="bold")
    ax.set_ylabel("Image Count")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = LOG_DIR / "elpd_commercial_dataset_stats.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Dataset stats chart: {out}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  ELPD COMMERCIAL PIPELINE")
    print("  COCO → YOLO | Augment | Train | Evaluate | Visualize")
    print("=" * 60)

    # Clean previous run
    if OUT_DATASET.exists():
        print(f"\n  Removing old dataset at {OUT_DATASET} ...")
        shutil.rmtree(OUT_DATASET)

    n_trn, n_val = prepare_dataset(val_ratio=0.15)
    total_trn    = augment_dataset()
    generate_yaml()
    weights      = train_model(epochs=40)
    summary      = evaluate(weights, total_trn, n_val)
    generate_visualizations(weights)

    print("\n" + "=" * 60)
    print("  🎉 PIPELINE COMPLETE!")
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
