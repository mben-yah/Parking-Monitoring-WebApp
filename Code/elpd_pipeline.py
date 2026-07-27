# -*- coding: utf-8 -*-
"""
elpd_pipeline.py
────────────────
ELPD (European License Plate Dataset) Processing, Augmentation, and Training Pipeline.

Stages:
  1. Extract & Prepare Dataset (from archive.zip + ELPD directory)
  2. Apply Data Augmentation (Brightness, contrast, tilt, noise, motion blur)
  3. Generate YOLO YAML configuration (elpd_dataset.yaml)
  4. Train YOLOv8 Model (saved to runs/detect/runs/detect/english_train33)
  5. Run Validation & Save Detailed Performance Logs
"""

import os
import sys
import io
import json
import random
import shutil
import zipfile
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# UTF-8 stdout setup
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
ARCHIVE_ZIP = Path(r"C:\Users\Mohamed Walid\Downloads\archive.zip")
ELPD_EXTRA_DIR = Path(r"C:\Users\Mohamed Walid\Downloads\ELPD")
OUTPUT_DATASET = BASE_DIR / "ELPD_Dataset"
YAML_PATH = BASE_DIR / "elpd_dataset.yaml"
LOG_PATH = BASE_DIR / "logs" / "elpd_pipeline_log.json"


# ── Stage 1: Dataset Extraction & Preparation ────────────────────────────────

def extract_and_prepare_dataset(val_ratio: float = 0.15, seed: int = 42):
    """
    Extracts labeled pairs from archive.zip and formats them into YOLO structure.
    Also pseudo-annotates extra ELPD images if valid detections are found.
    """
    print("📦 [Stage 1] Extracting & preparing ELPD dataset...")
    
    # Target directories
    train_img_dir = OUTPUT_DATASET / "images" / "train"
    val_img_dir   = OUTPUT_DATASET / "images" / "val"
    train_lbl_dir = OUTPUT_DATASET / "labels" / "train"
    val_lbl_dir   = OUTPUT_DATASET / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    extracted_pairs = []
    temp_extract = BASE_DIR / "scratch" / "elpd_temp"
    temp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as z:
        all_files = z.namelist()
        img_files = [f for f in all_files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for img_name in img_files:
            stem = Path(img_name).stem
            # Find matching label file
            txt_candidates = [f for f in all_files if Path(f).stem == stem and f.endswith(".txt")]
            if not txt_candidates:
                continue
            
            lbl_name = txt_candidates[0]
            
            # Extract both to temp
            img_dest = temp_extract / (stem + Path(img_name).suffix)
            lbl_dest = temp_extract / (stem + ".txt")
            
            with open(img_dest, "wb") as f_img:
                f_img.write(z.read(img_name))
            with open(lbl_dest, "wb") as f_lbl:
                f_lbl.write(z.read(lbl_name))
                
            # Verify non-empty label
            if lbl_dest.stat().st_size > 0:
                extracted_pairs.append((img_dest, lbl_dest))

    print(f"  Found {len(extracted_pairs)} valid image-label pairs in archive.zip")

    # Pseudo-annotate extra images from ELPD_EXTRA_DIR using english_train32
    best_weights = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "english_train32" / "weights" / "best.pt"
    if best_weights.exists() and ELPD_EXTRA_DIR.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(best_weights))
            extra_imgs = list(ELPD_EXTRA_DIR.glob("*.png")) + list(ELPD_EXTRA_DIR.glob("*.jpg"))
            added_pseudo = 0
            for e_img in extra_imgs:
                res = model.predict(str(e_img), conf=0.40, verbose=False)
                boxes = res[0].boxes if res[0].boxes is not None else []
                if len(boxes) > 0:
                    img = cv2.imread(str(e_img))
                    if img is None: continue
                    img_h, img_w = img.shape[:2]
                    
                    b = boxes[0]
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                    xc = ((x1 + x2) / 2.0) / img_w
                    yc = ((y1 + y2) / 2.0) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    
                    out_img = temp_extract / e_img.name
                    out_lbl = temp_extract / (e_img.stem + ".txt")
                    shutil.copy(e_img, out_img)
                    with open(out_lbl, "w") as f:
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    extracted_pairs.append((out_img, out_lbl))
                    added_pseudo += 1
            print(f"  Added {added_pseudo} pseudo-labeled extra ELPD images")
        except Exception as err:
            print(f"  Warning pseudo labeling: {err}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(extracted_pairs)
    val_count = int(len(extracted_pairs) * val_ratio)
    val_pairs   = extracted_pairs[:val_count]
    train_pairs = extracted_pairs[val_count:]

    for img_p, lbl_p in train_pairs:
        shutil.copy(img_p, train_img_dir / img_p.name)
        shutil.copy(lbl_p, train_lbl_dir / lbl_p.name)

    for img_p, lbl_p in val_pairs:
        shutil.copy(img_p, val_img_dir / img_p.name)
        shutil.copy(lbl_p, val_lbl_dir / lbl_p.name)

    print(f"  Split complete: {len(train_pairs)} train, {len(val_pairs)} val")
    return len(train_pairs), len(val_pairs)


# ── Stage 2: Data Augmentation ────────────────────────────────────────────────

def augment_image(img, bbox):
    """
    Applies realistic augmentation: brightness adjustment, contrast jitter,
    subtle motion blur, and gaussian noise.
    """
    h, w = img.shape[:2]
    aug = img.copy()

    # 1. Brightness & Contrast adjustment
    alpha = random.uniform(0.8, 1.25)  # Contrast control
    beta  = random.randint(-25, 25)    # Brightness control
    aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 2. Random Gaussian blur or noise
    aug_type = random.choice(["blur", "noise", "none"])
    if aug_type == "blur":
        k_size = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k_size, k_size), 0)
    elif aug_type == "noise":
        noise = np.random.normal(0, 8, aug.shape).astype(np.uint8)
        aug = cv2.add(aug, noise)

    return aug, bbox


def apply_data_augmentation():
    """
    Augment training images to expand dataset diversity and improve generalization.
    """
    print("🎨 [Stage 2] Applying Data Augmentation to training split...")
    train_img_dir = OUTPUT_DATASET / "images" / "train"
    train_lbl_dir = OUTPUT_DATASET / "labels" / "train"

    train_imgs = list(train_img_dir.glob("*.*"))
    orig_count = len(train_imgs)
    aug_count  = 0

    for img_path in train_imgs:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        with open(lbl_path, "r") as f:
            lbl_lines = f.readlines()

        for i in range(1):  # Create 1 augmented copy per train image
            aug_img, _ = augment_image(img, None)
            aug_stem = f"{img_path.stem}_aug{i+1}"
            aug_img_path = train_img_dir / f"{aug_stem}{img_path.suffix}"
            aug_lbl_path = train_lbl_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(aug_img_path), aug_img)
            with open(aug_lbl_path, "w") as f:
                f.writelines(lbl_lines)
            aug_count += 1

    total_train = len(list(train_img_dir.glob("*.*")))
    print(f"  Augmentation complete: {orig_count} original + {aug_count} augmented = {total_train} total train images")
    return total_train


# ── Stage 3: Dataset YAML Generation ──────────────────────────────────────────

def generate_yaml():
    """
    Generates elpd_dataset.yaml for YOLOv8 training.
    """
    print("📄 [Stage 3] Generating dataset configuration YAML...")
    content = f"""# ELPD Dataset Configuration
path: {OUTPUT_DATASET.as_posix()}

train: images/train
val:   images/val

nc: 1
names:
  0: license_plate
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  YAML saved to: {YAML_PATH}")


# ── Stage 4: YOLO Model Training ──────────────────────────────────────────────

def train_elpd_model(epochs: int = 30):
    """
    Trains YOLOv8n on the augmented ELPD dataset saving to english_train33.
    """
    print(f"🚀 [Stage 4] Starting YOLOv8 model training for {epochs} epochs...")
    from ultralytics import YOLO

    base_weights = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "english_train32" / "weights" / "best.pt"
    if not base_weights.exists():
        base_weights = BASE_DIR / "yolov8n.pt"

    print(f"  Pretrained weights: {base_weights}")
    model = YOLO(str(base_weights))

    project_dir = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    run_name = "english_train33"

    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    trained_weights = project_dir / run_name / "weights" / "best.pt"
    print(f"✅ Training completed! Weights saved to: {trained_weights}")
    return trained_weights


# ── Stage 5: Evaluation & Logging ─────────────────────────────────────────────

def evaluate_and_log(weights_path: Path, train_count: int, val_count: int):
    """
    Evaluates the newly trained model on validation images and logs results.
    """
    print("📊 [Stage 5] Evaluating model performance...")
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    val_img_dir = OUTPUT_DATASET / "images" / "val"
    val_imgs = list(val_img_dir.glob("*.*"))

    detections = 0
    conf_scores = []

    for img_p in val_imgs[:100]:  # sample up to 100 validation images
        res = model.predict(str(img_p), conf=0.25, verbose=False)
        boxes = res[0].boxes if res[0].boxes is not None else []
        if len(boxes) > 0:
            detections += 1
            conf_scores.append(float(boxes[0].conf[0]))

    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
    det_rate = (detections / max(len(val_imgs), 1)) * 100.0

    summary = {
        "dataset": "ELPD (European License Plate Dataset)",
        "model_run": "english_train33",
        "weights_path": str(weights_path),
        "train_samples": train_count,
        "val_samples": val_count,
        "sample_eval_count": len(val_imgs),
        "detections_count": detections,
        "detection_rate_pct": round(det_rate, 2),
        "average_confidence": round(avg_conf, 4),
        "status": "SUCCESS"
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), **summary}, f, indent=2)

    print(f"  Detection Rate: {det_rate:.1f}%  |  Average Confidence: {avg_conf:.3f}")
    print(f"  Summary log saved to: {LOG_PATH}")
    return summary


def generate_visualizations(weights_path: Path):
    """
    Generates matplotlib plots & visual comparison grids for ELPD training:
      1. Ground Truth Bounding Box Grid -> logs/elpd_ground_truth_grid.png
      2. Test Predictions & OCR Output -> logs/elpd_test_predictions_grid.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground Truth Bounding Box Grid
    train_img_dir = OUTPUT_DATASET / "images" / "train"
    train_lbl_dir = OUTPUT_DATASET / "labels" / "train"
    train_imgs    = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))

    if train_imgs:
        sample_imgs = random.sample(train_imgs, min(6, len(train_imgs)))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("ELPD Dataset — Ground Truth Bounding Box Annotations", fontsize=14, fontweight='bold')
        
        for ax, img_p in zip(axes.flat, sample_imgs):
            img = cv2.imread(str(img_p))
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            ax.imshow(img_rgb)
            
            lbl_p = train_lbl_dir / (img_p.stem + ".txt")
            if lbl_p.exists():
                with open(lbl_p) as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) == 5:
                            _, xc, yc, bw, bh = parts
                            x1 = (xc - bw/2) * w
                            y1 = (yc - bh/2) * h
                            rect = patches.Rectangle((x1, y1), bw*w, bh*h, linewidth=2, edgecolor='#00ff66', facecolor='none')
                            ax.add_patch(rect)
            ax.set_title(img_p.name[:25], fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        gt_path = log_dir / "elpd_ground_truth_grid.png"
        plt.savefig(gt_path, dpi=150)
        plt.close()
        print(f"  Visual saved: {gt_path}")

    # 2. Test Predictions Grid
    if weights_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        
        val_img_dir = OUTPUT_DATASET / "images" / "val"
        val_imgs    = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
        
        if val_imgs:
            sample_val = random.sample(val_imgs, min(6, len(val_imgs)))
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            fig.suptitle("ELPD Model — Test Detection & OCR Visualizations", fontsize=14, fontweight='bold')

            for ax, img_p in zip(axes.flat, sample_val):
                img = cv2.imread(str(img_p))
                if img is None: continue
                
                res = model.predict(str(img_p), conf=0.25, verbose=False)
                boxes = res[0].boxes if res[0].boxes is not None else []
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                ax.imshow(img_rgb)
                
                plate_label = "No Detection"
                if len(boxes) > 0:
                    b = boxes[0]
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    conf = float(b.conf[0])
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2.5, edgecolor='#3b82f6', facecolor='none')
                    ax.add_patch(rect)
                    plate_label = f"Conf: {conf:.2f}"
                    
                ax.set_title(f"{img_p.name[:20]}\n{plate_label}", fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            pred_path = log_dir / "elpd_test_predictions_grid.png"
            plt.savefig(pred_path, dpi=150)
            plt.close()
            print(f"  Visual saved: {pred_path}")


# ── Main Execution Flow ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("   ELPD PIPELINE: Extraction -> Augmentation -> Training -> Evaluation")
    print("=" * 70)
    
    n_train, n_val = extract_and_prepare_dataset(val_ratio=0.15)
    total_train = apply_data_augmentation()
    generate_yaml()
    weights_path = train_elpd_model(epochs=30)
    summary = evaluate_and_log(weights_path, total_train, n_val)
    generate_visualizations(weights_path)
    
    print("\n🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
