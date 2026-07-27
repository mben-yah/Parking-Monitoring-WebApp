"""
train_arabic4.py
────────────────
Train arabic_train4 with YOLOv8s backbone on the augmented Arabic dataset.
Uses a higher-quality baseline model (yolov8s) for better detection accuracy.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR  = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
DATA_YAML = BASE_DIR / "ar_dataset_augmented.yaml"

# Check dataset
train_imgs = BASE_DIR / "Ar_Dataset_Augmented" / "images" / "train"
val_imgs   = BASE_DIR / "Ar_Dataset_Augmented" / "images" / "val"
print(f"Train images: {len(list(train_imgs.glob('*.*')))}")
print(f"Val   images: {len(list(val_imgs.glob('*.*')))}")
print(f"YAML:  {DATA_YAML}")

# Start with YOLOv8s (small = better than nano, still fast enough on CPU)
model = YOLO("yolov8s.pt")

results = model.train(
    data       = str(DATA_YAML),
    epochs     = 60,
    imgsz      = 640,
    batch      = 8,
    project    = str(BASE_DIR / "runs" / "detect" / "runs" / "detect"),
    name       = "arabic_train4",
    exist_ok   = False,
    patience   = 15,          # early stop if no improvement
    # Augmentation – conservative (30% scale matches user preference)
    degrees    = 5.0,         # small rotation
    scale      = 0.3,         # ±30% scale
    shear      = 2.0,
    perspective= 0.0003,
    hsv_h      = 0.015,
    hsv_s      = 0.5,
    hsv_v      = 0.3,
    flipud     = 0.0,         # plates don't flip vertically
    fliplr     = 0.0,         # plates read left-to-right — no flip
    mosaic     = 0.6,
    mixup      = 0.1,
    copy_paste = 0.0,
    # Optimizer
    optimizer  = "AdamW",
    lr0        = 0.001,
    lrf        = 0.01,
    warmup_epochs = 3,
    verbose    = True,
)

print("\n=== TRAINING COMPLETE ===")
print(f"Best weights: {results.save_dir}/weights/best.pt")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
