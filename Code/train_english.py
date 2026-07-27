# -*- coding: utf-8 -*-
"""
train_english.py
----------------
Trains a YOLOv8 model on the English license-plate dataset.

Usage
-----
    python train_english.py
    python train_english.py --epochs 100 --model yolov8m.pt --batch 8
"""
import argparse
import sys
import io
from pathlib import Path

# Force UTF-8 console on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ultralytics import YOLO

BASE_DIR  = Path(__file__).parent
DATA_YAML = BASE_DIR / "dataset.yaml"


def train(
    model_name: str = "yolov8n.pt",
    epochs:     int = 100,
    imgsz:      int = 640,
    batch:      int = 16,
    patience:   int = 20,
    project:    str = "runs/detect",
    name:       str = "english_train",
    data_yaml: Path | None = None,
):
    data_yaml = data_yaml or DATA_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found at {data_yaml}")

    print("=" * 60)
    print("  English Plate Detector — YOLOv8 Training")
    print(f"  Model   : {model_name}")
    print(f"  Data    : {data_yaml}")
    print(f"  Epochs  : {epochs}  (early stop patience={patience})")
    print(f"  Img size: {imgsz}   Batch: {batch}")
    print(f"  Output  : {project}/{name}/")
    print("=" * 60)

    model = YOLO(model_name)   # downloads pretrained weights if not cached

    results = model.train(
        data      = str(data_yaml),
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        patience  = patience,
        project   = project,
        name      = name,
        # Augmentation
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 5.0,
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
        # Optimiser
        lr0       = 0.01,
        lrf       = 0.01,
        optimizer = "SGD",
        verbose   = True,
    )

    best_pt = Path(project) / name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    if best_pt.exists():
        print(f"  Training complete!")
        print(f"  Best weights : {best_pt.resolve()}")
        sz = best_pt.stat().st_size // 1024
        print(f"  File size    : {sz} KB")
    else:
        print("  Training ended — weights not found at expected path.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on English plate dataset")
    parser.add_argument("--data",    default=None,              help="Path to dataset YAML (default: dataset.yaml)")
    parser.add_argument("--model",   default="yolov8n.pt", help="Base model (yolov8n/s/m/l.pt)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--patience",type=int, default=20)
    parser.add_argument("--name",    default="english_train", help="Run folder name")
    args = parser.parse_args()

    data_yaml = Path(args.data) if args.data else BASE_DIR / "dataset.yaml"

    train(
        model_name = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        patience   = args.patience,
        name       = args.name,
        data_yaml  = data_yaml,
    )
