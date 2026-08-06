# -*- coding: utf-8 -*-
"""
train_from_kaggle.py
────────────────────
Automated Kaggle Dataset Downloader & Local YOLOv8 Trainer for Magnetite Vision.

Usage Example:
  C:\Anaconda\envs\yolo_env\python.exe train_from_kaggle.py --dataset "andrewmvd/car-plate-detection" --epochs 40 --name "elpd_kaggle_train"

Or pass a full Kaggle dataset URL:
  C:\Anaconda\envs\yolo_env\python.exe train_from_kaggle.py --dataset "https://www.kaggle.com/datasets/andrewmvd/car-plate-detection"
"""

import argparse
import os
import sys
import shutil
import re
import ssl
from pathlib import Path

# Bypass SSL certificate verification for local machine network environments
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "Dataset_Kaggle"
YAML_PATH = BASE_DIR / "dataset_kaggle.yaml"


def extract_kaggle_handle(dataset_or_url: str) -> str:
    """Extract Kaggle dataset handle (owner/dataset-name) from URL or handle string."""
    dataset_or_url = dataset_or_url.strip()
    match = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", dataset_or_url)
    if match:
        return match.group(1)
    return dataset_or_url


def download_from_kaggle(handle_or_url: str) -> Path:
    """Download dataset directly from Kaggle using official kagglehub."""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if not hasattr(requests.Session, '_ssl_disabled_send'):
            requests.Session._ssl_disabled_send = requests.Session.send
            def _unverified_send(self, request, **kwargs):
                kwargs['verify'] = False
                return self._ssl_disabled_send(request, **kwargs)
            requests.Session.send = _unverified_send
    except Exception:
        pass

    try:
        import kagglehub
    except ImportError:
        print("[+] Installing kagglehub helper package...")
        os.system(f'"{sys.executable}" -m pip install kagglehub')
        import kagglehub

    handle = extract_kaggle_handle(handle_or_url)
    print(f"[+] Downloading Kaggle dataset: {handle}...")
    
    # Download dataset via kagglehub
    downloaded_path_str = kagglehub.dataset_download(handle)
    downloaded_path = Path(downloaded_path_str)
    
    print(f"[+] Kaggle dataset downloaded to: {downloaded_path}")
    return downloaded_path


def setup_dataset_structure(source_dir: Path) -> Path:
    """Copy and organize Kaggle dataset into standard local YOLO folder structure."""
    if DATASETS_DIR.exists():
        print(f"[+] Cleaning existing Kaggle dataset directory: {DATASETS_DIR}")
        shutil.rmtree(DATASETS_DIR)

    print(f"[+] Preparing dataset files in {DATASETS_DIR}...")
    shutil.copytree(source_dir, DATASETS_DIR, dirs_exist_ok=True)
    return DATASETS_DIR


def generate_dataset_yaml(dataset_dir: Path) -> Path:
    """Auto-detect image & label paths and generate dataset_kaggle.yaml."""
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"

    if not images_train.exists():
        # Fallback to root or images folder
        images_train = dataset_dir / "images" if (dataset_dir / "images").exists() else dataset_dir

    if not images_val.exists():
        images_val = images_train

    yaml_content = f"""# Magnetite Vision Kaggle Dataset Configuration
path: {dataset_dir.as_posix()}
train: {images_train.as_posix()}
val: {images_val.as_posix()}

# Classes
nc: 1
names: ['license_plate']
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[+] Created YOLO dataset YAML configuration at: {YAML_PATH}")
    return YAML_PATH


def run_yolo_training(yaml_path: Path, epochs: int = 40, batch: int = 16, imgsz: int = 640, run_name: str = "elpd_kaggle_train"):
    """Start local YOLOv8 fine-tuning training using downloaded Kaggle dataset."""
    from ultralytics import YOLO

    # Check for existing base model weights (ELPD Commercial v2)
    base_weights = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "elpd_commercial_train2" / "weights" / "best.pt"
    if base_weights.exists():
        print(f"[+] Loading base model weights: {base_weights}")
        model = YOLO(str(base_weights))
    else:
        print("[+] Base model weights not found, using default pretrained 'yolov8n.pt'")
        model = YOLO("yolov8n.pt")

    print("=" * 70)
    print(f"  Starting Local YOLOv8 Training from Kaggle Dataset: {run_name}")
    print(f"  Dataset YAML : {yaml_path}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch}")
    print(f"  Image size   : {imgsz}x{imgsz}")
    print("=" * 70)

    project_dir = BASE_DIR / "runs" / "detect" / "runs" / "detect"

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        patience=15,
        save=True
    )

    trained_weights = project_dir / run_name / "weights" / "best.pt"
    print("=" * 70)
    print(f"  Training Complete! New Trained Weights Saved To:")
    print(f"  👉 {trained_weights}")
    print("=" * 70)
    return trained_weights


def main():
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle & Train YOLOv8 Locally")
    parser.add_argument("--dataset", type=str, help="Kaggle dataset handle (e.g., owner/dataset-name) or URL")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (default: 640)")
    parser.add_argument("--name", type=str, default="elpd_kaggle_train", help="Training run name")

    args = parser.parse_args()

    dataset_input = args.dataset
    if not dataset_input:
        print("=" * 70)
        print("  Magnetite Vision — Kaggle Dataset Trainer")
        print("=" * 70)
        dataset_input = input("Paste Kaggle dataset handle (e.g., andrewmvd/car-plate-detection) or URL: ").strip()

    if not dataset_input:
        print("[-] Error: Kaggle dataset handle or URL is required!")
        sys.exit(1)

    # Step 1: Download from Kaggle
    source_dir = download_from_kaggle(dataset_input)

    # Step 2: Unpack & setup dataset
    dataset_dir = setup_dataset_structure(source_dir)

    # Step 3: Generate dataset.yaml
    yaml_file = generate_dataset_yaml(dataset_dir)

    # Step 4: Run YOLO training locally
    run_yolo_training(
        yaml_path=yaml_file,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        run_name=args.name
    )


if __name__ == "__main__":
    main()
