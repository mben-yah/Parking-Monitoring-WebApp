# -*- coding: utf-8 -*-
"""
train_from_gdrive.py
────────────────────
Automated Google Drive Dataset Downloader & Local YOLOv8 Trainer for Magnetite Vision.

Usage Example:
  C:\Anaconda\envs\yolo_env\python.exe train_from_gdrive.py --url "https://drive.google.com/file/d/1ABC123XYZ.../view?usp=sharing" --epochs 40 --name "elpd_commercial_train3"

Or simply run interactively:
  C:\Anaconda\envs\yolo_env\python.exe train_from_gdrive.py
"""

import argparse
import os
import sys
import zipfile
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
    import requests
    if not hasattr(requests.Session, '_ssl_disabled_send'):
        requests.Session._ssl_disabled_send = requests.Session.send
        def _unverified_send(self, request, **kwargs):
            kwargs['verify'] = False
            return self._ssl_disabled_send(request, **kwargs)
        requests.Session.send = _unverified_send
except Exception:
    pass

BASE_DIR = Path(__file__).parent.resolve()
DOWNLOADS_DIR = BASE_DIR / "downloads"
DATASETS_DIR = BASE_DIR / "Dataset_Custom"
YAML_PATH = BASE_DIR / "dataset_custom.yaml"


def extract_gdrive_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from standard shareable links or return raw ID."""
    url_or_id = url_or_id.strip()
    match = re.search(r"/(?:file/d/|d/|folders/)([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id


def download_from_gdrive(gdrive_url_or_id: str) -> Path:
    """Download file or folder from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("[+] Installing gdown helper package...")
        os.system(f'"{sys.executable}" -m pip install gdown')
        import gdown

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = extract_gdrive_id(gdrive_url_or_id)
    print(f"[+] Downloading dataset from Google Drive (ID: {file_id})...")

    # Destination zip path
    zip_out = DOWNLOADS_DIR / f"dataset_{file_id}.zip"

    # Try downloading as single zip file first
    output = gdown.download(id=file_id, output=str(zip_out), quiet=False)

    if not output or not Path(output).exists():
        # Fallback to folder download
        print("[+] Trying gdown folder download mode...")
        folder_out = DOWNLOADS_DIR / f"dataset_{file_id}"
        gdown.download_folder(id=file_id, output=str(folder_out), quiet=False)
        return folder_out

    return Path(output)


def apply_dataset_augmentations(dataset_dir: Path):
    """Apply dataset augmentations (brightness/contrast, gaussian blur, HSV color shift)."""
    import cv2
    import numpy as np

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    if not images_dir.exists():
        images_dir = dataset_dir

    if not labels_dir.exists():
        labels_dir = dataset_dir

    img_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    if not img_paths:
        print("[!] No images found for dataset augmentation. Skipping augmentation stage.")
        return

    print(f"[+] Applying dataset augmentations to {len(img_paths)} source images...")
    aug_count = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_content = ""
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                label_content = lf.read()

        # Augmentation 1: Brightness & Contrast
        aug_bc = cv2.convertScaleAbs(img, alpha=1.25, beta=15)
        aug_bc_name = f"{img_path.stem}_aug_bc{img_path.suffix}"
        cv2.imwrite(str(images_dir / aug_bc_name), aug_bc)
        if label_content:
            with open(labels_dir / f"{img_path.stem}_aug_bc.txt", "w", encoding="utf-8") as lf:
                lf.write(label_content)
        aug_count += 1

        # Augmentation 2: Gaussian Blur (low-res surveillance simulation)
        aug_blur = cv2.GaussianBlur(img, (5, 5), 0)
        aug_blur_name = f"{img_path.stem}_aug_blur{img_path.suffix}"
        cv2.imwrite(str(images_dir / aug_blur_name), aug_blur)
        if label_content:
            with open(labels_dir / f"{img_path.stem}_aug_blur.txt", "w", encoding="utf-8") as lf:
                lf.write(label_content)
        aug_count += 1

        # Augmentation 3: HSV Color Shift
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 20)
        hsv_mod = cv2.merge((h, s, v))
        aug_hsv = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
        aug_hsv_name = f"{img_path.stem}_aug_hsv{img_path.suffix}"
        cv2.imwrite(str(images_dir / aug_hsv_name), aug_hsv)
        if label_content:
            with open(labels_dir / f"{img_path.stem}_aug_hsv.txt", "w", encoding="utf-8") as lf:
                lf.write(label_content)
        aug_count += 1

    print(f"[+] Dataset augmentation complete: Generated {aug_count} augmented training samples!")


def setup_dataset_structure(downloaded_path: Path) -> Path:
    """Unpack zip, apply dataset augmentations, and organize into YOLO standard structure."""
    if DATASETS_DIR.exists():
        print(f"[+] Cleaning existing custom dataset folder: {DATASETS_DIR}")
        shutil.rmtree(DATASETS_DIR)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if downloaded_path.is_file() and downloaded_path.suffix.lower() == ".zip":
        print(f"[+] Unzipping {downloaded_path.name}...")
        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
            zip_ref.extractall(DATASETS_DIR)
    elif downloaded_path.is_dir():
        print(f"[+] Copying dataset folder from {downloaded_path}...")
        shutil.copytree(downloaded_path, DATASETS_DIR, dirs_exist_ok=True)

    print(f"[+] Dataset extracted to {DATASETS_DIR}")

    # Apply dataset augmentations
    apply_dataset_augmentations(DATASETS_DIR)

    return DATASETS_DIR


def generate_dataset_yaml(dataset_dir: Path) -> Path:
    """Create dataset_custom.yaml configured for YOLOv8 local training."""
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"

    # Fallbacks if images are flat in root or train directory
    if not images_train.exists():
        images_train = dataset_dir

    if not images_val.exists():
        images_val = images_train

    yaml_content = f"""# Magnetite Vision Custom Dataset Configuration
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


def run_yolo_training(yaml_path: Path, epochs: int = 40, batch: int = 16, imgsz: int = 640, run_name: str = "elpd_commercial_custom"):
    """Start local YOLOv8 fine-tuning training."""
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
    print(f"  Starting Local YOLOv8 Training: {run_name}")
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
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive & Train YOLOv8 Locally")
    parser.add_argument("--url", type=str, help="Google Drive shareable link or file/folder ID")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (default: 640)")
    parser.add_argument("--name", type=str, default="elpd_commercial_custom", help="Training run name")

    args = parser.parse_args()

    gdrive_url = args.url
    if not gdrive_url:
        print("=" * 70)
        print("  Magnetite Vision — Google Drive Dataset Trainer")
        print("=" * 70)
        gdrive_url = input("Paste your Google Drive link or File ID: ").strip()

    if not gdrive_url:
        print("[-] Error: Google Drive link or File ID is required!")
        sys.exit(1)

    # Step 1: Download from Google Drive
    dl_path = download_from_gdrive(gdrive_url)

    # Step 2: Unpack dataset
    dataset_dir = setup_dataset_structure(dl_path)

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
