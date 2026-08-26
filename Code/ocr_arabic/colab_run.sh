#!/bin/bash
# Helper runner for Colab/remote. Assumes working dir is repository root.

# Install dependencies
pip install -r requirements.txt
pip install -r Code/ocr_arabic/requirements-ocr.txt

# Copy dataset (if already in Drive mounted at /content/drive)
# Adjust DRIVE_BASE variable to your mounted Drive folder
DRIVE_BASE=/content/drive/MyDrive/SharedPlateDataset
mkdir -p Data/ELPD_raw
rsync -av "$DRIVE_BASE/" Data/ELPD_raw/

# Prepare dataset (use repo's elpd_prep.py)
python Code/elpd_prep.py --src Data/ELPD_raw --out ELPD_Commercial

# Augment
python Code/ocr_arabic/augment_arabic.py --images-dir ELPD_Commercial/images/train --labels-dir ELPD_Commercial/labels/train --out-dir Code/ocr_arabic/augmented --aug-per-image 3

# Prepare OCR crops
python Code/ocr_arabic/data_prep_arabic.py --images-dir Code/ocr_arabic/augmented/images --labels-dir Code/ocr_arabic/augmented/labels --out-crops Code/ocr_arabic/ocr_crops

# Train OCR model (save to Drive)
python Code/ocr_arabic/train_crnn_arabic.py --base-dir Code/ocr_arabic/ocr_crops --train-list Code/ocr_arabic/ocr_crops/labels.txt --val-list Code/ocr_arabic/ocr_crops/labels.txt --epochs 40 --batch 32 --save-dir "$DRIVE_BASE/checkpoints/crnn_ar"
