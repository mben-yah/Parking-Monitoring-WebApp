# Arabic OCR improvement workspace

This folder contains scripts to prepare Arabic license-plate crops, train a lightweight CRNN OCR model (Arabic + western digits), evaluate models and normalize Arabic reads.

Overview
- data_prep_arabic.py: convert YOLO-labeled dataset into cropped images + labels for OCR training
- arabic_normalization.py: helpers to normalize Arabic digits/characters and common confusions
- train_crnn_arabic.py: PyTorch CRNN training script (CTC) — configurable
- evaluate_ocr_arabic.py: evaluate a trained checkpoint (CER, plate-level accuracy)
- requirements-ocr.txt: suggested additional dependencies for OCR training

Usage (quick)
1. Prepare crops and labels:
   python Code/ocr_arabic/data_prep_arabic.py --images-dir "ELPD_Commercial/images/train" --labels-dir "ELPD_Commercial/labels/train" --out-crops "ocr_crops/train"

2. Train:
   python Code/ocr_arabic/train_crnn_arabic.py --train-list ocr_crops/train/labels.txt --val-list ocr_crops/val/labels.txt --epochs 60 --save-dir checkpoints/crnn_ar

3. Evaluate:
   python Code/ocr_arabic/evaluate_ocr_arabic.py --ckpt checkpoints/crnn_ar/best.pt --val-list ocr_crops/val/labels.txt

Notes
- The scripts are intentionally simple and documented to be runnable on a single GPU. They are a starting point; you can swap the CRNN encoder for a transformer later.
- Do not commit large model files to the repo; use external storage or GitHub Releases.
