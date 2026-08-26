# Running the Arabic OCR pipeline in Google Colab

This guide automates mounting your shared Google Drive folder and running the dataset prep, augmentation,
OCR crop extraction, training and evaluation steps. The notebook / shell steps below assume you gave
Editor rights to the Drive folder you shared (so we can upload checkpoints/logs back to Drive).

Quick steps to run in Colab

1. Open a new Colab notebook or open your own and mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Create workspace and install dependencies

```bash
%cd /content
mkdir -p platevision_workspace
cd platevision_workspace
# Clone your repo (or upload the Code/ folder)
!git clone https://github.com/mben-yah/Parking-Monitoring-WebApp.git
cd Parking-Monitoring-WebApp

# Install dependencies
!pip install -r requirements.txt
!pip install -r Code/ocr_arabic/requirements-ocr.txt
```

3. Copy dataset from your shared Drive folder into the workspace (replace DRIVE_FOLDER with the path)

```bash
# Path where your shared folder is mounted. For example:
DRIVE_BASE=/content/drive/MyDrive/SharedPlateDataset
# Copy into workspace
rsync -av "$DRIVE_BASE/" ./Data/ELPD_raw/
```

4. Prepare ELPD dataset (COCO -> YOLO) — if your repo contains elpd_prep.py use it; otherwise adapt

```bash
# Example using the repo's elpd_prep.py (adjust args as needed)
python Code/elpd_prep.py --src Data/ELPD_raw --out ELPD_Commercial
```

5. Run augmentations (cropping, zooming, color jitter)

```bash
python Code/ocr_arabic/augment_arabic.py --images-dir ELPD_Commercial/images/train --labels-dir ELPD_Commercial/labels/train --out-dir Code/ocr_arabic/augmented --aug-per-image 3
```

6. Produce OCR crops and labels

```bash
python Code/ocr_arabic/data_prep_arabic.py --images-dir Code/ocr_arabic/augmented/images --labels-dir Code/ocr_arabic/augmented/labels --out-crops Code/ocr_arabic/ocr_crops
```

7. Train CRNN OCR model

```bash
python Code/ocr_arabic/train_crnn_arabic.py --base-dir Code/ocr_arabic/ocr_crops --train-list Code/ocr_arabic/ocr_crops/labels.txt --val-list Code/ocr_arabic/ocr_crops/labels.txt --epochs 40 --batch 32 --save-dir /content/drive/MyDrive/SharedPlateDataset/checkpoints/crnn_ar
```

Notes
- I recommend using GPU runtime in Colab (Runtime -> Change runtime type -> GPU).
- Checkpoints and logs will be saved directly to your shared Drive folder under `checkpoints/crnn_ar`.
- If you prefer I can add a runnable notebook (.ipynb) to the branch so you can click-and-run; tell me and I will push it.
