# -*- coding: utf-8 -*-
# =============================================================================
# License Plate Recognition Pipeline  -  English Dataset (AOLP Subset_LE)
# =============================================================================
# Stages:
#   1.  Convert AOLP localization annotations -> YOLO format
#   2.  Build the train/val dataset split
#   3.  (Training already done - load best.pt weights)
#   4.  Visualize ground-truth bounding boxes
#   5.  End-to-end inference: YOLO detection -> crop -> EasyOCR -> annotated image
#   6.  Batch OCR evaluation
# =============================================================================

from pathlib import Path
import cv2
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
import json
from datetime import datetime
# EasyOCR and ultralytics imports are done inside functions to avoid heavy import at module load

def _save_log(data: dict, log_path: Path):
    """Save dictionary as pretty‑printed JSON with a timestamp.
    Creates parent directories if needed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # embed timestamp for reproducibility
    data_to_save = {"timestamp": datetime.now().isoformat(), **data}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    print(f"Log saved to {log_path}")
def _run_ocr_paddle(image_bgr):
    """
    Run PaddleOCR on a BGR image (numpy array) and return the concatenated text.
    The function lazily imports PaddleOCR to avoid a hard dependency when not used.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise ImportError("PaddleOCR is not installed. Install it via 'pip install paddleocr' to use the paddle OCR engine.") from e
    import cv2
    # Convert BGR to RGB as PaddleOCR expects RGB images
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Initialize PaddleOCR (CPU mode, English language)
    ocr = PaddleOCR(lang='en', use_gpu=False, det=True, rec=True, cls=False, show_log=False)
    result = ocr.ocr(img_rgb, cls=False)
    # Extract text components
    texts = []
    for line in result:
        if line and len(line) > 0:
            txt, _ = line[1]
            texts.append(txt)
    return " ".join(texts).strip()


# ---------------------------------------------------------------------------
# PATHS  (edit BASE_DIR if you move the project)
# ---------------------------------------------------------------------------
BASE_DIR   = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
SUBSET_DIR = BASE_DIR / "Subset_LE"

IMAGES_SRC = SUBSET_DIR / "Image"                     # source images
LABELS_SRC = SUBSET_DIR / "groundtruth_localization"  # raw pixel-coord labels
RECOG_SRC  = SUBSET_DIR / "groundtruth_recognition"   # plate text labels

DATASET_DIR  = BASE_DIR / "Dataset"
YAML_PATH    = BASE_DIR / "dataset.yaml"
def _resolve_best_weights() -> Path:
    """Auto-detect the highest-numbered english_trainN weights available."""
    search_root = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    import re
    best_run   = None
    best_num   = -1
    if search_root.exists():
        for d in search_root.iterdir():
            m = re.fullmatch(r"english_train(\d*)", d.name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                pt  = d / "weights" / "best.pt"
                if pt.exists() and num > best_num:
                    best_num = num
                    best_run = pt
    # Fallback to hard-coded path
    if best_run is None:
        best_run = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "english_train2" / "weights" / "best.pt"
    return best_run

BEST_WEIGHTS   = _resolve_best_weights()
ARABIC_WEIGHTS = BASE_DIR / "runs" / "detect" / "train9" / "weights" / "best.pt"
print(f"[recognition_pipeline] Active model: {BEST_WEIGHTS}")


# =============================================================================
# STAGE 1 - Helper: convert absolute pixel bbox -> YOLO normalised format
# =============================================================================

def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert top-left/bottom-right pixel coords to YOLO centre format."""
    # BUG FIX: original Cell 1 had  / 8--  (syntax error) instead of / img_w
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width    = (x2 - x1) / img_w
    height   = (y2 - y1) / img_h
    return x_center, y_center, width, height


def read_localization(label_path):
    """
    Read an AOLP localization file.
    Coords may be stored as floats with scientific notation, e.g.:
        200
        2.110000e+002
        293
        2.430000e+002
    Returns (x1, y1, x2, y2) as floats.
    """
    with open(label_path) as f:
        values = [float(v) for v in f.read().split()]
    if len(values) == 4:
        return values   # x1 y1 x2 y2
    raise ValueError("Unexpected format in {}: {}".format(label_path, values))


# =============================================================================
# STAGE 2 - Build YOLO dataset (images + label .txt files)
# =============================================================================

def build_dataset(val_split=0.2, seed=42):
    """
    Scan Subset_LE/Image, convert localization labels to YOLO format,
    copy everything into Dataset/{images,labels}/{train,val}.
    """
    # Create output dirs
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # BUG FIX: original dataset-prep cell used  base / "Image"  which is wrong.
    #           Images live in  Subset_LE/Image,  not  Code/Image.
    all_images = sorted(IMAGES_SRC.glob("*.jpg"))
    valid = []
    for img_path in all_images:
        loc_path = LABELS_SRC / (img_path.stem + ".txt")
        if loc_path.exists():
            valid.append(img_path)
        else:
            print("WARNING: No localization for {} - skipped".format(img_path.name))

    random.seed(seed)
    random.shuffle(valid)
    split_idx   = int((1.0 - val_split) * len(valid))
    train_files = valid[:split_idx]
    val_files   = valid[split_idx:]

    def process(files, split):
        ok = skipped = 0
        for img_path in files:
            loc_path = LABELS_SRC / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None:
                print("ERROR: Cannot read {}".format(img_path.name))
                skipped += 1
                continue

            img_h, img_w = img.shape[:2]
            try:
                x1, y1, x2, y2 = read_localization(loc_path)
            except Exception as e:
                print("ERROR: Bad label {}: {}".format(loc_path.name, e))
                skipped += 1
                continue

            xc, yc, bw, bh = convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)

            # Copy image
            shutil.copy(img_path, DATASET_DIR / "images" / split / img_path.name)

            # Write YOLO label
            label_out = DATASET_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_out, "w") as f:
                f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(xc, yc, bw, bh))
            ok += 1

        print("  [{}] {} processed, {} skipped".format(split, ok, skipped))

    print("Building dataset ...")
    process(train_files, "train")
    process(val_files,   "val")
    print("Dataset build complete!")


# =============================================================================
# STAGE 3 - Write / verify dataset.yaml
# =============================================================================

def write_yaml():
    content = (
        "path: {}\n\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
    ).format(DATASET_DIR.as_posix())

    with open(YAML_PATH, "w") as f:
        f.write(content)
    print("dataset.yaml written to {}".format(YAML_PATH))


# =============================================================================
# STAGE 4 - Visualize ground-truth bounding boxes (sanity check)
# =============================================================================

def visualize_ground_truth(n=6, split="train"):
    """Show n random images with their YOLO ground-truth bounding boxes."""
    img_dir   = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    files = list(img_dir.glob("*.jpg"))
    if not files:
        print("No images found in {}".format(img_dir))
        return

    sample = random.sample(files, min(n, len(files)))
    cols   = 3
    rows   = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)

        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = parts
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        rect = patches.Rectangle(
                            (x1, y1), bw * w, bh * h,
                            linewidth=2, edgecolor="lime", facecolor="none"
                        )
                        ax.add_patch(rect)
        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Ground-truth boxes - {} split".format(split), fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STAGE 5 - Training (already done - shown for reference only)
# =============================================================================

def train_model(epochs=50):
    """Train YOLOv8n - skips automatically if best.pt already exists."""
    if BEST_WEIGHTS.exists():
        print("Trained weights found at {} - skipping training.".format(BEST_WEIGHTS))
        return

    from ultralytics import YOLO
    model = YOLO(str(BASE_DIR / "yolov8n.pt"))
    model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        project=str(BASE_DIR / "runs" / "detect"),
        name="train",
    )


# =============================================================================
# STAGE 6 - End-to-end inference: YOLO -> crop -> EasyOCR -> annotated output
# =============================================================================

def run_pipeline(image_path=None, conf_threshold=0.25, show=True, save_path=None):
    """
    Full pipeline on a single image:
      1. YOLO detects the plate bounding box.
      2. The plate region is cropped.
      3. EasyOCR reads characters from the crop.
      4. An annotated image (box + text) is displayed and/or saved.

    Parameters
    ----------
    image_path      : str | Path | None  - defaults to Subset_LE/Image/1.jpg
    conf_threshold  : float              - minimum YOLO confidence to accept
    show            : bool               - display result with matplotlib
    save_path       : str | Path | None  - save annotated image here if given

    Returns
    -------
    dict: bbox, confidence, ocr_text, annotated_image
    """
    from ultralytics import YOLO
    import easyocr

    # 1. Resolve image path
    if image_path is None:
        image_path = SUBSET_DIR / "Image" / "1.jpg"
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Image not found: {}".format(image_path))

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(
            "Trained weights not found at {}.\n"
            "Run train_model() first.".format(BEST_WEIGHTS)
        )

    # 2. Load model
    model = YOLO(str(BEST_WEIGHTS))

    # 3. Read image and run YOLO inference
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError("cv2 could not read: {}".format(image_path))

    results  = model(img_bgr, conf=conf_threshold, verbose=False)
    result   = results[0]
    img_h, img_w = img_bgr.shape[:2]

    annotated = img_bgr.copy()
    ocr_text  = ""
    bbox_info = None
    confidence = None

    if len(result.boxes) == 0:
        print("WARNING: No plate detected in this image.")
    else:
        # 4. Pick highest-confidence detection
        boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)
        best  = boxes[0]

        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        confidence = float(best.conf[0])
        bbox_info  = (x1, y1, x2, y2)

        # Clamp to image bounds
        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(img_w, x2);  y2c = min(img_h, y2)

        # 5. Crop plate region
        plate_crop = img_bgr[y1c:y2c, x1c:x2c]

        # 6. EasyOCR - BUG FIX: old cell used a hardcoded wrong path;
        #    now we pass the actual cropped plate numpy array directly.
        reader  = easyocr.Reader(['en'], gpu=False, verbose=False)
        ocr_res = reader.readtext(plate_crop)
        ocr_text = " ".join(r[1] for r in ocr_res).strip()

        print("YOLO confidence : {:.2%}".format(confidence))
        print("Bounding box    : ({}, {}) -> ({}, {})".format(x1, y1, x2, y2))
        print("OCR result      : {}".format(ocr_text if ocr_text else "(no text detected)"))

        # 7. Draw bounding box + label on original image
        box_color  = (0, 230, 80)    # bright green
        text_color = (255, 255, 255)
        thickness  = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        label = "{} [{:.0%}]".format(ocr_text if ocr_text else "plate", confidence)
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.7
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)

        pad = 4
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 2 * pad),
            (x1 + tw + 2 * pad, y1),
            box_color, -1
        )
        cv2.putText(
            annotated, label,
            (x1 + pad, y1 - pad),
            font, scale, text_color, 1, cv2.LINE_AA
        )

    # 8. Display side-by-side (original vs annotated)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(annotated_rgb)
        axes[1].set_title(
            "Detection + OCR: '{}'".format(ocr_text) if ocr_text
            else "Detection (no OCR text)",
            fontsize=12
        )
        axes[1].axis("off")

        plt.suptitle(image_path.name, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print("Saved annotated image to {}".format(save_path))

    return {
        "bbox"           : bbox_info,
        "confidence"     : confidence,
        "ocr_text"       : ocr_text,
        "annotated_image": annotated_rgb,
    }

# PaddleOCR support removed – using EasyOCR only



def run_pipeline_multi_ocr(image_path=None, conf_threshold=0.25, show=True, save_path=None, ocr_engines=('easyocr_en_cpu', 'easyocr_en_gpu')):
    """Run detection once and apply selected EasyOCR back‑ends.

    Parameters
    ----------
    image_path : str | Path | None
        Path to the image. Defaults to Subset_LE/Image/1.jpg.
    conf_threshold : float
        Minimum detection confidence.
    show : bool
        Show side‑by‑side visualisation of each OCR result.
    save_path : str | Path | None
        Optional path to store a composite image with all OCR annotations.
    ocr_engines : iterable of str
        Which OCR back-ends to run.
        Supported values:
          ``'easyocr_en_cpu'``  - EasyOCR English, CPU
          ``'easyocr_en_gpu'``  - EasyOCR English, GPU (falls back to CPU if unavailable)
    """
    if image_path is None:
        image_path = SUBSET_DIR / "Image" / "1.jpg"
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"Trained weights not found at {BEST_WEIGHTS}")

    from ultralytics import YOLO
    import easyocr
    import cv2

    model = YOLO(str(BEST_WEIGHTS))
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"cv2 could not read: {image_path}")

    results = model(img_bgr, conf=conf_threshold, verbose=False)[0]

    # ── Handle the case where no plate is detected ──────────────────────────
    if len(results.boxes) == 0:
        print(f"WARNING: No plate detected in {image_path.name} – skipping OCR.")
        return {
            "bbox": None,
            "confidence": None,
            "ocr_results": {eng: "" for eng in ocr_engines},
            "image_path": str(image_path),
        }

    best = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    h, w = img_bgr.shape[:2]
    crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    ocr_results = {}
    for engine in ocr_engines:
        if engine in ('easyocr_en_cpu', 'easyocr'):
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            ocr_res = reader.readtext(crop)
            raw = " ".join(r[1] for r in ocr_res).strip()
            ocr_text = normalize_plate(raw)
        elif engine in ('easyocr_en_gpu', 'easyocr2'):
            # Falls back to CPU automatically if no GPU is available
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            ocr_res = reader.readtext(crop)
            raw = " ".join(r[1] for r in ocr_res).strip()
            ocr_text = normalize_plate(raw)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine!r}")
        ocr_results[engine] = ocr_text

    # ── Always build the annotated frame (needed for show AND save) ──────────
    base_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated = base_rgb.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 80), 2)

    if show:
        n = len(ocr_engines)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, engine in zip(axes, ocr_engines):
            ax.imshow(annotated)
            ax.set_title(f"{engine} OCR: {ocr_results[engine]}")
            ax.axis('off')
        plt.suptitle(image_path.name, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sp), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved image with OCR results to {sp}")

    return {"bbox": (x1, y1, x2, y2), "confidence": float(best.conf[0]), "ocr_results": ocr_results, "image_path": str(image_path)}


# =============================================================================
# STAGE 7 - Batch evaluation: run on multiple val images, report accuracy
# =============================================================================

def normalize_plate(text):
    """Upper‑case, strip non‑alphanumeric characters and whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.upper()
    # keep only letters and digits
    return re.sub(r'[^A-Z0-9]', '', text)


def evaluate_ocr(n=20, split="val", conf_threshold=0.25):
    """
    Run the full pipeline on n images, compare OCR output against
    ground‑truth recognition text, and print an accuracy report.
    """
    from ultralytics import YOLO
    import easyocr

    model  = YOLO(str(BEST_WEIGHTS))
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    img_dir = DATASET_DIR / "images" / split
    files   = list(img_dir.glob("*.jpg"))
    if not files:
        print("No images in {}".format(img_dir))
        return

    sample = random.sample(files, min(n, len(files)))
    exact_matches = 0
    results_log   = []

    for img_path in sample:
        gt_path = RECOG_SRC / (img_path.stem + ".txt")
        gt_raw = gt_path.read_text() if gt_path.exists() else ""
        gt_text = normalize_plate(gt_raw)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        yolo_res = model(img_bgr, conf=conf_threshold, verbose=False)[0]
        pred_text = ""

        if len(yolo_res.boxes) > 0:
            best_box = sorted(yolo_res.boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            h, w = img_bgr.shape[:2]
            crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            ocr_res = reader.readtext(crop)
            pred_raw = " ".join(r[1] for r in ocr_res).strip()
            pred_text = normalize_plate(pred_raw)

        match = (pred_text == gt_text)
        if match:
            exact_matches += 1

        results_log.append({
            "image": img_path.name,
            "gt": gt_text,
            "pred": pred_text,
            "match": match,
        })

    print("\n" + "=" * 60)
    print("  OCR Batch Evaluation  ({}, n={})".format(split, len(results_log)))
    print("=" * 60)
    for r in results_log:
        status = "OK " if r["match"] else "ERR"
        print("  {}  {:15s}  GT: {:15s}  Pred: {}".format(
            status, r["image"], repr(r["gt"]), repr(r["pred"])))
    print("=" * 60)
    pct = 100 * exact_matches / len(results_log) if results_log else 0
    print("  Exact-match accuracy: {}/{} ({:.1f}%)".format(
        exact_matches, len(results_log), pct))
    print("=" * 60 + "\n")
    return results_log


# =============================================================================
# ENTRY POINT - run the complete pipeline end-to-end
# =============================================================================

if __name__ == "__main__":
    print("\n--- Stage 1 & 2: Build YOLO dataset ---")
    build_dataset()

    print("\n--- Stage 3: Write dataset.yaml ---")
    write_yaml()

    print("\n--- Stage 4: Training check ---")
    train_model()   # no-op if best.pt already exists

    print("\n--- Stage 5: Visualize ground-truth boxes ---")
    visualize_ground_truth(n=6, split="train")

    print("\n--- Stage 6: End-to-end inference on image 1 ---")
    out = run_pipeline(
        image_path=SUBSET_DIR / "Image" / "1.jpg",
        show=True,
        save_path=BASE_DIR / "results" / "annotated_1.jpg",
    )
    print("\nFinal OCR text:", out["ocr_text"] or "(none)")
