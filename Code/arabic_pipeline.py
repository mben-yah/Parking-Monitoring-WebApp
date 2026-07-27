# -*- coding: utf-8 -*-
# =============================================================================
#  Arabic License Plate Recognition Pipeline  –  Ar_Dataset_Split
# =============================================================================
#  Dataset layout
#  ──────────────
#  Ar_Dataset_Split/
#    images/
#      train/   0001.jpg … 1276.jpg
#      val/     0001.jpg … 0320.jpg
#    labels/                         ← YOLO-format bbox labels (already split)
#      train/   0001.txt …
#      val/     0001.txt …
#
#  Each label file contains one or more rows:
#      <class_id>  <cx>  <cy>  <w>  <h>   (all normalised 0-1)
#
#  Stages
#  ──────
#  1.  verify_dataset()          – sanity-check that every image has a label
#  2.  write_yaml()              – produce arabic_dataset.yaml for YOLOv8
#  3.  train_model()             – fine-tune YOLOv8n on Arabic plates
#                                  (skip if best weights already exist)
#  4.  visualize_ground_truth()  – draw GT boxes on a random sample
#  5.  run_inference_single()    – YOLO detection → crop → EasyOCR (Arabic)
#  6.  run_batch()               – process many images, log results + accuracy
# =============================================================================

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS   (edit BASE_DIR if you move the project)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
AR_DIR    = BASE_DIR / "Ar_Dataset_Split"

IMAGES    = {split: AR_DIR / "images" / split for split in ("train", "val")}
LABELS    = {split: AR_DIR / "labels" / split for split in ("train", "val")}
YAML_PATH = BASE_DIR / "arabic_dataset.yaml"

# Where to store runs / results
RESULTS_DIR = BASE_DIR / "results" / "arabic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO weights – training saved to a nested runs/detect/runs/detect/arabic_train path
BEST_WEIGHTS = BASE_DIR / "runs" / "detect" / "runs" / "detect" / "arabic_train" / "weights" / "best.pt"
# Fallback: reuse the English weights (same detection task, different plates)
FALLBACK_WEIGHTS = BASE_DIR / "runs" / "detect" / "train9" / "weights" / "best.pt"


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _active_weights() -> Path:
    """Return best available YOLO weight file."""
    if BEST_WEIGHTS.exists():
        return BEST_WEIGHTS
    if FALLBACK_WEIGHTS.exists():
        print(f"[WARN] Arabic weights not found – using English fallback: {FALLBACK_WEIGHTS}")
        return FALLBACK_WEIGHTS
    raise FileNotFoundError(
        "No trained weights found. Run train_model() first or place weights at:\n"
        f"  {BEST_WEIGHTS}"
    )


def _read_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO txt label file into a list of (cls, cx, cy, w, h) tuples."""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            rows.append((cls, cx, cy, w, h))
    return rows


def _yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised coords → pixel x1,y1,x2,y2 (clamped)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))


def _cer(pred: str, ref: str) -> float:
    """Character Error Rate = Levenshtein / len(ref).  Returns 0 if ref is empty."""
    if not ref:
        return 0.0
    pred, ref = pred.lower().strip(), ref.lower().strip()
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        ndp = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            ndp[j] = min(ndp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[n] / m


# ── OCR reader/model cache ────────────────────────────────────────────────────
_READER_CACHE: dict[tuple, object] = {}

OCR_CONF_THRESHOLD = 0.30   # minimum EasyOCR confidence to accept a token
MIN_CROP_HEIGHT    = 64     # upscale crops shorter than this before OCR


# ─── Shared preprocessing ────────────────────────────────────────────────────

def _preprocess_crop(image_bgr: np.ndarray) -> np.ndarray:
    """
    Improve OCR accuracy by:
      1. Upscaling small crops (preserves aspect ratio)
      2. CLAHE contrast enhancement on the L-channel
      3. Unsharp-mask sharpening
    """
    h, w = image_bgr.shape[:2]

    if h < MIN_CROP_HEIGHT:
        scale     = MIN_CROP_HEIGHT / h
        image_bgr = cv2.resize(image_bgr, (max(1, int(w * scale)), MIN_CROP_HEIGHT),
                               interpolation=cv2.INTER_CUBIC)

    lab       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    l         = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    image_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    blur      = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.5)
    image_bgr = cv2.addWeighted(image_bgr, 1.6, blur, -0.6, 0)
    return image_bgr


def _clean_plate_text(text: str) -> str:
    """
    Strip common OCR noise from plate text:
    keep Arabic letters, Arabic-Indic digits (٠-٩), Western digits (0-9),
    and Latin letters (for mixed plates).  Remove everything else.
    """
    import re
    # Arabic block: \u0600-\u06FF, Arabic-Indic digits: ٠١٢٣٤٥٦٧٨٩
    cleaned = re.sub(r"[^\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9A-Za-z0-9]", "", text)
    return cleaned.strip()


# ─── EasyOCR ─────────────────────────────────────────────────────────────────

def _get_reader(langs: list[str], gpu: bool = False):
    """Return a cached EasyOCR Reader (one per language combo)."""
    import easyocr
    key = (tuple(sorted(langs)), gpu)
    if key not in _READER_CACHE:
        try:
            _READER_CACHE[key] = easyocr.Reader(langs, gpu=gpu)
        except Exception:
            _READER_CACHE[key] = easyocr.Reader(langs, gpu=False)
    return _READER_CACHE[key]


def _run_easyocr(image_bgr: np.ndarray, langs: list[str], gpu: bool = False) -> str:
    """Preprocess → EasyOCR beam-search → confidence filter → clean."""
    processed = _preprocess_crop(image_bgr)
    rgb       = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    reader    = _get_reader(langs, gpu=gpu)
    raw       = reader.readtext(rgb, detail=1, paragraph=False,
                                decoder="beamsearch", beamWidth=10)
    tokens    = [t for (_, t, c) in raw if c >= OCR_CONF_THRESHOLD]
    return _clean_plate_text(" ".join(tokens))


def _run_easyocr_gpu(image_bgr: np.ndarray, langs: list[str]) -> str:
    return _run_easyocr(image_bgr, langs, gpu=True)


# ─── PaddleOCR ───────────────────────────────────────────────────────────────

_PADDLE_CACHE: dict[str, object] = {}

def _run_paddleocr(image_bgr: np.ndarray, lang: str = "ar") -> str:
    """
    Run PaddleOCR v3 on the plate crop.
    Uses the updated v3.6.0 API (lang, device, text_det_thresh, etc.)
    Falls back gracefully if PaddleOCR is not installed.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "[PaddleOCR not installed – run: pip install paddlepaddle paddleocr]"

    if lang not in _PADDLE_CACHE:
        _PADDLE_CACHE[lang] = PaddleOCR(
            lang=lang,
            device="cpu",
            enable_mkldnn=False,        # disable oneDNN – crashes on some CPU configs
            use_textline_orientation=True,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_rec_score_thresh=OCR_CONF_THRESHOLD,
        )

    processed = _preprocess_crop(image_bgr)
    rgb       = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    try:
        results = _PADDLE_CACHE[lang].predict(rgb)
        texts = []
        for res in results:
            # v3 result: all data is inside res.json['res']
            data = res.json.get("res", {}) if isinstance(res.json, dict) else {}
            rec_texts  = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                if score >= OCR_CONF_THRESHOLD:
                    texts.append(text)
    except Exception as exc:
        return f"[PaddleOCR error: {exc}]"


    return _clean_plate_text(" ".join(texts))



# ─── Surya OCR ───────────────────────────────────────────────────────────────

_SURYA_MODEL_CACHE: dict = {}

def _run_surya(image_bgr: np.ndarray, langs: list[str] | None = None) -> str:
    """
    Run Surya OCR using the new two-stage API (detection + recognition).
    Install: pip install surya-ocr
    """
    try:
        from surya.recognition import RecognitionPredictor
        from PIL import Image as PILImage
    except ImportError:
        return "[surya-ocr not installed – run: pip install surya-ocr]"
    except Exception as exc:
        return f"[surya import error: {exc}]"

    # Cache predictor (no detection stage needed with full_page=True)
    if "rec" not in _SURYA_MODEL_CACHE:
        _SURYA_MODEL_CACHE["rec"] = RecognitionPredictor()

    processed = _preprocess_crop(image_bgr)
    pil_img   = PILImage.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    try:
        rec_pred = _SURYA_MODEL_CACHE["rec"]

        # full_page=True: recognise all text directly without layout labels
        rec_results = rec_pred(images=[pil_img], layout_results=None, full_page=True)

        texts = []
        for page in rec_results:
            lines = getattr(page, "text_lines", None) or getattr(page, "lines", [])
            for line in lines:
                text = getattr(line, "text", "").strip()
                conf = getattr(line, "confidence", 1.0)
                if conf >= OCR_CONF_THRESHOLD and text:
                    texts.append(text)
        return _clean_plate_text(" ".join(texts))
    except Exception as exc:
        return f"[Surya error: {exc}]"




# ─── Tesseract ───────────────────────────────────────────────────────────────

def _run_tesseract(image_bgr: np.ndarray, lang: str = "ara") -> str:
    """
    Run Tesseract OCR.
    Requires:  pip install pytesseract
               + Tesseract binary with Arabic lang pack installed.
    lang: Tesseract language code – 'ara' for Arabic, 'eng' for English.
    """
    try:
        import pytesseract
    except ImportError:
        return "[pytesseract not installed – run: pip install pytesseract]"

    processed = _preprocess_crop(image_bgr)
    gray      = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    # PSM 7 = single line of text (ideal for licence plates)
    cfg       = f"--oem 3 --psm 7 -l {lang}"
    text      = pytesseract.image_to_string(gray, config=cfg)
    return _clean_plate_text(text)


# ─── Shared ENGINE_MAP builder ────────────────────────────────────────────────

def _build_engine_map() -> dict:
    """
    Return a dict mapping engine-name → callable(image_bgr) -> str.
    Add new engines here; they will be picked up everywhere automatically.
    """
    return {
        # ── EasyOCR ──────────────────────────────────────────────────────────
        "easyocr_ar":         lambda img: _run_easyocr(img, ["ar"],       gpu=False),
        "easyocr_ar_en":      lambda img: _run_easyocr(img, ["ar","en"],  gpu=False),
        "easyocr_ar_en_gpu":  lambda img: _run_easyocr(img, ["ar","en"],  gpu=True),
        # ── PaddleOCR ────────────────────────────────────────────────────────
        "paddleocr_ar":       lambda img: _run_paddleocr(img, lang="ar"),
        # ── Surya ────────────────────────────────────────────────────────────
        "surya_ar":           lambda img: _run_surya(img, langs=["ar"]),
        "surya_ar_en":        lambda img: _run_surya(img, langs=["ar","en"]),
        # ── Tesseract ────────────────────────────────────────────────────────
        "tesseract_ar":       lambda img: _run_tesseract(img, lang="ara"),
    }


def _save_jsonl(record: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")




# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 – Dataset sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset(splits: Iterable[str] = ("train", "val")) -> dict:
    """
    Verify every image has a corresponding YOLO label file.
    Prints a short summary and returns counts.
    """
    summary = {}
    for split in splits:
        img_dir = IMAGES[split]
        lbl_dir = LABELS[split]
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))
        missing_labels, missing_images = [], []

        for img in imgs:
            lbl = lbl_dir / img.with_suffix(".txt").name
            if not lbl.exists():
                missing_labels.append(img.name)

        for lbl in sorted(lbl_dir.glob("*.txt")):
            # try jpg and jpeg
            found = any((img_dir / lbl.with_suffix(ext).name).exists()
                        for ext in (".jpg", ".jpeg", ".png"))
            if not found:
                missing_images.append(lbl.name)

        print(f"\n[{split.upper()}]  images={len(imgs)}  "
              f"missing_labels={len(missing_labels)}  "
              f"missing_images={len(missing_images)}")
        if missing_labels:
            print(f"  ⚠ Images without labels: {missing_labels[:5]}{'…' if len(missing_labels)>5 else ''}")
        if missing_images:
            print(f"  ⚠ Labels without images: {missing_images[:5]}{'…' if len(missing_images)>5 else ''}")

        summary[split] = {"images": len(imgs),
                          "missing_labels": missing_labels,
                          "missing_images": missing_images}

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 – Write dataset YAML for YOLOv8
# ─────────────────────────────────────────────────────────────────────────────

def write_yaml() -> Path:
    """Write the arabic_dataset.yaml required for YOLOv8 training."""
    content = (
        f"path: {AR_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['plate']\n"
    )
    YAML_PATH.write_text(content, encoding="utf-8")
    print(f"YAML written → {YAML_PATH}")
    return YAML_PATH


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 – Train YOLOv8
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    base_model: str = "yolov8s.pt",   # upgraded from nano → small for better accuracy
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "arabic_train_s",
) -> None:
    """
    Fine-tune YOLOv8s (small) on the Arabic plate dataset.
    Skipped automatically if best weights already exist.
    """
    if BEST_WEIGHTS.exists():
        print(f"Trained weights found at {BEST_WEIGHTS} – skipping training.")
        return

    yaml = YAML_PATH if YAML_PATH.exists() else write_yaml()

    from ultralytics import YOLO
    model = YOLO(base_model)
    model.train(
        data=str(yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=15,
        save=True,
        workers=0,          # set >0 on Linux for speed
        verbose=True,
    )
    print(f"\nTraining done. Best weights: {BEST_WEIGHTS}")


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 – Visualise ground-truth boxes
# ─────────────────────────────────────────────────────────────────────────────

def visualize_ground_truth(n: int = 6, split: str = "train", seed: int = 0) -> None:
    """
    Display a random grid of `n` images from `split` with GT bounding boxes drawn.
    """
    img_dir = IMAGES[split]
    lbl_dir = LABELS[split]
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))

    random.seed(seed)
    sample = random.sample(all_imgs, min(n, len(all_imgs)))

    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, sample):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        ax.imshow(img_rgb)
        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        for _, cx, cy, bw, bh in _read_yolo_label(lbl_path):
            x1, y1, x2, y2 = _yolo_to_pixel(cx, cy, bw, bh, w, h)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="#00e5ff", facecolor="none"
            )
            ax.add_patch(rect)

    # hide empty axes
    for ax in axes[len(sample):]:
        ax.set_visible(False)

    plt.suptitle(f"Ground-truth boxes  [{split}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 – Single-image inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_single(
    image_path: str | Path,
    ocr_engines: tuple[str, ...] = ("paddleocr_ar",),
    show: bool = True,
    save_path: str | Path | None = None,
) -> dict:
    """
    Run YOLO detection + Arabic OCR on a single image.

    Parameters
    ----------
    image_path   : path to the input image.
    ocr_engines  : which OCR back-ends to use (can list multiple to compare).
                   Options: 'paddleocr_ar'       → PaddleOCR Arabic (default, best)
                            'surya_ar'           → Surya transformer OCR
                            'surya_ar_en'        → Surya Arabic + English
                            'easyocr_ar'         → EasyOCR Arabic
                            'easyocr_ar_en'      → EasyOCR Arabic + English
                            'easyocr_ar_en_gpu'  → EasyOCR GPU
                            'tesseract_ar'       → Tesseract Arabic
    show         : display the annotated result.
    save_path    : if given, save the annotated image here.

    Returns
    -------
    dict with keys: bbox, confidence, ocr_results, annotated_image
    """
    from ultralytics import YOLO

    weights = _active_weights()
    model   = YOLO(str(weights))

    img_path = Path(image_path)
    img_bgr  = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    img_h, img_w = img_bgr.shape[:2]

    # ── YOLO detection ────────────────────────────────────────────────────────
    yolo_results = model.predict(str(img_path), conf=0.15, verbose=False)
    boxes        = yolo_results[0].boxes if yolo_results[0].boxes is not None else []

    annotated = img_bgr.copy()

    if len(boxes) == 0:
        print(f"[{img_path.name}] No plate detected by YOLO.")
        return {
            "bbox": None, "confidence": None,
            "ocr_results": {eng: "" for eng in ocr_engines},
            "annotated_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        }

    # Use the highest-confidence box
    conf_scores = boxes.conf.cpu().numpy()
    best_idx    = int(np.argmax(conf_scores))
    box         = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    confidence  = float(conf_scores[best_idx])

    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    plate_crop = img_bgr[y1:y2, x1:x2]

    # ── Draw detection box ────────────────────────────────────────────────────
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 229, 255), 2)
    cv2.putText(annotated, f"plate {confidence:.2f}", (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)

    # ── OCR ──────────────────────────────────────────────────────────────────
    ENGINE_MAP = _build_engine_map()

    ocr_results: dict[str, str] = {}
    for engine in ocr_engines:
        fn = ENGINE_MAP.get(engine)
        if fn is None:
            print(f"[WARN] Unknown engine '{engine}', skipping.")
            continue
        try:
            text = fn(plate_crop)
        except Exception as exc:
            text = f"ERROR: {exc}"
        ocr_results[engine] = text

        # Overlay OCR text on the annotated image
        y_txt = y2 + 24 * (list(ocr_results.keys()).index(engine) + 1)
        cv2.putText(annotated, f"{engine}: {text}", (x1, min(y_txt, img_h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=(10, 6))
        plt.imshow(annotated_rgb)
        plt.title(f"{img_path.name}  |  conf={confidence:.3f}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"Saved → {save_path}")

    result = {
        "bbox":           [int(x1), int(y1), int(x2), int(y2)],
        "confidence":     confidence,
        "ocr_results":    ocr_results,
        "annotated_image": annotated_rgb,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 – Batch inference with logging
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    split: str = "val",
    ocr_engines: tuple[str, ...] = ("paddleocr_ar",),
    show: bool = False,
    output_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    sample_size: int | None = None,
    random_seed: int = 42,
) -> list[dict]:
    """
    Run YOLO + Arabic OCR on a full split (or a random sample of it).

    Parameters
    ----------
    split        : 'train' or 'val'
    ocr_engines  : OCR back-ends (see run_inference_single for options)
    show         : show annotated images during processing
    output_dir   : folder to save annotated images (default: results/arabic/<split>)
    log_path     : JSONL log file path (default: results/arabic/<split>_log.jsonl)
    sample_size  : if set, process only this many images (random sample)
    random_seed  : seed for reproducible sampling

    Returns
    -------
    list of result dicts (one per image)
    """
    from ultralytics import YOLO

    img_dir = IMAGES[split]
    lbl_dir = LABELS[split]

    all_imgs = sorted(img_dir.glob("*.jpg")) + \
               sorted(img_dir.glob("*.jpeg")) + \
               sorted(img_dir.glob("*.png"))

    if not all_imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # ── Subsample ─────────────────────────────────────────────────────────────
    if sample_size is not None and sample_size < len(all_imgs):
        random.seed(random_seed)
        all_imgs = random.sample(all_imgs, sample_size)
        all_imgs.sort()

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir  = Path(output_dir) if output_dir else RESULTS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) if log_path else RESULTS_DIR / f"{split}_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO once ────────────────────────────────────────────────────────
    weights = _active_weights()
    model   = YOLO(str(weights))

    ENGINE_MAP = _build_engine_map()

    results, cer_acc = [], {eng: [] for eng in ocr_engines}

    print(f"\n{'='*60}")
    print(f"  Arabic Batch  |  split={split}  |  images={len(all_imgs)}")
    print(f"  Engines: {list(ocr_engines)}")
    print(f"{'='*60}")

    for idx, img_path in enumerate(all_imgs, 1):
        save_path = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
        lbl_path  = lbl_dir / img_path.with_suffix(".txt").name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{idx}/{len(all_imgs)}] Cannot read {img_path.name} – skipping")
            continue

        img_h, img_w = img_bgr.shape[:2]
        annotated    = img_bgr.copy()

        # ── YOLO ──────────────────────────────────────────────────────────────
        yolo_out = model.predict(str(img_path), conf=0.15, verbose=False)
        boxes    = yolo_out[0].boxes if yolo_out[0].boxes is not None else []

        ocr_results: dict[str, str] = {eng: "" for eng in ocr_engines}
        bbox, confidence = None, None

        if len(boxes) > 0:
            conf_scores = boxes.conf.cpu().numpy()
            best_idx    = int(np.argmax(conf_scores))
            box_arr     = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            confidence  = float(conf_scores[best_idx])

            x1, y1, x2, y2 = box_arr
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            bbox    = [int(x1), int(y1), int(x2), int(y2)]

            plate_crop = img_bgr[y1:y2, x1:x2]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 229, 255), 2)

            # ── OCR ───────────────────────────────────────────────────────────
            for i, engine in enumerate(ocr_engines):
                fn = ENGINE_MAP.get(engine)
                if fn is None:
                    continue
                try:
                    text = fn(plate_crop)
                except Exception as exc:
                    text = f"ERROR: {exc}"
                ocr_results[engine] = text

                y_txt = y2 + 22 * (i + 1)
                cv2.putText(annotated, f"{engine}: {text}",
                            (x1, min(y_txt, img_h - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # ── Save annotated image ───────────────────────────────────────────────
        cv2.imwrite(str(save_path), annotated)

        if show:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.title(img_path.name); plt.axis("off"); plt.show()

        # ── Per-image CER (no text GT, so we compare between engines) ─────────
        # If only one engine, nothing to compare against; skip CER in that case.
        inter_cer = None
        engine_list = [e for e in ocr_engines if ocr_results.get(e)]
        if len(engine_list) >= 2:
            inter_cer = _cer(ocr_results[engine_list[1]], ocr_results[engine_list[0]])

        # ── Log record ────────────────────────────────────────────────────────
        record = {
            "timestamp":  datetime.now().isoformat(),
            "image":      img_path.name,
            "split":      split,
            "bbox":       bbox,
            "confidence": confidence,
            "ocr":        ocr_results,
            "inter_engine_cer": inter_cer,
        }
        _save_jsonl(record, log_file)

        result = {
            "image_path":    str(img_path),
            "annotated_path": str(save_path),
            **record,
        }
        results.append(result)

        # ── Live print ────────────────────────────────────────────────────────
        status = f"conf={confidence:.2f}" if confidence else "NO PLATE"
        print(f"[{idx:>4}/{len(all_imgs)}] {img_path.name:<15}  {status}")
        for eng, txt in ocr_results.items():
            print(f"           {eng:<22}: {txt!r}")
        if inter_cer is not None:
            print(f"           inter-engine CER     : {inter_cer:.3f}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    detected  = sum(1 for r in results if r["confidence"] is not None)
    avg_conf  = np.mean([r["confidence"] for r in results if r["confidence"] is not None] or [0])
    all_cers  = [r["inter_engine_cer"] for r in results if r["inter_engine_cer"] is not None]
    avg_cer   = np.mean(all_cers) if all_cers else None

    print(f"\n{'='*60}")
    print(f"  Processed    : {len(results)} images")
    print(f"  Plates found : {detected} ({detected/max(1,len(results))*100:.1f}%)")
    print(f"  Avg YOLO conf: {avg_conf:.3f}")
    if avg_cer is not None:
        print(f"  Avg inter-engine CER : {avg_cer:.3f}")
    print(f"  Log saved    : {log_file}")
    print(f"{'='*60}\n")

    return results
