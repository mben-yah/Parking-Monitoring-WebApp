# -*- coding: utf-8 -*-
"""
smart_plate_reader.py
─────────────────────
Runs both PaddleOCR (best for digits) and EasyOCR (captures Arabic letters)
on a detected plate crop, then intelligently merges the two results:

    • Digits   → taken from PaddleOCR (80 %+ confidence, clean number reads)
    • Letters  → taken from EasyOCR   (reads Arabic script characters)

The merge preserves the spatial order of character groups so the final
plate string reflects the actual left-to-right layout of the plate.

Usage
─────
    python smart_plate_reader.py                          # runs on default val image
    python smart_plate_reader.py path\\to\\your\\plate.jpg  # custom image
    python smart_plate_reader.py --batch                  # run on full val set

Or import in a notebook:
    from smart_plate_reader import read_plate
    text = read_plate(r"C:\\...\\0001.jpg")
    print(text)
"""

from __future__ import annotations

import re
import sys
import io
import warnings
import importlib
import cv2
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── ensure arabic_pipeline is fresh ──────────────────────────────────────────
import arabic_pipeline
importlib.reload(arabic_pipeline)
from arabic_pipeline import run_inference_single, run_batch, _preprocess_crop

# ─────────────────────────────────────────────────────────────────────────────
#  Character utilities
# ─────────────────────────────────────────────────────────────────────────────

_RE_ARABIC  = re.compile(r"[\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9]+")
_RE_WESTERN = re.compile(r"[0-9]+")
_RE_NOISE   = re.compile(r"[^\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9A-Za-z0-9]")


def _extract_arabic_letters(text: str) -> str:
    """Return only Arabic alphabet characters (no digits) from text."""
    # Arabic-Indic digits (٠-٩ / ۰-۹) are treated as digits and excluded
    letters_only = re.sub(r"[\u0660-\u0669\u06F0-\u06F9]", "", text)
    return "".join(_RE_ARABIC.findall(letters_only))


def _extract_digits(text: str) -> str:
    """Return only Western digit characters (0-9) from text."""
    return "".join(_RE_WESTERN.findall(text))


def _infer_letter_position(easy_text: str) -> str:
    """
    Determine whether Arabic letters appear BEFORE or AFTER digits in the
    EasyOCR output.  Returns 'before' or 'after'.

    On most Arabic licence plates the letter group is on the RIGHT (which in
    LTR string representation appears AFTER the digit group), but EasyOCR
    sometimes reverses segments.  We look at the index of the first Arabic
    character vs. the first digit.
    """
    first_ar  = next((i for i, c in enumerate(easy_text)
                      if "\u0600" <= c <= "\u06FF"), None)
    first_dig = next((i for i, c in enumerate(easy_text)
                      if c.isdigit()), None)

    if first_ar is None:
        return "after"   # no Arabic found – put letters (empty) after digits
    if first_dig is None:
        return "before"  # no digits in easy text – just put letters before

    # If Arabic chars come first in the string → 'before'
    return "before" if first_ar < first_dig else "after"


# ─────────────────────────────────────────────────────────────────────────────
#  Core merge function
# ─────────────────────────────────────────────────────────────────────────────

def merge_ocr_results(paddle_text: str, easy_text: str) -> str:
    """
    Merge PaddleOCR and EasyOCR outputs into a single plate string.

    Parameters
    ----------
    paddle_text : raw string from PaddleOCR  (accurate digits)
    easy_text   : raw string from EasyOCR    (accurate Arabic letters)

    Returns
    -------
    Merged plate string, e.g. "أبج 1234" or "1234 أبج"
    """
    digits  = _extract_digits(paddle_text)
    letters = _extract_arabic_letters(easy_text)

    # If EasyOCR found no letters fall back to whatever PaddleOCR returned
    if not letters:
        return digits or paddle_text

    # If PaddleOCR found no digits fall back to EasyOCR entirely
    if not digits:
        return easy_text

    position = _infer_letter_position(easy_text)

    if position == "before":
        merged = f"{letters} {digits}"
    else:
        merged = f"{digits} {letters}"

    return merged.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  High-level API
# ─────────────────────────────────────────────────────────────────────────────

def read_plate(
    image_path: str | Path,
    show: bool = False,
    verbose: bool = True,
) -> str:
    """
    Full pipeline: YOLO detection → dual OCR → smart merge.

    Parameters
    ----------
    image_path : path to the car image (full scene, not just the crop).
    show       : display the annotated image with the merged text.
    verbose    : print per-engine results before the merge.

    Returns
    -------
    Final merged plate string.
    """
    result = run_inference_single(
        image_path,
        ocr_engines=("paddleocr_ar", "easyocr_ar"),
        show=False,           # we'll annotate ourselves if show=True
        save_path=None,
    )

    paddle_text = result["ocr_results"].get("paddleocr_ar", "")
    easy_text   = result["ocr_results"].get("easyocr_ar",   "")
    merged      = merge_ocr_results(paddle_text, easy_text)

    if verbose:
        print(f"  PaddleOCR  → {paddle_text!r}")
        print(f"  EasyOCR    → {easy_text!r}")
        print(f"  ✅ Merged  → {merged!r}")

    if show and result["annotated_image"] is not None:
        annotated = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)

        # Overlay the merged text (use a Unicode-capable font via PIL)
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            pil_img  = PILImage.fromarray(result["annotated_image"])
            draw     = ImageDraw.Draw(pil_img)

            # Try to load a font that supports Arabic; fall back to default
            font = None
            for candidate in [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\tahoma.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]:
                if Path(candidate).exists():
                    font = ImageFont.truetype(candidate, size=32)
                    break
            if font is None:
                font = ImageFont.load_default()

            draw.text((10, 10), f"Plate: {merged}", fill=(0, 230, 255), font=font)
            annotated_rgb = np.array(pil_img)
            cv2.imshow("Smart Plate Reader", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ImportError:
            # Pillow not available – show without text overlay
            cv2.imshow("Smart Plate Reader", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return merged


def read_plates_batch(
    split: str = "val",
    max_images: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Run smart dual-OCR on a whole dataset split.

    Parameters
    ----------
    split      : 'train' or 'val'
    max_images : cap the number of images processed (None = all)
    verbose    : print per-image details

    Returns
    -------
    List of dicts: {image, paddle, easy, merged}
    """
    from arabic_pipeline import AR_DIR, _active_weights, _preprocess_crop, _build_engine_map
    from ultralytics import YOLO

    images_dir = AR_DIR / "images" / split
    all_imgs   = sorted(images_dir.glob("*.jpg"))
    if max_images:
        all_imgs = all_imgs[:max_images]

    weights = _active_weights()
    model   = YOLO(str(weights))
    engines = _build_engine_map()
    paddle_fn = engines["paddleocr_ar"]
    easy_fn   = engines["easyocr_ar"]

    records = []
    print(f"\n{'='*55}")
    print(f"  Smart Dual-OCR  |  split={split}  |  images={len(all_imgs)}")
    print(f"{'='*55}")

    for img_path in all_imgs:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        yolo_res = model.predict(str(img_path), conf=0.15, verbose=False)
        boxes    = yolo_res[0].boxes if yolo_res[0].boxes is not None else []

        if len(boxes) == 0:
            records.append({"image": img_path.name, "paddle": "", "easy": "", "merged": ""})
            if verbose:
                print(f"  [{img_path.name}] — no plate detected")
            continue

        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        crop = img_bgr[y1:y2, x1:x2]
        crop = _preprocess_crop(crop)

        paddle_text = paddle_fn(crop)
        easy_text   = easy_fn(crop)
        merged      = merge_ocr_results(paddle_text, easy_text)

        records.append({
            "image":  img_path.name,
            "paddle": paddle_text,
            "easy":   easy_text,
            "merged": merged,
        })

        conf = float(boxes.conf[best_idx])
        print(f"  [{img_path.name}]  det={conf:.2f}  "
              f"paddle={paddle_text!r}  easy={easy_text!r}  "
              f"→ {merged!r}")

    print(f"\nProcessed {len(records)} images.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Force UTF-8 output for Windows console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    DEFAULT_IMAGE = (
        r"C:\Users\Mohamed Walid\Desktop\Internship\Code"
        r"\Ar_Dataset_Split\images\val\0001.jpg"
    )

    args = sys.argv[1:]

    if "--batch" in args:
        split = "val"
        records = read_plates_batch(split=split, verbose=True)
        print(f"\nDone. {len(records)} plates processed.")

    else:
        img_path = args[0] if args else DEFAULT_IMAGE
        print(f"\nImage: {img_path}")
        merged = read_plate(img_path, show=False, verbose=True)
        print(f"\nFinal plate: {merged}")
