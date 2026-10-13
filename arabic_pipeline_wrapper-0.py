# -*- coding: utf-8 -*-
"""Arabic pipeline compatibility wrapper.

Drop-in replacement for modified_pipeline.py, but wired to the Arabic pipeline
(arabic_pipeline.py) and defaulting to PaddleOCR instead of EasyOCR.

Exported names
--------------
- run_pipeline_multi        alias → arabic_pipeline.run_inference_single
- run_batch                 Arabic-aware batch runner (labels_dir-aware)
- run_pipeline_multi_with_accuracy
- visualize_ground_truth    alias → arabic_pipeline.visualize_ground_truth
- evaluate_ocr              alias → arabic_pipeline.evaluate_ocr  (if present)
"""

from __future__ import annotations

import difflib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Load arabic_pipeline from disk
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
PIPELINE_MODULE = BASE_DIR / "arabic_pipeline.py"

_spec   = importlib.util.spec_from_file_location("arabic_pipeline", str(PIPELINE_MODULE))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
#  Public aliases
# ─────────────────────────────────────────────────────────────────────────────

# Core single-image inference (YOLO + OCR)
run_pipeline_multi = _module.run_inference_single

# Visualisation / evaluation helpers
visualize_ground_truth = _module.visualize_ground_truth

# evaluate_ocr may not exist in every version of the module
evaluate_ocr = getattr(_module, "evaluate_ocr", None)


# ─────────────────────────────────────────────────────────────────────────────
#  Accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def _calc_accuracy(pred: str, truth: str) -> float:
    """Return a character-level similarity ratio between *pred* and *truth* (0–1)."""
    return difflib.SequenceMatcher(None, pred, truth).ratio()


# ─────────────────────────────────────────────────────────────────────────────
#  run_pipeline_multi_with_accuracy
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_multi_with_accuracy(
    image_path: str,
    ocr_engines: tuple[str, ...] = ("paddleocr_ar", "easyocr_ar"),
    show: bool = False,
    save_path: str | None = None,
    ground_truth_path: str | None = None,
) -> dict:
    """Run the Arabic pipeline and optionally compute per-engine OCR accuracy.

    Parameters
    ----------
    image_path        : path to the input image.
    ocr_engines       : OCR back-ends to use.

                        Available engines (defined in arabic_pipeline._build_engine_map):
                          'paddleocr_ar'       – PaddleOCR Arabic  ← NEW DEFAULT
                          'easyocr_ar'         – EasyOCR Arabic-only
                          'easyocr_ar_en'      – EasyOCR Arabic + English
                          'easyocr_ar_en_gpu'  – EasyOCR Arabic + English (GPU)
                          'surya_ar'           – Surya transformer, Arabic
                          'surya_ar_en'        – Surya transformer, Arabic + English
                          'tesseract_ar'       – Tesseract Arabic

    show              : display the annotated result.
    save_path         : save annotated image here (optional).
    ground_truth_path : path to a UTF-8 .txt file with the expected plate string.
                        When supplied, ``ocr_accuracy`` and ``ground_truth`` are
                        added to the returned dict.

    Returns
    -------
    dict with keys: bbox, confidence, ocr_results, annotated_image,
                    and optionally ocr_accuracy, ground_truth.
    """
    result = run_pipeline_multi(
        image_path=image_path,
        ocr_engines=ocr_engines,
        show=show,
        save_path=save_path,
    )

    if ground_truth_path and Path(ground_truth_path).is_file():
        truth = Path(ground_truth_path).read_text(encoding="utf-8").strip()
        result["ground_truth"] = truth
        result["ocr_accuracy"] = {
            eng: _calc_accuracy(txt, truth)
            for eng, txt in result.get("ocr_results", {}).items()
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  run_batch  –  Arabic-aware (separate labels/ tree, split-based paths)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    image_dir: str,
    ocr_engines: tuple[str, ...] = ("paddleocr_ar", "easyocr_ar"),
    show: bool = False,
    output_dir: str | None = None,
    log_path: str | None = None,
    sample_size: int | None = None,
    random_seed: int = 0,
    labels_dir: str | None = None,
) -> list[dict]:
    """Run the Arabic OCR pipeline on a directory of images.

    Parameters
    ----------
    image_dir    : folder containing image files (jpg/png/bmp/…).
    ocr_engines  : OCR back-ends to invoke per image.
                   Default is ``('paddleocr_ar', 'easyocr_ar')`` — PaddleOCR
                   is tried first as the primary non-EasyOCR engine.
    show         : display each annotated image during processing.
    output_dir   : where to save annotated images.
                   Defaults to the input image folder.
    log_path     : JSON-lines log file path.
                   Defaults to ``logs/batch_YYYYMMDD_HHMMSS.log``.
    sample_size  : process only this many images (random subset). ``None`` = all.
    random_seed  : RNG seed for reproducible sampling.
    labels_dir   : directory containing ground-truth ``.txt`` files.
                   For the Arabic dataset layout the labels live in a separate
                   ``labels/train`` or ``labels/val`` tree; pass that path here.
                   When ``None`` the code looks for a ``.txt`` next to each image
                   (English / flat layout).

    Returns
    -------
    list[dict]
        One result dict per processed image.
    """
    import os
    import random

    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise ValueError(f"'{image_dir}' is not a directory")

    # ── Gather images ─────────────────────────────────────────────────────────
    SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in SUFFIXES]
    all_imgs.sort()

    if sample_size is not None:
        random.seed(random_seed)
        all_imgs = random.sample(all_imgs, min(sample_size, len(all_imgs)))
        all_imgs.sort()

    # ── Output / log paths ────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else img_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"batch_arabic_{ts}.log"
    else:
        log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Per-engine accuracy accumulator ──────────────────────────────────────
    acc_accumulator: dict[str, list[float]] = {eng: [] for eng in ocr_engines}
    results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Arabic Batch Wrapper")
    print(f"  images  : {len(all_imgs)}")
    print(f"  engines : {list(ocr_engines)}")
    print(f"{'='*60}")

    for idx, img_path in enumerate(all_imgs, 1):
        save_path = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"

        # Resolve ground-truth label
        if labels_dir is not None:
            gt_path = Path(labels_dir) / img_path.with_suffix(".txt").name
        else:
            gt_path = img_path.with_suffix(".txt")   # flat / English layout

        try:
            result = run_pipeline_multi_with_accuracy(
                image_path=str(img_path),
                ocr_engines=ocr_engines,
                show=show,
                save_path=str(save_path),
                ground_truth_path=str(gt_path) if gt_path.is_file() else None,
            )

            result["image_path"]     = str(img_path)
            result["annotated_path"] = str(save_path)

            # Print OCR results
            print(f"[{idx:>4}/{len(all_imgs)}] {img_path.name}")
            for eng, txt in result.get("ocr_results", {}).items():
                print(f"           {eng:<22}: {txt!r}")

            # Collect accuracy if ground truth was available
            if "ocr_accuracy" in result:
                for eng, acc in result["ocr_accuracy"].items():
                    acc_accumulator.setdefault(eng, []).append(acc)
                    print(f"           accuracy [{eng}]: {acc:.2%}")

            # ── Log per-image entry ──────────────────────────────────────────
            _log_record(log_file, {
                "timestamp": datetime.now().isoformat(),
                "result":    {k: v for k, v in result.items() if k != "annotated_image"},
            })

        except Exception as exc:
            print(f"[{idx:>4}/{len(all_imgs)}] ERROR {img_path.name}: {exc}")
            _log_record(log_file, {
                "timestamp":  datetime.now().isoformat(),
                "image_path": str(img_path),
                "error":      str(exc),
            })
            continue

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        eng: (sum(vals) / len(vals) if vals else None)
        for eng, vals in acc_accumulator.items()
    }
    _log_record(log_file, {
        "timestamp":        datetime.now().isoformat(),
        "summary_accuracy": summary,
    })

    print(f"\n{'='*60}")
    print(f"  Processed : {len(results)} / {len(all_imgs)} images")
    for eng, avg in summary.items():
        if avg is not None:
            print(f"  Avg accuracy [{eng}]: {avg:.2%}")
    print(f"  Log       : {log_file}")
    print(f"{'='*60}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_record(log_file: Path, record: dict) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "run_pipeline_multi",
    "run_pipeline_multi_with_accuracy",
    "run_batch",
    "visualize_ground_truth",
    "evaluate_ocr",
]
