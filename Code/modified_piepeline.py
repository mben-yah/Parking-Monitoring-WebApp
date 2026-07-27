# -*- coding: utf-8 -*-
"""Compatibility wrapper for the modified pipeline.

This module provides the names expected by existing user scripts:
- `run_pipeline_multi` (alias for `run_pipeline_multi_ocr`)
- `visualize_ground_truth`
- `evaluate_ocr`

It simply re‑exports the implementations from `recognition_pipeline.py`.
"""

from pathlib import Path

# Resolve the path to the main pipeline module
BASE_DIR = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
PIPELINE_MODULE = BASE_DIR / "recognition_pipeline.py"

# Import the required symbols
import importlib.util
spec = importlib.util.spec_from_file_location("recognition_pipeline", str(PIPELINE_MODULE))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore

# Alias the multi‑OCR pipeline function to the old name expected by the user
run_pipeline_multi = module.run_pipeline_multi_ocr

# Re‑export the visualization and evaluation utilities
import json
from datetime import datetime

def run_batch(
    image_dir: str,
    ocr_engines=('easyocr', 'easyocr2'),
    show: bool = False,
    output_dir: str = None,
    log_path: str = None,
    sample_size: int = None,          # number of images to process; None = all
    random_seed: int = 0,
    labels_dir: str = None,           # separate labels folder (e.g. Arabic dataset)
):
    
    """Run the OCR pipeline on a (possibly sampled) subset of images.

    Parameters
    ----------
    image_dir : str
        Directory containing image files (jpg/png/bmp/etc.).
    ocr_engines : tuple
        Which OCR back‑ends to invoke for each image.
    show : bool, default False
        Whether to display the annotated image during processing.
    output_dir : str, optional
        Folder where annotated images will be saved. If ``None`` the original
        image folder is used.
    log_path : str, optional
        Path to a JSON‑lines log file. If omitted a timestamped file
        ``logs/batch_YYYYMMDD_HHMMSS.log`` will be created.
    sample_size : int, optional
        Process only the first ``sample_size`` images (or a random subset if
        ``sample_size`` is provided). ``None`` processes the whole folder.
    random_seed : int, default 0
        Seed for the random sampler when ``sample_size`` is used.
    labels_dir : str, optional
        Directory containing ground‑truth ``.txt`` files.  If ``None``, the
        code looks for a ``.txt`` file next to each image (English dataset
        layout).  Set this to the corresponding ``labels/train`` or
        ``labels/val`` folder for the Arabic dataset layout, where labels live
        in a separate tree from images.

    Returns
    -------
    list[dict]
        A list of result dictionaries, one per processed image.
    """
    import os, random
    from pathlib import Path

    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise ValueError(f"'{image_dir}' is not a directory")

    # -----------------------------------------------------------------
    # Gather image list and optionally sample a subset
    # -----------------------------------------------------------------
    all_imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    if sample_size is not None:
        random.seed(random_seed)
        all_imgs = random.sample(all_imgs, min(sample_size, len(all_imgs)))

    # Resolve output directory
    out_dir = Path(output_dir) if output_dir else img_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare log file
    if log_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"batch_{ts}.log"
    else:
        log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    # Store per‑engine accuracies for summary statistics
    acc_accumulator = {eng: [] for eng in ocr_engines}

    for img_path in all_imgs:
        save_path = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
        # Resolve the ground-truth label file
        if labels_dir is not None:
            gt_path = Path(labels_dir) / img_path.with_suffix('.txt').name
        else:
            gt_path = img_path.with_suffix('.txt')   # English layout: label next to image
        try:
            result = run_pipeline_multi_with_accuracy(
                image_path=str(img_path),
                ocr_engines=ocr_engines,
                show=show,
                save_path=str(save_path),
                ground_truth_path=str(gt_path) if gt_path.is_file() else None,
            )
            print(f"OCR results for {img_path.name}: {result.get('ocr_results')}")
            # Meta‑info
            result["image_path"] = str(img_path)
            result["annotated_path"] = str(save_path)

            # Collect accuracies if they were computed
            if "ocr_accuracy" in result:
                for eng, acc in result["ocr_accuracy"].items():
                    acc_accumulator.setdefault(eng, []).append(acc)
                    print(f"Accuracy for {img_path.name} - {eng}: {acc:.2%}")

            # Write per‑image entry to log
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "result": result,
                }, f, ensure_ascii=False)
                f.write("\n")
        except Exception as exc:
            # Log failure but continue
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "image_path": str(img_path),
                    "error": str(exc),
                }, f, ensure_ascii=False)
                f.write("\n")
            continue

        results.append(result)

    # -----------------------------------------------------------------
    # Write a summary of average accuracies per engine (if any)
    # -----------------------------------------------------------------
    summary = {
        eng: (sum(vals) / len(vals) if vals else None)
        for eng, vals in acc_accumulator.items()
    }

    with open(log_file, "a", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary_accuracy": summary,
        }, f, ensure_ascii=False)
        f.write("\n")

    return results

# -------------------------------------------------------------
#  Wrapper that also returns character‑level OCR accuracy
# -------------------------------------------------------------
import difflib

def _calc_accuracy(pred: str, truth: str) -> float:
    """Return a similarity ratio between *pred* and *truth* (0‑1)."""
    return difflib.SequenceMatcher(None, pred, truth).ratio()

def run_pipeline_multi_with_accuracy(
    image_path: str,
    ocr_engines=('easyocr', 'easyocr2'),
    show: bool = False,
    save_path: str = None,
    ground_truth_path: str = None,
):
    """Run the pipeline and (optionally) compute OCR accuracy.

    If *ground_truth_path* points to a text file containing the expected plate
    string, the function adds an ``ocr_accuracy`` entry to the result dict.
    The accuracy is measured per OCR engine using a simple character‑level
    similarity (difflib.SequenceMatcher).  Values are in the range ``0.0‑1.0``.
    """
    # Core pipeline execution
    result = run_pipeline_multi(
        image_path=image_path,
        ocr_engines=ocr_engines,
        show=show,
        save_path=save_path,
    )

    # If a ground‑truth file is supplied, compute accuracy per engine
    if ground_truth_path and Path(ground_truth_path).is_file():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            truth = f.read().strip()
        accuracies = {}
        for eng, txt in result.get("ocr_results", {}).items():
            accuracies[eng] = _calc_accuracy(txt, truth)
        result["ocr_accuracy"] = accuracies
        result["ground_truth"] = truth
    return result

visualize_ground_truth = module.visualize_ground_truth
evaluate_ocr = module.evaluate_ocr

__all__ = ["run_pipeline_multi", "visualize_ground_truth", "evaluate_ocr", "run_batch"]
