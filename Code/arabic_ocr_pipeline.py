# -*- coding: utf-8 -*-
"""
arabic_ocr_pipeline.py
─────────────────────
Morocco license-plate recognition pipeline.

Stage 1 — Detection  : YOLOv8  (arabic_train* — auto-selects highest-numbered)
Stage 2 — OCR        : fast-plate-ocr (cct-s-v2-global-model) for digits
                       EasyOCR  Arabic for the middle letter
Stage 3 — Parser     : Morocco format  ← Number | Arabic-Letter | Regional-Code

Morocco plate format (white bg, black text, left→right):
  [left_seq]  [arabic_letter]  [region_code]
  e.g.  12345  ب  06
"""
from __future__ import annotations
import re, sys, io
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ─── Arabic letter map (series) ────────────────────────────────────────────────
ARABIC_LETTERS = {
    "أ": "A", "ا": "A",
    "ب": "B",
    "د": "D",
    "ه": "H",
    "و": "W",
    "ط": "T",
    "ي": "Y",
}
ARABIC_LETTER_SET = set(ARABIC_LETTERS.keys())

# ─── Regional code → city (official Morocco plate prefix) ─────────────────────
REGIONAL_CODES: dict[str, str] = {
    # Rabat-Salé-Kénitra region
    "1":  "Rabat",           "2":  "Salé",
    "3":  "Kénitra",         "4":  "Témara",
    "5":  "Khémisset",       "14": "Sidi Kacem",
    # Casablanca-Settat region
    "6":  "Casablanca-Anfa", "7":  "Mohammedia",
    "8":  "Settat",          "9":  "Berrechid",
    "10": "Ben Slimane",     "20": "El Jadida",
    # Béni Mellal-Khénifra
    "11": "Béni Mellal",     "12": "Khénifra",
    "13": "Azilal",          "15": "Fquih Ben Salah",
    # Marrakech-Safi
    "16": "Marrakech",       "17": "Safi",
    "18": "El Kelâa",        "19": "Essaouira",
    "21": "Youssoufia",      "30": "Chichaoua",
    # Fès-Meknès
    "22": "Fès",             "23": "Meknès",
    "24": "Ifrane",          "25": "El Hajeb",
    "26": "Sefrou",          "27": "Taounate",
    "28": "Taza",
    # Oriental
    "29": "Oujda",           "31": "Nador",
    "32": "Berkane",         "33": "Taourirt",
    "34": "Jerada",          "35": "Guercif",
    "36": "Driouch",         "37": "Figuig",
    "38": "Al Hoceïma",
    # Souss-Massa
    "39": "Agadir",          "40": "Tanger",
    "41": "Taroudant",          "42": "Chtouka-Aït Baha",
    "43": "Inezgane-Aït Melloul",
    # Drâa-Tafilalet
    "44": "Tétouan",      "44": "Errachidia",     "45": "Ouarzazate",
    "46": "Zagora",          "47": "Tinghir",
    "48": "Midelt",
    # Tanger-Tétouan-Al Hoceïma
    "49": "Tanger",          "50": "Tétouan",
    "51": "Larache",         "52": "Chefchaouen",
    "53": "Tétouan",        "54": "Mdiq-Fnideq",
    # Laâyoune-Sakia El Hamra
    "55": "Laâyoune",        "56": "Smara",
    "57": "Boujdour",        "58": "Es-Semara",
    # Guelmim-Oued Noun
    "59": "Guelmim",         "60": "Tan-Tan",
    "61": "Sidi Ifni",       "62": "Assa-Zag",
    # Dakhla-Oued Ed-Dahab
    "63": "Dakhla",          "64": "Aousserd",
    # Khouribga
    "65": "Khouribga",
    # Rabat (additional old codes still seen on roads)
    "66": "Skhirat",         "67": "Témara-Est",
}


# ─── Auto-resolve best Arabic YOLO weights ─────────────────────────────────────
def _resolve_arabic_weights() -> Path:
    import re as _re
    search = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    best_pt, best_num = None, -1
    if search.exists():
        for d in search.iterdir():
            m = _re.fullmatch(r"arabic_train(\d*)", d.name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                pt  = d / "weights" / "best.pt"
                if pt.exists() and num > best_num:
                    best_num, best_pt = num, pt
    # Fallback: check non-nested runs/detect/
    if best_pt is None:
        alt = BASE_DIR / "runs" / "detect"
        if alt.exists():
            for d in alt.iterdir():
                m = _re.fullmatch(r"arabic_train(\d*)", d.name)
                if m:
                    num = int(m.group(1)) if m.group(1) else 1
                    pt  = d / "weights" / "best.pt"
                    if pt.exists() and num > best_num:
                        best_num, best_pt = num, pt
    return best_pt  # can be None if no model exists


ARABIC_WEIGHTS = _resolve_arabic_weights()
print(f"[arabic_ocr_pipeline] Active YOLO: {ARABIC_WEIGHTS}")


# ─── fast-plate-ocr recognizer (lazy) ──────────────────────────────────────────
_fast_ocr = None

def _get_fast_ocr():
    global _fast_ocr
    if _fast_ocr is None:
        from fast_plate_ocr import LicensePlateRecognizer
        _fast_ocr = LicensePlateRecognizer("cct-s-v2-global-model")
    return _fast_ocr


# ─── EasyOCR Arabic reader (lazy) ─────────────────────────────────────────────
_easy_ar = None

def _get_easy_ar():
    global _easy_ar
    if _easy_ar is None:
        import easyocr
        _easy_ar = easyocr.Reader(["ar"], gpu=False, verbose=False)
    return _easy_ar


# ─── Plate parser ─────────────────────────────────────────────────────────────
# Format: NNNNN | X | CC
#   NNNNN — sequence number, LEFT  (1–99 999)
#   X     — Arabic letter,   MIDDLE (أ ب د ه و ط ي)
#   CC    — region code,     RIGHTMOST (1-2 digits)
#
# The Arabic letter acts as a DIVIDER: everything before it = sequence,
# everything after it = region code.  Never assume the letter is last.

def _first_arabic_letter(text: str) -> tuple[str | None, int]:
    """Return (letter, index) of the first Arabic series letter in text, or (None, -1)."""
    for i, ch in enumerate(text):
        if ch in ARABIC_LETTER_SET:
            return ch, i
    return None, -1


def parse_morocco_plate(fast_text: str, arabic_text: str) -> dict:
    """
    Parse Morocco plate into left_seq | letter | region_code.

    Strategy
    --------
    1. Search fast_text (left-to-right OCR) for the Arabic series letter.
       If found at position i:
         left_part  = fast_text[:i]  → digits = left_seq (all digits concatenated)
         right_part = fast_text[i+1:] → digits = region_code (take ≤2 digits)
    2. If letter not in fast_text, try arabic_text.
    3. If still no letter, fallback: longest digit group = left_seq,
       shortest digit group that is ≤ 2 chars = region_code.
    """
    letter, letter_idx = _first_arabic_letter(fast_text)

    if letter is not None:
        # Letter found in the fast-plate-ocr text — use positional split
        left_part  = fast_text[:letter_idx]
        right_part = fast_text[letter_idx + 1:]
        left_seq    = "".join(re.findall(r"\d+", left_part))
        # Region code: take the very first run of 1-2 digits on the right
        right_digits = re.findall(r"\d{1,2}", right_part)
        region_code  = right_digits[0] if right_digits else ""
    else:
        # Letter not in fast_text — check arabic_text for the letter
        letter, _ = _first_arabic_letter(arabic_text)
        # Fallback digit split: all digits from fast_text
        all_digits = re.findall(r"\d+", fast_text)
        if len(all_digits) >= 2:
            # Longest chunk = sequence (up to 5 digits), last short chunk = region
            by_len = sorted(enumerate(all_digits), key=lambda x: -len(x[1]))
            seq_idx = by_len[0][0]          # index of longest group
            left_seq = by_len[0][1]
            # Region: shortest group that appears AFTER seq_idx in original order
            region_candidates = [
                d for i, d in enumerate(all_digits)
                if i > seq_idx and len(d) <= 2
            ]
            region_code = region_candidates[0] if region_candidates else ""
        elif len(all_digits) == 1:
            left_seq, region_code = all_digits[0], ""
        else:
            left_seq, region_code = "", ""

    letter_lat = ARABIC_LETTERS.get(letter, "?") if letter else "?"
    city       = REGIONAL_CODES.get(region_code, "Unknown")

    return {
        "left_seq":     left_seq,
        "letter":       letter or "",
        "letter_latin": letter_lat,
        "region_code":  region_code,
        "city":         city,
        "raw_fast":     fast_text,
        "raw_arabic":   arabic_text,
    }


def format_plate_display(parsed: dict) -> str:
    """Render as  NNNNN | X | CC  — letter is always in the MIDDLE."""
    parts = []
    if parsed["left_seq"]:    parts.append(parsed["left_seq"])
    if parsed["letter"]:      parts.append(parsed["letter"])
    if parsed["region_code"]: parts.append(parsed["region_code"])
    return " | ".join(parts) if parts else "—"


# ─── Main inference function ───────────────────────────────────────────────────

def run_arabic_inference(image_path: str | Path, conf: float = 0.15) -> dict:
    """
    Detect + OCR a single image.
    Returns dict with keys:
      detected, bbox, confidence, parsed, plate_display,
      annotated_bgr (np.ndarray or None)
    """
    image_path = Path(image_path)
    result = {
        "detected":       False,
        "bbox":           None,
        "confidence":     None,
        "parsed":         None,
        "plate_display":  "—",
        "annotated_bgr":  None,
    }

    if ARABIC_WEIGHTS is None or not ARABIC_WEIGHTS.exists():
        result["error"] = "No Arabic YOLO model found"
        return result

    from ultralytics import YOLO
    model   = YOLO(str(ARABIC_WEIGHTS))
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        result["error"] = f"Cannot read image: {image_path}"
        return result

    H, W = img_bgr.shape[:2]
    yolo_res = model.predict(str(image_path), conf=conf, verbose=False)
    boxes    = yolo_res[0].boxes

    if boxes is None or len(boxes) == 0:
        return result

    # Pick highest-confidence box
    confs    = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    conf_val = float(confs[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    result["detected"]   = True
    result["confidence"] = conf_val
    result["bbox"]       = [int(x1), int(y1), int(x2), int(y2)]

    # Crop plate
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = img_bgr[y1c:y2c, x1c:x2c]

    if crop.size == 0:
        return result

    # ── fast-plate-ocr on full crop ────────────────────────────────────────────
    try:
        recognizer = _get_fast_ocr()
        fast_results = recognizer.run(crop)
        # run() returns list of (text, confidence) or just text depending on version
        if isinstance(fast_results, (list, tuple)) and fast_results:
            first = fast_results[0]
            fast_text = first if isinstance(first, str) else (first[0] if isinstance(first, (list, tuple)) else str(first))
        else:
            fast_text = str(fast_results)
    except Exception as e:
        fast_text = ""
        print(f"[fast-plate-ocr] Error: {e}")

    # ── EasyOCR Arabic on full crop (for letter extraction) ───────────────────
    try:
        reader    = _get_easy_ar()
        ar_result = reader.readtext(crop, detail=0)
        arabic_text = " ".join(ar_result).strip()
    except Exception as e:
        arabic_text = ""
        print(f"[EasyOCR-ar] Error: {e}")

    parsed = parse_morocco_plate(fast_text, arabic_text)
    result["parsed"]       = parsed
    result["plate_display"] = format_plate_display(parsed)

    # ── Annotate image ─────────────────────────────────────────────────────────
    ann = img_bgr.copy()
    COLOR = (0, 200, 255)
    cv2.rectangle(ann, (x1, y1), (x2, y2), COLOR, 3)
    label_str = f"{result['plate_display']} [{conf_val:.2f}]"
    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    ly1 = max(y1 - th - 14, 0)
    cv2.rectangle(ann, (x1, ly1), (x1 + tw + 10, ly1 + th + 10), COLOR, -1)
    cv2.putText(ann, label_str, (x1 + 5, ly1 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    result["annotated_bgr"] = ann

    return result


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        r = run_arabic_inference(path)
        print("Detected :", r["detected"])
        print("Plate    :", r["plate_display"])
        print("Parsed   :", r["parsed"])
    else:
        print("Usage: python arabic_ocr_pipeline.py <image_path>")
