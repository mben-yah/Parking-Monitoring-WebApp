# -*- coding: utf-8 -*-
"""
Final OCR engine comparison — writes results to a UTF-8 log file to avoid
Windows console encoding issues with Arabic characters.

Run:  C:\Anaconda\envs\yolo_env\python.exe ocr_test.py
"""
import cv2, warnings, json, sys, io
warnings.filterwarnings("ignore")

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

IMG_PATH  = r"C:\Users\Mohamed Walid\Desktop\Internship\Code\Ar_Dataset_Split\images\val\0001.jpg"
BBOX      = (320, 570, 664, 705)

img  = cv2.imread(IMG_PATH)
x1, y1, x2, y2 = BBOX
crop = img[y1:y2, x1:x2]
h, w = crop.shape[:2]
crop_big = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
print(f"Crop shape: {crop_big.shape}\n")

# ─── PaddleOCR ────────────────────────────────────────────────────────────────
print("=" * 50)
print("PaddleOCR (Arabic, PP-OCRv5)")
print("=" * 50)
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ar", device="cpu", enable_mkldnn=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.1, text_det_box_thresh=0.2,
                    text_rec_score_thresh=0.1)
    rgb = cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB)
    for res in ocr.predict(rgb):
        data      = res.json.get("res", {}) if isinstance(res.json, dict) else {}
        rec_texts = data.get("rec_texts", [])
        rec_scores= data.get("rec_scores", [])
        for text, score in zip(rec_texts, rec_scores):
            print(f"  [{score:.3f}] {text!r}")
        print("  FINAL:", " ".join(rec_texts))
except Exception as e:
    print("  ERROR:", e)

# ─── Surya ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Surya (DetectionPredictor + RecognitionPredictor)")
print("=" * 50)
try:
    from PIL import Image as PILImage
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    pil = PILImage.fromarray(cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB))
    det_pred    = DetectionPredictor()
    rec_pred    = RecognitionPredictor()
    layout_res  = det_pred([pil])
    rec_results = rec_pred(images=[pil], layout_results=layout_res)
    for page in rec_results:
        lines = getattr(page, "text_lines", None) or getattr(page, "lines", [])
        print(f"  {len(lines)} text lines found")
        for line in lines:
            txt  = getattr(line, "text", "").strip()
            conf = getattr(line, "confidence", 0)
            print(f"  [{conf:.3f}] {txt!r}")
except Exception as e:
    print("  ERROR:", e)

# ─── EasyOCR ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EasyOCR (Arabic)")
print("=" * 50)
try:
    import easyocr
    reader = easyocr.Reader(["ar"], gpu=False)
    rgb    = cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB)
    raw    = reader.readtext(rgb, detail=1, paragraph=False,
                             decoder="beamsearch", beamWidth=10)
    for (bbox, text, conf) in raw:
        print(f"  [{conf:.3f}] {text!r}")
    print("  FINAL:", " ".join(t for _, t, _ in raw))
except Exception as e:
    print("  ERROR:", e)
