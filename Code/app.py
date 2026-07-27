# -*- coding: utf-8 -*-
"""
Flask API backend for the License Plate Recognition web frontend.

Endpoints
---------
GET  /            -> serves index.html
GET  /video       -> serves video.html
GET  /logs        -> browser log viewer (last 300 lines)
POST /predict     -> accepts an image file, returns JSON
POST /predict_video -> streams SSE plate detections from a video
"""

import base64
import io
import os
import sys
import tempfile
import traceback
import logging
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows console so emoji/arrows don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import alert_engine

# ---------------------------------------------------------------------------
# Logging setup — writes to console AND logs/predict.log
# ---------------------------------------------------------------------------
BASE_DIR  = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
LOGS_DIR  = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE  = LOGS_DIR / "predict.log"

log = logging.getLogger("platevision")
log.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# File handler — always appends
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)

log.addHandler(fh)
log.addHandler(ch)

log.info("=" * 60)
log.info("PlateVision server starting")
log.info(f"Log file: {LOG_FILE}")
log.info("=" * 60)

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "recognition_pipeline", str(BASE_DIR / "recognition_pipeline.py")
)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)

run_pipeline = pipeline_module.run_pipeline_multi_ocr

# Also log which YOLO weights the English pipeline uses
try:
    _w = getattr(pipeline_module, "BEST_WEIGHTS", None) or getattr(pipeline_module, "WEIGHTS_PATH", None)
    log.info(f"English pipeline weights path: {_w}")
    weights_path = Path(str(_w))
    if weights_path.exists():
        log.info(f"  [OK] Weights file exists ({weights_path.stat().st_size // 1024} KB)")
    elif _w:
        log.warning(f"  [MISS] Weights file NOT FOUND at that path")
except Exception as _e:
    log.warning(f"Could not read weights path from pipeline: {_e}")

# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", template_folder="frontend")
CORS(app)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

import re as _re

def _get_all_models() -> list[tuple[int, str, Path]]:
    """
    Scan runs/detect/runs/detect/ for all english_trainN folders that have best.pt.
    Returns list of (number, name, path) sorted newest-first.
    """
    search = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    found  = []
    if search.exists():
        for d in search.iterdir():
            m = _re.fullmatch(r"english_train(\d*)", d.name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                pt  = d / "weights" / "best.pt"
                if pt.exists():
                    label = f"english_train{'2' if num == 2 else str(num)} (overnight)" if num == 2 else f"english_train{num} (augmented)"
                    found.append((num, d.name, pt, label))
    found.sort(key=lambda x: x[0], reverse=True)  # newest first
    return found

log.info(f"Available models: {[m[1] for m in _get_all_models()]}")


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/video")
def video_page():
    return send_from_directory("frontend", "video.html")


@app.route("/<path:filename>")
def static_files(filename):
    if filename == "logs":          # don't let this route eat /logs
        return log_viewer()
    return send_from_directory("frontend", filename)


# ---------------------------------------------------------------------------
# Log viewer
# ---------------------------------------------------------------------------
@app.route("/logs")
def log_viewer():
    """Return the last N lines of the log file as a styled HTML page."""
    n = int(request.args.get("n", 300))
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-n:]
    except FileNotFoundError:
        tail = ["(log file not found yet)\n"]

    def colour(line):
        if "ERROR" in line or "❌" in line:
            return f'<span style="color:#ef4444">{line}</span>'
        if "WARNING" in line or "WARN" in line or "⚠" in line:
            return f'<span style="color:#f59e0b">{line}</span>'
        if "✅" in line or "INFO" in line:
            return f'<span style="color:#a3e635">{line}</span>'
        if "DEBUG" in line:
            return f'<span style="color:#6b7a99">{line}</span>'
        return f'<span>{line}</span>'

    coloured = "".join(colour(l.rstrip()) + "\n" for l in tail)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>PlateVision Logs</title>
  <meta http-equiv="refresh" content="5">
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#080b12;color:#e8edf8;font-family:'JetBrains Mono',monospace;font-size:13px;padding:24px}}
    h1{{font-size:18px;font-weight:700;margin-bottom:16px;color:#3b82f6}}
    .meta{{font-size:11px;color:#6b7a99;margin-bottom:12px}}
    pre{{background:#0f1420;border:1px solid rgba(255,255,255,0.07);border-radius:10px;
         padding:20px;overflow-x:auto;line-height:1.7;white-space:pre-wrap;word-break:break-all}}
    .controls{{margin-bottom:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
    a.btn{{padding:7px 18px;border-radius:8px;font-size:13px;font-weight:600;text-decoration:none;
           background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);color:#60a5fa;transition:all .2s}}
    a.btn:hover{{background:rgba(59,130,246,0.25)}}
    .back{{color:#6b7a99;font-size:12px}}
  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
</head>
<body>
  <h1>🪵 PlateVision — Live Logs</h1>
  <div class="meta">Last {len(tail)} of {len(lines)} lines · auto-refreshes every 5s · {datetime.now().strftime('%H:%M:%S')}</div>
  <div class="controls">
    <a class="btn" href="/logs?n=100">Last 100</a>
    <a class="btn" href="/logs?n=300">Last 300</a>
    <a class="btn" href="/logs?n=1000">Last 1000</a>
    <a class="btn" href="/">← Image Mode</a>
    <a class="btn" href="/video">🎥 Video Mode</a>
  </div>
  <pre>{coloured}</pre>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Image prediction
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    req_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log.info(f"[{req_id}] -- /predict request received --")

    if "image" not in request.files:
        log.error(f"[{req_id}] No image file in request")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    log.info(f"[{req_id}] File: '{file.filename}'")

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed(file.filename):
        log.error(f"[{req_id}] Unsupported extension: {Path(file.filename).suffix}")
        return jsonify({"error": f"Unsupported file type: {Path(file.filename).suffix}"}), 400

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    log.info(f"[{req_id}] Saved to temp: {tmp_path}  ({Path(tmp_path).stat().st_size} bytes)")

    try:
        from ultralytics import YOLO
        import easyocr

        models = _get_all_models()   # sorted newest-first
        if not models:
            return jsonify({"error": "No trained models found"}), 500

        log.info(f"[{req_id}] Trying {len(models)} model(s): {[m[1] for m in models]}")

        plate_text    = ""
        confidence    = None
        bbox          = None
        annotated_b64 = None
        model_used    = None
        model_label   = None

        for num, name, pt, label in models:
            log.info(f"[{req_id}] Trying model: {name} ({pt})")
            try:
                _model    = YOLO(str(pt))
                _yolo_res = _model.predict(tmp_path, conf=0.15, verbose=False)
                _boxes    = _yolo_res[0].boxes

                if _boxes is None or len(_boxes) == 0:
                    log.info(f"[{req_id}]   {name}: no boxes — skipping")
                    continue

                confs    = _boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                conf_val = float(confs[best_idx])
                x1, y1, x2, y2 = _boxes.xyxy[best_idx].cpu().numpy().astype(int)

                log.info(f"[{req_id}]   {name}: box conf={conf_val:.3f} xyxy=[{x1},{y1},{x2},{y2}]")

                # Read image and crop
                img_bgr = cv2.imread(tmp_path)
                H, W    = img_bgr.shape[:2]
                crop    = img_bgr[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]

                if crop.size == 0:
                    log.warning(f"[{req_id}]   {name}: empty crop — skipping")
                    continue

                # OCR on crop (CPU EasyOCR, English)
                reader  = easyocr.Reader(['en'], gpu=False, verbose=False)
                ocr_res = reader.readtext(crop)
                raw     = " ".join(r[1] for r in ocr_res).strip()
                plate_text = raw.upper()

                log.info(f"[{req_id}]   {name}: OCR result = {plate_text!r}")

                confidence  = conf_val
                bbox        = [int(x1), int(y1), int(x2), int(y2)]
                model_used  = name
                model_label = label

                # Annotate image
                ann = img_bgr.copy()
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 229, 255), 3)
                lbl_str = f"{plate_text} [{name}] {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ly1 = max(y1 - th - 12, 0)
                cv2.rectangle(ann, (x1, ly1), (x1 + tw + 8, ly1 + th + 8), (0, 229, 255), -1)
                cv2.putText(ann, lbl_str, (x1 + 4, ly1 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                log.info(f"[{req_id}] SUCCESS via {name}: plate='{plate_text}'  conf={conf_val:.3f}")
                # Save to DB
                try:
                    from mongodb_client import save_detection as _save
                    _save(plate_text, conf_val, [int(x1),int(y1),int(x2),int(y2)],
                          model_used=name, source="image", source_url=file.filename or "")
                except Exception as _dbe:
                    log.warning(f"[{req_id}] DB save failed: {_dbe}")
                # Parking alert check
                try:
                    alert_engine.check_plate(plate_text, source="image",
                        confidence=conf_val, snapshot_b64=annotated_b64)
                except Exception:
                    pass
                break   # done — don't try next model

            except Exception as model_exc:
                log.warning(f"[{req_id}]   {name}: exception — {model_exc}")
                continue

        if not plate_text:
            log.warning(f"[{req_id}] FAIL: no model detected a plate")

        return jsonify({
            "plate_text":      plate_text,
            "confidence":      confidence,
            "bbox":            bbox,
            "annotated_image": annotated_b64,
            "model_used":      model_used,
            "model_label":     model_label,
        })

    except Exception as exc:
        log.error(f"[{req_id}] Exception in /predict:\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        log.info(f"[{req_id}] -- /predict done --")


# ---------------------------------------------------------------------------
# Arabic / Morocco plate prediction
# ---------------------------------------------------------------------------
_arabic_pipeline_mod = None

def _load_arabic_pipeline():
    global _arabic_pipeline_mod
    if _arabic_pipeline_mod is None:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("arabic_ocr_pipeline",
                    str(BASE_DIR / "arabic_ocr_pipeline.py"))
        _arabic_pipeline_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_arabic_pipeline_mod)
    return _arabic_pipeline_mod


@app.route("/predict_arabic", methods=["POST"])
def predict_arabic():
    req_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log.info(f"[{req_id}] -- /predict_arabic request received --")

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    log.info(f"[{req_id}] Arabic inference: {file.filename} ({Path(tmp_path).stat().st_size} bytes)")

    try:
        mod    = _load_arabic_pipeline()
        result = mod.run_arabic_inference(tmp_path, conf=0.15)

        log.info(f"[{req_id}] Detected: {result['detected']}  Plate: {result['plate_display']}")
        if result.get("parsed"):
            p = result["parsed"]
            log.info(f"[{req_id}]   left_seq={p['left_seq']}  letter={p['letter']}  "
                     f"region={p['region_code']}  city={p['city']}")
            log.info(f"[{req_id}]   raw_fast={p['raw_fast']!r}  raw_ar={p['raw_arabic']!r}")

        # Encode annotated image
        ann_b64 = None
        if result.get("annotated_bgr") is not None:
            _, buf = cv2.imencode(".jpg", result["annotated_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        model_name = str(mod.ARABIC_WEIGHTS) if mod.ARABIC_WEIGHTS else "N/A"

        # Parking alert check
        try:
            alert_engine.check_plate(result.get("plate_display",""), source="image",
                confidence=result.get("confidence"), snapshot_b64=ann_b64)
        except Exception:
            pass

        # Save to DB
        try:
            from mongodb_client import save_detection as _save
            _ar_model = Path(model_name).parent.parent.name if mod.ARABIC_WEIGHTS else "arabic_unknown"
            _save(
                result.get("plate_display",""),
                result.get("confidence"), result.get("bbox"),
                model_used=_ar_model, source="image",
                source_url=file.filename or "",
                extra={"mode":"arabic","parsed":result.get("parsed")},
            )
        except Exception as _dbe:
            log.warning(f"[{req_id}] Arabic DB save failed: {_dbe}")

        return jsonify({
            "detected":       result["detected"],
            "plate_display":  result["plate_display"],
            "parsed":         result.get("parsed"),
            "confidence":     result.get("confidence"),
            "bbox":           result.get("bbox"),
            "annotated_image": ann_b64,
            "model_used":     Path(model_name).parent.parent.name if mod.ARABIC_WEIGHTS else "—",
            "model_label":    f"Arabic YOLO — {Path(model_name).parent.parent.name}",
        })

    except Exception as exc:
        log.error(f"[{req_id}] /predict_arabic exception:\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        log.info(f"[{req_id}] -- /predict_arabic done --")


# ---------------------------------------------------------------------------
# Video prediction (SSE streaming)
# ---------------------------------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route("/predict_video", methods=["POST"])
def predict_video():

    req_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log.info(f"[{req_id}] ── /predict_video request received ─────────────────")

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not allowed_video(file.filename):
        return jsonify({"error": "Unsupported video type"}), 400

    mode        = request.form.get("mode", "english").lower()
    frame_skip  = int(request.form.get("frame_skip", 15))
    conf_thresh = float(request.form.get("conf", 0.15))
    ocr_engine  = request.form.get("engine", "paddleocr_ar")
    save_db     = request.form.get("save_db", "true").lower() == "true"

    log.info(f"[{req_id}] Video: '{file.filename}'  mode={mode}  frame_skip={frame_skip}  conf={conf_thresh}  engine={ocr_engine}  save_db={save_db}")

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    log.info(f"[{req_id}] Video saved: {tmp_path}  ({Path(tmp_path).stat().st_size//1024} KB)")

    def generate():
        try:
            from ultralytics import YOLO
            import importlib

            if mode == "arabic":
                import arabic_ocr_pipeline as ar_pipe
                importlib.reload(ar_pipe)
                weights = ar_pipe._resolve_arabic_weights() or ar_pipe.ARABIC_WEIGHTS
                log.info(f"[{req_id}] Video mode=ARABIC, weights={weights}")
                model = YOLO(str(weights))

                fast_ocr = None
                easy_ar  = None
                try:
                    fast_ocr = ar_pipe._get_fast_ocr()
                except Exception:
                    pass
                try:
                    easy_ar = ar_pipe._get_easy_ar()
                except Exception:
                    pass

                def _ocr_crop(crop):
                    crop_proc = ar_pipe._preprocess_crop(crop)
                    fast_text = ""
                    if fast_ocr:
                        try:
                            fast_text = fast_ocr.predict(crop_proc)
                        except Exception:
                            pass

                    ar_text = ""
                    if easy_ar:
                        try:
                            raw = easy_ar.readtext(crop)
                            ar_text = " ".join([t[1] for t in raw]).strip() if raw else ""
                        except Exception:
                            pass

                    parsed = ar_pipe.parse_morocco_plate(fast_text, ar_text)
                    return ar_pipe.format_plate_display(parsed)

            else:  # english mode
                import recognition_pipeline as en_pipe
                importlib.reload(en_pipe)
                weights = en_pipe._resolve_best_weights() or en_pipe.BEST_WEIGHTS
                log.info(f"[{req_id}] Video mode=ENGLISH, weights={weights}")
                model = YOLO(str(weights))

                def _ocr_crop(crop):
                    try:
                        return en_pipe._run_ocr_paddle(crop)
                    except Exception:
                        from arabic_pipeline import _build_engine_map
                        engines = _build_engine_map()
                        ocr_fn = engines.get(ocr_engine) or engines["paddleocr_ar"]
                        crop_proc = cv2.resize(crop, (0,0), fx=2, fy=2)
                        return ocr_fn(crop_proc)

            cap          = cv2.VideoCapture(tmp_path)
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration     = total_frames / fps

            log.info(f"[{req_id}] Video: fps={fps:.1f}  frames={total_frames}  duration={duration:.1f}s")
            yield f"data: {json.dumps({'type':'meta','fps':fps,'total_frames':total_frames,'duration':round(duration,2)}, ensure_ascii=False)}\n\n"

            unique_plates = {}
            all_frame_detections = []
            frame_idx     = 0
            plates_found  = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp_s = round(frame_idx / fps, 2)
                pct = round(frame_idx / max(total_frames, 1) * 100, 1)
                yield f"data: {json.dumps({'type':'progress','frame':frame_idx,'total':total_frames,'pct':pct,'timestamp_s':timestamp_s}, ensure_ascii=False)}\n\n"

                tmp_frame = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                frame_path = tmp_frame.name
                tmp_frame.close()
                cv2.imwrite(frame_path, frame)

                try:
                    yolo_res = model.predict(frame_path, conf=conf_thresh, verbose=False)
                    boxes    = yolo_res[0].boxes if yolo_res[0].boxes is not None else []
                    n_boxes  = len(boxes) if boxes is not None else 0

                    if n_boxes == 0:
                        log.debug(f"[{req_id}] frame={frame_idx} ts={timestamp_s}s → 0 boxes")
                    else:
                        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                        conf_val = float(boxes.conf[best_idx])

                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            from ocr_consensus import validate_and_filter_plate, group_and_vote_detections
                            raw_plate = _ocr_crop(crop)
                            plate_text = validate_and_filter_plate(raw_plate)
                            
                            if not plate_text:
                                continue
                                
                            log.info(f"[{req_id}]   Frame {frame_idx} OCR [{mode}] = {plate_text!r}")
                            plates_found += 1

                            ann = frame.copy()
                            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,230,255), 3)
                            cv2.putText(ann, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,230,255), 3)
                            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                            db_id = None
                            is_new = plate_text not in unique_plates
                            if is_new:
                                unique_plates[plate_text] = {"first_seen": timestamp_s, "last_seen": timestamp_s, "count": 1}
                                # Alert check on live frame
                                try:
                                    alert_engine.check_plate(plate_text, source="video",
                                        confidence=conf_val, snapshot_b64=ann_b64)
                                except Exception:
                                    pass
                            else:
                                unique_plates[plate_text]["last_seen"] = timestamp_s
                                unique_plates[plate_text]["count"]    += 1

                            event = {"type":"plate","frame":frame_idx,"timestamp_s":timestamp_s,
                                     "plate_text":plate_text,"confidence":conf_val,
                                     "bbox":[int(x1),int(y1),int(x2),int(y2)],"annotated_b64":ann_b64}
                            all_frame_detections.append(event)
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                finally:
                    try:
                        os.unlink(frame_path)
                    except Exception:
                        pass

                frame_idx += 1

            cap.release()
            
            # Perform position-by-position character-level majority voting on aggregated unique plates
            from ocr_consensus import group_and_vote_detections
            voted_unique_plates = group_and_vote_detections(all_frame_detections)

            # Save ONLY the final character-voted consensus plates to DB (if option enabled)
            if save_db:
                try:
                    from mongodb_client import save_detection as _vsave
                    for vp in voted_unique_plates:
                        db_id = _vsave(
                            vp["plate_text"],
                            vp.get("confidence", 0.85),
                            [0,0,0,0],
                            model_used=Path(str(weights)).parent.parent.name if weights else "video_model",
                            source="video",
                            source_url=file.filename or "video",
                            extra={"detections_count": vp.get("count", 1), "first_seen": vp.get("first_seen", 0), "mode": mode}
                        )
                        vp["db_id"] = db_id
                except Exception as err:
                    log.error(f"[{req_id}] Error saving consensus plates to DB: {err}")

            log.info(f"[{req_id}] Video done. Plates found: {plates_found}  Unique character-voted: {len(voted_unique_plates)}")

            yield f"data: {json.dumps({'type':'done','unique_plates':voted_unique_plates}, ensure_ascii=False)}\n\n"

        except Exception as exc:
            log.error(f"[{req_id}] Video exception:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type':'error','message':str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )




# ---------------------------------------------------------------------------
# Live stream (IP Webcam) + MongoDB
# ---------------------------------------------------------------------------
import importlib as _imp

def _get_stream():
    mod = _imp.import_module("stream_manager")
    return mod.stream_manager

def _get_mongo():
    return _imp.import_module("mongodb_client")


@app.route("/livestream")
def livestream_page():
    return send_from_directory("frontend", "livestream.html")


@app.route("/stream/connect", methods=["POST"])
def stream_connect():
    data        = request.get_json(force=True) or {}
    url         = data.get("url", "").strip()
    device_name = data.get("name", "IP Webcam").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    sm = _get_stream()
    ok = sm.connect(url, device_name)
    if not ok:
        return jsonify({"error": f"Could not open stream at {url}"}), 502
    try:
        _get_mongo().upsert_device(device_name, url)
    except Exception as _e:
        log.warning(f"MongoDB device save failed: {_e}")
    log.info(f"[livestream] Connected to {device_name} @ {url}  session={sm.session_id}")
    return jsonify({"ok": True, "session_id": sm.session_id, **sm.status()})


@app.route("/stream/disconnect", methods=["POST"])
def stream_disconnect():
    sm = _get_stream()
    sid = sm.session_id
    sm.disconnect()
    log.info(f"[livestream] Disconnected. session={sid}")
    return jsonify({"ok": True})


@app.route("/stream/status")
def stream_status():
    return jsonify(_get_stream().status())


@app.route("/stream/feed")
def stream_feed():
    sm = _get_stream()
    def generate():
        blank = None
        import time as _t
        while True:
            jpg = sm.latest_jpeg()
            if not jpg:
                if blank is None:
                    import numpy as _np
                    _f = _np.zeros((240, 320, 3), dtype=_np.uint8)
                    cv2.putText(_f, "Waiting for stream...", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                    _, _buf = cv2.imencode(".jpg", _f)
                    blank = _buf.tobytes()
                jpg = blank
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            _t.sleep(0.04)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream/events")
def stream_events():
    sm = _get_stream()
    def generate():
        import time as _t
        yield f"data: {json.dumps({'type':'connected','session':sm.session_id})}\n\n"
        while True:
            events = sm.pop_events()
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if not events:
                yield ": heartbeat\n\n"
            _t.sleep(0.2)
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream/detections")
def stream_detections():
    limit  = int(request.args.get("limit", 50))
    source = request.args.get("source", None)
    try:
        docs = _get_mongo().get_recent_detections(limit=limit, source=source)
        return jsonify({"detections": docs, "count": len(docs)})
    except Exception as exc:
        return jsonify({"error": str(exc), "detections": []}), 500


@app.route("/stream/devices", methods=["GET", "POST"])
def stream_devices():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        name = data.get("name", "IP Webcam")
        url  = data.get("url", "")
        if not url:
            return jsonify({"error": "url required"}), 400
        try:
            _id = _get_mongo().upsert_device(name, url)
            return jsonify({"ok": True, "id": _id})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    try:
        return jsonify({"devices": _get_mongo().list_devices()})
    except Exception as exc:
        return jsonify({"devices": [], "error": str(exc)})
# ---------------------------------------------------------------------------
# DB inspection endpoints
# ---------------------------------------------------------------------------
@app.route("/db")
def db_admin():
    return send_from_directory("frontend", "db.html")


@app.route("/db/stats")
def db_stats():
    try:
        from mongodb_client import get_stats
        return jsonify(get_stats())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/db/detections")
def db_detections():
    limit  = int(request.args.get("limit", 100))
    source = request.args.get("source", None)
    try:
        from mongodb_client import get_recent_detections
        docs = get_recent_detections(limit=limit, source=source)
        return jsonify({"detections": docs, "count": len(docs)})
    except Exception as exc:
        return jsonify({"error": str(exc), "detections": []}), 500


@app.route("/db/detections/<path:detection_id>", methods=["DELETE"])
def db_delete_detection(detection_id):
    try:
        from mongodb_client import delete_detection_by_id
        ok = delete_detection_by_id(detection_id)
        return jsonify({"ok": ok})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/db/detections/clear", methods=["POST", "DELETE"])
def db_clear_detections():
    try:
        source = request.args.get("source") or (request.get_json(force=True, silent=True) or {}).get("source")
        from mongodb_client import clear_detections
        n = clear_detections(source=source)
        return jsonify({"ok": True, "deleted": n})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ---------------------------------------------------------------------------
# Dashboard & Analytics endpoints
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard_page():
    return send_from_directory("frontend", "dashboard.html")


@app.route("/api/dashboard/stats")
def api_dashboard_stats():
    try:
        from mongodb_client import get_stats, get_peak_hours_stats, get_sensor_health_stats, parking_stats
        return jsonify({
            "basic": get_stats(),
            "peak_hours": get_peak_hours_stats(),
            "sensors": get_sensor_health_stats(),
            "parking": parking_stats()
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Supervisor / Parking endpoints
# ---------------------------------------------------------------------------
@app.route("/supervisor")
def supervisor_page():
    return send_from_directory("frontend", "supervisor.html")


@app.route("/api/parking/stats")
def parking_stats():
    try:
        from mongodb_client import parking_stats as _ps
        return jsonify(_ps())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/parking/whitelist", methods=["GET", "POST"])
def parking_whitelist():
    from mongodb_client import list_whitelist, add_to_whitelist
    if request.method == "POST":
        data  = request.get_json(force=True) or {}
        plate = data.get("plate_text", "").strip()
        notes = data.get("notes", "")
        owner = data.get("owner", "")
        if not plate:
            return jsonify({"error": "plate_text required"}), 400
        _id = add_to_whitelist(plate, notes=notes, owner=owner)
        log.info(f"[supervisor] Added to whitelist: {plate!r}")
        return jsonify({"ok": True, "id": _id})
    return jsonify({"whitelist": list_whitelist()})


@app.route("/api/parking/whitelist/<path:plate>", methods=["DELETE"])
def parking_whitelist_delete(plate):
    from mongodb_client import remove_from_whitelist
    n = remove_from_whitelist(plate)
    log.info(f"[supervisor] Removed from whitelist: {plate!r}")
    return jsonify({"ok": True, "deleted": n})


@app.route("/api/parking/blacklist", methods=["GET", "POST"])
def parking_blacklist():
    from mongodb_client import list_blacklist, add_to_blacklist
    if request.method == "POST":
        data   = request.get_json(force=True) or {}
        plate  = data.get("plate_text", "").strip()
        reason = data.get("reason", "")
        owner  = data.get("owner", "")
        if not plate:
            return jsonify({"error": "plate_text required"}), 400
        _id = add_to_blacklist(plate, reason=reason, owner=owner)
        log.info(f"[supervisor] Added to blacklist: {plate!r}")
        return jsonify({"ok": True, "id": _id})
    return jsonify({"blacklist": list_blacklist()})


@app.route("/api/parking/blacklist/<path:plate>", methods=["DELETE"])
def parking_blacklist_delete(plate):
    from mongodb_client import remove_from_blacklist
    n = remove_from_blacklist(plate)
    log.info(f"[supervisor] Removed from blacklist: {plate!r}")
    return jsonify({"ok": True, "deleted": n})


@app.route("/api/parking/alerts")
def parking_alerts():
    from mongodb_client import get_alerts
    limit     = int(request.args.get("limit", 100))
    unack_only = request.args.get("unack_only", "false").lower() == "true"
    try:
        return jsonify({"alerts": get_alerts(limit=limit, unack_only=unack_only)})
    except Exception as exc:
        return jsonify({"error": str(exc), "alerts": []}), 500


@app.route("/api/parking/alerts/<alert_id>/acknowledge", methods=["POST"])
def parking_ack_alert(alert_id):
    from mongodb_client import acknowledge_alert
    acknowledge_alert(alert_id)
    return jsonify({"ok": True})


@app.route("/api/parking/alerts/acknowledge-all", methods=["POST"])
def parking_ack_all():
    from mongodb_client import acknowledge_all_alerts
    n = acknowledge_all_alerts()
    log.info(f"[supervisor] Acknowledged {n} alerts")
    return jsonify({"ok": True, "count": n})


@app.route("/api/parking/alert-stream")
def parking_alert_stream():
    """SSE stream — pushes alert events to the supervisor page in real time."""
    q = alert_engine.subscribe()

    def generate():
        # Send initial keepalive
        yield 'data: {"type":"ping"}\n\n'
        try:
            while True:
                try:
                    payload = q.get(timeout=25)
                    yield payload
                except Exception:
                    # timeout — send keepalive ping
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            alert_engine.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PlateVision API")
    print("  Image mode  : http://127.0.0.1:5000")
    print("  Live stream : http://127.0.0.1:5000/livestream")
    print("  Video mode  : http://127.0.0.1:5000/video")
    print("  Supervisor  : http://127.0.0.1:5000/supervisor")
    print("  Live logs   : http://127.0.0.1:5000/logs")
    print(f"  Log file    : {LOG_FILE}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
