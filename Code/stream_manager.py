# -*- coding: utf-8 -*-
"""
stream_manager.py
─────────────────
Manages a live MJPEG capture loop from IP Webcam (or any OpenCV-compatible URL).

Usage
-----
  sm = StreamManager()
  sm.connect("http://192.168.1.100:8080/video")
  frame_bytes = sm.latest_jpeg()     # annotated JPEG bytes
  sm.disconnect()

Thread model
------------
  _capture_thread : reads raw frames from the IP Webcam MJPEG stream
  _detect_thread  : YOLO + OCR on the latest frame every N frames
  Main thread     : serves MJPEG to browser via Flask generator
"""
from __future__ import annotations
import base64, re, threading, time, uuid, queue
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent

# ─── Auto-resolve best ELPD Commercial YOLO weights ───────────────────────────
def _resolve_elpd_weights() -> Path | None:
    """Auto-detect the highest-numbered ELPD Commercial dataset model weights available."""
    search = BASE_DIR / "runs" / "detect" / "runs" / "detect"
    best_pt, best_num = None, -1
    if search.exists():
        # 1. Primary: ELPD Commercial dataset models
        for d in search.iterdir():
            m = re.fullmatch(r"elpd_commercial_train(\d*)", d.name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                pt  = d / "weights" / "best.pt"
                if pt.exists() and num > best_num:
                    best_num, best_pt = num, pt
        if best_pt is not None:
            return best_pt

        # 2. Fallback: English AOLP models
        for d in search.iterdir():
            m = re.fullmatch(r"english_train(\d*)", d.name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                pt  = d / "weights" / "best.pt"
                if pt.exists() and num > best_num:
                    best_num, best_pt = num, pt
    return best_pt


class StreamManager:
    """Singleton-ish class that manages one live camera stream."""

    def __init__(self):
        self.device_name: str  = ""
        self.device_url:  str  = ""
        self.session_id:  str  = ""
        self.connected:   bool = False
        self.running:     bool = False

        self._cap:          cv2.VideoCapture | None = None
        self._raw_frame:    np.ndarray | None       = None
        self._ann_frame:    np.ndarray | None       = None
        self._jpeg_bytes:   bytes                   = b""
        self._lock:         threading.Lock          = threading.Lock()
        self._detect_lock:  threading.Lock          = threading.Lock()
        self._event_queue:  queue.Queue             = queue.Queue(maxsize=200)

        self._capture_thread: threading.Thread | None = None
        self._detect_thread:  threading.Thread | None = None

        self._model = None
        self._weights_path: Path | None = None

        self.frame_count:    int = 0
        self.detect_count:   int = 0
        self.plates_found:   int = 0

        # Detection callback → called with detection dict
        self._on_detection: Callable | None = None

        # Frame skip: run YOLO every N captured frames
        self.detect_every: int = 6
        self._ann_frame_expiry: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def connect(self, url: str, device_name: str = "IP Webcam") -> bool:
        if self.running:
            self.disconnect()

        self.device_url  = url.strip()
        self.device_name = device_name
        self.session_id  = str(uuid.uuid4())
        self.frame_count = self.detect_count = self.plates_found = 0
        self._ann_frame  = None
        self._ann_frame_expiry = 0.0

        # Load YOLO model (once)
        if self._model is None:
            wp = _resolve_elpd_weights()
            if wp is None:
                return False
            from ultralytics import YOLO
            self._model        = YOLO(str(wp))
            self._weights_path = wp
            print(f"[stream_manager] Active ELPD Commercial weights: {wp}")

        # Try opening the stream
        cap = cv2.VideoCapture(self.device_url)
        if not cap.isOpened():
            return False

        self._cap     = cap
        self.running  = True
        self.connected = True

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detect_thread  = threading.Thread(target=self._detect_loop,  daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()
        return True

    def disconnect(self):
        self.running   = False
        self.connected = False
        time.sleep(0.3)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._ann_frame = None

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg_bytes

    def pop_events(self) -> list[dict]:
        events = []
        try:
            while True:
                events.append(self._event_queue.get_nowait())
        except queue.Empty:
            pass
        return events

    def status(self) -> dict:
        return {
            "connected":     self.connected,
            "device_name":   self.device_name,
            "device_url":    self.device_url,
            "session_id":    self.session_id,
            "frame_count":   self.frame_count,
            "detect_count":  self.detect_count,
            "plates_found":  self.plates_found,
            "model":         str(self._weights_path.parent.parent.name) if self._weights_path else "—",
        }

    # ── Internal loops ─────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Continuously read raw frames; encode blank JPEG if stream drops."""
        fail_count = 0
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.5)
                continue
            ret, frame = self._cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 15:  # Auto reconnect after 15 dropped frames
                    try:
                        self._cap.release()
                        self._cap = cv2.VideoCapture(self.device_url)
                        fail_count = 0
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            fail_count = 0
            self.frame_count += 1
            with self._lock:
                self._raw_frame = frame.copy()
                # Check if annotated frame has expired (expire after 1.2 seconds)
                if self._ann_frame is not None and time.time() > self._ann_frame_expiry:
                    self._ann_frame = None

                # Use annotated frame if fresh, else live camera frame
                display = self._ann_frame if self._ann_frame is not None else frame
                ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self._jpeg_bytes = buf.tobytes()

    def _detect_loop(self):
        """Run YOLO + OCR every detect_every frames."""
        last_processed = -1
        # Lazy-load EasyOCR English
        reader = None

        while self.running:
            time.sleep(0.05)
            if self.frame_count - last_processed < self.detect_every:
                continue
            with self._lock:
                frame = self._raw_frame.copy() if self._raw_frame is not None else None
            if frame is None:
                continue

            last_processed = self.frame_count
            self.detect_count += 1

            try:
                # YOLO inference
                results = self._model.predict(frame, conf=0.15, verbose=False)
                boxes   = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue

                confs    = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                conf_val = float(confs[best_idx])
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

                H, W = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(W, x2), min(H, y2)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                # OCR
                if reader is None:
                    import easyocr
                    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                ocr_res    = reader.readtext(crop, detail=0)
                raw_text   = " ".join(ocr_res).strip().upper()
                from ocr_consensus import validate_and_filter_plate
                plate_text = validate_and_filter_plate(raw_text)

                if not plate_text:
                    continue

                self.plates_found += 1

                # Annotate frame
                ann = frame.copy()
                COLOR = (0, 230, 255)
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLOR, 3)
                label = f"{plate_text} [{conf_val:.2f}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                ly = max(y1 - th - 14, 0)
                cv2.rectangle(ann, (x1, ly), (x1 + tw + 10, ly + th + 10), COLOR, -1)
                cv2.putText(ann, label, (x1 + 5, ly + th + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

                with self._lock:
                    self._ann_frame = ann
                    self._ann_frame_expiry = time.time() + 1.2

                # Encode annotated crop for event
                _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                event = {
                    "type":        "plate",
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "plate_text":  plate_text,
                    "confidence":  round(conf_val, 4),
                    "bbox":        [int(x1), int(y1), int(x2), int(y2)],
                    "frame":       self.frame_count,
                    "annotated":   ann_b64,
                    "model":       str(self._weights_path.parent.parent.name) if self._weights_path else "—",
                    "session_id":  self.session_id,
                    "device":      self.device_name,
                }
                try:
                    self._event_queue.put_nowait(event)
                except queue.Full:
                    pass  # drop oldest if queue full — browser will catch up

                # Save to MongoDB
                try:
                    from mongodb_client import save_detection
                    save_detection(
                        plate_text  = plate_text,
                        confidence  = conf_val,
                        bbox        = [int(x1), int(y1), int(x2), int(y2)],
                        model_used  = str(self._weights_path.parent.parent.name) if self._weights_path else "unknown",
                        source      = "livestream",
                        source_url  = self.device_url,
                        session_id  = self.session_id,
                        extra       = {"device": self.device_name, "frame": self.frame_count},
                    )
                except Exception as _e:
                    print(f"[stream_manager] MongoDB save failed: {_e}")

            except Exception as exc:
                print(f"[stream_manager] Detect error: {exc}")


# ── Global singleton ────────────────────────────────────────────────────────────
stream_manager = StreamManager()
