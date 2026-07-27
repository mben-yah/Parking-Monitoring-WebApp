# -*- coding: utf-8 -*-
"""
alert_engine.py
───────────────
Parking alert system for PlateVision.

After every plate detection, call `check_plate(...)`.
It will:
  1. Check if the plate is whitelisted  → OK, no alert
  2. Check if the plate is blacklisted  → BLACKLISTED alert (high priority)
  3. Otherwise                          → UNKNOWN alert

Alerts are stored in MongoDB and pushed via Server-Sent Events (SSE)
to any connected supervisor clients.
"""
from __future__ import annotations
import json
import queue
import threading
from datetime import datetime, timezone

# ── SSE broadcast queue ───────────────────────────────────────────────────────
# Each connected /api/parking/alert-stream client gets its own Queue.
_subscriber_lock: threading.Lock = threading.Lock()
_subscribers: list[queue.Queue] = []


def _broadcast(event: dict) -> None:
    """Push an alert event to all connected SSE clients."""
    payload = "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"
    with _subscriber_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def subscribe() -> queue.Queue:
    """Register a new SSE subscriber. Returns a Queue to read events from."""
    q: queue.Queue = queue.Queue(maxsize=50)
    with _subscriber_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _subscriber_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


# ── Core check ────────────────────────────────────────────────────────────────

def check_plate(
    plate_text: str,
    source: str,
    confidence: float | None = None,
    snapshot_b64: str | None = None,
    session_id: str = "",
    extra: dict | None = None,
) -> dict | None:
    """
    Check a detected plate against the whitelist / blacklist.
    Returns an alert dict if an alert was raised, else None.
    Called automatically by all detection endpoints.
    """
    if not plate_text or not plate_text.strip():
        return None

    try:
        import mongodb_client as db
        db.get_db()   # ensure connected

        wl = db.is_whitelisted(plate_text)
        if wl:
            return None   # ✅ known-good car — no alert

        bl = db.is_blacklisted(plate_text)
        alert_type = "blacklisted" if bl else "unknown"

        alert_id = db.create_alert(
            plate_text   = plate_text,
            alert_type   = alert_type,
            source       = source,
            confidence   = confidence,
            snapshot_b64 = snapshot_b64,
            session_id   = session_id,
            extra        = extra,
        )

        event = {
            "type":        "alert",
            "alert_id":    alert_id,
            "plate_text":  plate_text.strip().upper(),
            "alert_type":  alert_type,
            "source":      source,
            "confidence":  round(float(confidence), 4) if confidence is not None else None,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        _broadcast(event)
        return event

    except Exception as e:
        # Never crash a detection because of alert logic
        import logging
        logging.getLogger(__name__).warning(f"[alert_engine] check_plate failed: {e}")
        return None


def get_unacked_count() -> int:
    """Quick unacknowledged alert count (for badge on nav)."""
    try:
        import mongodb_client as db
        db.get_db()
        return db._col("alerts").count_documents({"acknowledged": False})
    except Exception:
        return 0
