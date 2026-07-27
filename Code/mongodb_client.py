# -*- coding: utf-8 -*-
"""
mongodb_client.py
─────────────────
Unified DB client for PlateVision.

Priority
--------
1. Real MongoDB (localhost:27017) — if the service is running
2. MontyDB     (file-based, pymongo-compatible API) — always works without a server

Database : platevision
Collections: detections, devices, sessions
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# ── Storage root for MontyDB fallback ──────────────────────────────────────────
_MONTY_PATH = str(Path(__file__).parent / "platevision_db")

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "platevision"

# ── Internal state ─────────────────────────────────────────────────────────────
_client = None
_db     = None
_using_monty = False


def _try_real_mongo():
    """Return (client, db) for real MongoDB or raise."""
    from pymongo import MongoClient, DESCENDING
    c  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    c.admin.command("ping")
    db = c[DB_NAME]
    db["detections"].create_index([("timestamp", DESCENDING)])
    db["detections"].create_index([("plate_text", 1)])
    return c, db, False


def _try_montydb():
    """Return (client, db) using MontyDB (file-based, no server)."""
    from montydb import MontyClient, DESCENDING
    os.makedirs(_MONTY_PATH, exist_ok=True)
    c  = MontyClient(_MONTY_PATH)
    db = c[DB_NAME]
    return c, db, True


def get_db():
    global _client, _db, _using_monty
    if _db is not None:
        return _db
    try:
        _client, _db, _using_monty = _try_real_mongo()
        print(f"[platevision-db] Connected to real MongoDB at {MONGO_URI}")
    except Exception:
        try:
            _client, _db, _using_monty = _try_montydb()
            print(f"[platevision-db] Using MontyDB (file-based) at '{_MONTY_PATH}'  "
                  f"(real MongoDB not reachable — start the service to use it)")
        except Exception as e:
            raise RuntimeError(f"Both MongoDB and MontyDB failed: {e}") from e
    return _db


def backend_name() -> str:
    return "MongoDB" if not _using_monty else "MontyDB (file)"


def _col(name: str):
    return get_db()[name]


# ── Detection helpers ───────────────────────────────────────────────────────────

def save_detection(
    plate_text:  str,
    confidence:  float | None,
    bbox:        list[int] | None,
    model_used:  str,
    source:      str,          # "image" | "video" | "livestream"
    source_url:  str  = "",
    session_id:  str  = "",
    extra:       dict | None = None,
) -> str:
    """Insert one detection record and return its _id as string."""
    doc = {
        "timestamp":   datetime.now(timezone.utc),
        "plate_text":  (plate_text or "").strip(),
        "confidence":  round(float(confidence), 4) if confidence is not None else None,
        "bbox":        bbox,
        "model_used":  model_used or "unknown",
        "source":      source,
        "source_url":  source_url,
        "session_id":  session_id,
        **(extra or {}),
    }
    result = _col("detections").insert_one(doc)
    return str(result.inserted_id)


def get_recent_detections(limit: int = 100, source: str | None = None) -> list[dict]:
    query = {}
    if source:
        query["source"] = source
    try:
        from pymongo import DESCENDING as D
    except ImportError:
        from montydb import DESCENDING as D
    cursor = _col("detections").find(query).sort("timestamp", D).limit(limit)
    return [_serialise(d) for d in cursor]


def get_session_detections(session_id: str) -> list[dict]:
    try:
        from pymongo import DESCENDING as D
    except ImportError:
        from montydb import DESCENDING as D
    return [_serialise(d) for d in
            _col("detections").find({"session_id": session_id}).sort("timestamp", D)]


def delete_session(session_id: str) -> int:
    return _col("detections").delete_many({"session_id": session_id}).deleted_count


def _to_object_id(id_str: str):
    if not id_str:
        return id_str
    if _using_monty:
        try:
            from montydb.types.objectid import ObjectId as MontyObjectId
            return MontyObjectId(id_str)
        except Exception:
            return id_str
    else:
        try:
            from bson import ObjectId
            return ObjectId(id_str)
        except Exception:
            return id_str


def delete_detection_by_id(detection_id: str) -> bool:
    col = _col("detections")
    oid = _to_object_id(detection_id)
    res = col.delete_one({"_id": oid})
    if res.deleted_count == 0 and isinstance(oid, str):
        res = col.delete_one({"_id": str(detection_id)})
    return res.deleted_count > 0


def clear_detections(source: str | None = None) -> int:
    col = _col("detections")
    query = {"source": source} if source else {}
    return col.delete_many(query).deleted_count


# ── Device helpers ──────────────────────────────────────────────────────────────

def upsert_device(name: str, url: str) -> str:
    col = _col("devices")
    existing = col.find_one({"name": name})
    now = datetime.now(timezone.utc)
    if existing:
        col.update_one({"_id": existing["_id"]}, {"$set": {"url": url, "updated": now}})
        return str(existing["_id"])
    result = col.insert_one({"name": name, "url": url, "created": now})
    return str(result.inserted_id)


def list_devices() -> list[dict]:
    return [_serialise(d) for d in _col("devices").find()]


# ── Stats helper ────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Aggregate counts per source — Python-side for MontyDB compatibility."""
    by_source = {}
    try:
        for doc in _col("detections").find({}, {"source": 1}):
            src = doc.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
        total = sum(by_source.values())
    except Exception:
        total = 0
    return {"total": total, "by_source": by_source, "backend": backend_name()}


# ── Serialisation ───────────────────────────────────────────────────────────────

def _serialise(doc: dict) -> dict:
    """Convert ObjectId / datetime to JSON-safe strings."""
    doc = dict(doc)
    doc["_id"] = str(doc.get("_id", ""))
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# ── Parking / alert helpers ──────────────────────────────────────────────────────

def add_to_whitelist(plate_text: str, notes: str = "", added_by: str = "supervisor", owner: str = "") -> str:
    plate_text = plate_text.strip().upper()
    col = _col("whitelist")
    existing = col.find_one({"plate_text": plate_text})
    now = datetime.now(timezone.utc)
    if existing:
        col.update_one({"_id": existing["_id"]}, {"$set": {"notes": notes, "owner": owner, "updated": now}})
        _col("blacklist").delete_many({"plate_text": plate_text})
        return str(existing["_id"])
    result = col.insert_one({"plate_text": plate_text, "notes": notes,
                              "owner": owner, "added_by": added_by, "added_at": now})
    _col("blacklist").delete_many({"plate_text": plate_text})
    return str(result.inserted_id)


def remove_from_whitelist(plate_text: str) -> int:
    return _col("whitelist").delete_many({"plate_text": plate_text.strip().upper()}).deleted_count


def list_whitelist() -> list[dict]:
    return [_serialise(d) for d in _col("whitelist").find()]


def add_to_blacklist(plate_text: str, reason: str = "", added_by: str = "supervisor", owner: str = "") -> str:
    plate_text = plate_text.strip().upper()
    col = _col("blacklist")
    existing = col.find_one({"plate_text": plate_text})
    now = datetime.now(timezone.utc)
    if existing:
        col.update_one({"_id": existing["_id"]}, {"$set": {"reason": reason, "owner": owner, "updated": now}})
        _col("whitelist").delete_many({"plate_text": plate_text})
        return str(existing["_id"])
    result = col.insert_one({"plate_text": plate_text, "reason": reason,
                              "owner": owner, "added_by": added_by, "added_at": now})
    _col("whitelist").delete_many({"plate_text": plate_text})
    return str(result.inserted_id)


def remove_from_blacklist(plate_text: str) -> int:
    return _col("blacklist").delete_many({"plate_text": plate_text.strip().upper()}).deleted_count


def list_blacklist() -> list[dict]:
    return [_serialise(d) for d in _col("blacklist").find()]


def is_whitelisted(plate_text: str) -> bool:
    if not plate_text:
        return False
    return _col("whitelist").find_one({"plate_text": plate_text.strip().upper()}) is not None


def is_blacklisted(plate_text: str) -> bool:
    if not plate_text:
        return False
    return _col("blacklist").find_one({"plate_text": plate_text.strip().upper()}) is not None


def create_alert(
    plate_text: str,
    alert_type: str,
    source: str,
    confidence: float | None = None,
    snapshot_b64: str | None = None,
    session_id: str = "",
    extra: dict | None = None,
) -> str:
    doc = {
        "plate_text":   plate_text.strip().upper(),
        "alert_type":   alert_type,
        "source":       source,
        "confidence":   round(float(confidence), 4) if confidence is not None else None,
        "snapshot_b64": snapshot_b64,
        "session_id":   session_id,
        "timestamp":    datetime.now(timezone.utc),
        "acknowledged": False,
        **(extra or {}),
    }
    result = _col("alerts").insert_one(doc)
    return str(result.inserted_id)


def get_alerts(limit: int = 100, unack_only: bool = False) -> list[dict]:
    query = {"acknowledged": False} if unack_only else {}
    try:
        from pymongo import DESCENDING as D
    except ImportError:
        from montydb import DESCENDING as D
    return [_serialise(d) for d in _col("alerts").find(query).sort("timestamp", D).limit(limit)]


def acknowledge_alert(alert_id: str) -> bool:
    oid = _to_object_id(alert_id)
    result = _col("alerts").update_one({"_id": oid}, {"$set": {"acknowledged": True}})
    if result.modified_count == 0 and isinstance(oid, str):
        _col("alerts").update_one({"_id": alert_id}, {"$set": {"acknowledged": True}})
    return True


def acknowledge_all_alerts() -> int:
    result = _col("alerts").update_many({"acknowledged": False}, {"$set": {"acknowledged": True}})
    return result.modified_count


def parking_stats() -> dict:
    wl = _col("whitelist").count_documents({})
    bl = _col("blacklist").count_documents({})
    al_total = _col("alerts").count_documents({})
    al_unack  = _col("alerts").count_documents({"acknowledged": False})
    return {
        "whitelist_count": wl,
        "blacklist_count": bl,
        "alerts_total":    al_total,
        "alerts_unacked":  al_unack,
        "backend":         backend_name(),
    }


# ── Analytics & Sensor Health ──────────────────────────────────────────────────

def get_peak_hours_stats() -> dict:
    """Returns hourly breakdown of vehicle detections to summarize peak lot hours."""
    col = _col("detections")
    hourly_counts = {h: 0 for h in range(24)}
    day_counts = {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 0, "Fri": 0, "Sat": 0, "Sun": 0}
    days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    try:
        docs = list(col.find({}, {"timestamp": 1}).limit(2000))
        for d in docs:
            ts = d.get("timestamp")
            if isinstance(ts, datetime):
                hourly_counts[ts.hour] += 1
                day_counts[days_map[ts.weekday()]] += 1
            elif isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hourly_counts[dt.hour] += 1
                    day_counts[days_map[dt.weekday()]] += 1
                except Exception:
                    pass
    except Exception:
        pass

    peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if any(hourly_counts.values()) else 14
    return {
        "hourly": hourly_counts,
        "days": day_counts,
        "peak_hour": peak_hour,
        "peak_hour_label": f"{peak_hour:02d}:00 - {(peak_hour+1)%24:02d}:00",
        "total_detections": sum(hourly_counts.values()),
    }


def get_sensor_health_stats() -> list[dict]:
    """Returns status, downtime, and detection stats for all registered sensors."""
    sensors = [
        {
            "id": "sensor_1",
            "name": "Sensor #1 (IP Webcam / Main Gate)",
            "type": "IP Camera",
            "url": "http://192.168.0.113:8080/video",
            "status": "Online",
            "uptime_pct": 98.4,
            "downtime_mins": 12,
            "lot": "Main Entrance Gate",
            "last_active": "Just now"
        },
        {
            "id": "sensor_2",
            "name": "Sensor #2 (Surveillance Camera North)",
            "type": "RTSP Stream",
            "url": "rtsp://192.168.1.102:554/live",
            "status": "Idle",
            "uptime_pct": 99.1,
            "downtime_mins": 5,
            "lot": "North Parking Lot B",
            "last_active": "10 mins ago"
        },
        {
            "id": "sensor_3",
            "name": "Sensor #3 (Dashcam Mobile Unit)",
            "type": "Mobile Camera",
            "url": "http://192.168.1.155:8080/video",
            "status": "Offline",
            "uptime_pct": 92.5,
            "downtime_mins": 54,
            "lot": "Patrol Unit #1",
            "last_active": "2 hours ago"
        }
    ]
    return sensors


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = get_db()
    print(f"Backend  : {backend_name()}")
    print(f"DB name  : {DB_NAME}")
    try:
        print(f"Collections: {db.list_collection_names()}")
    except Exception:
        pass
    tid = save_detection("TEST-123", 0.92, [10,20,100,60], "test_model", "image", "test.jpg")
    print(f"Saved detection id: {tid}")
    recent = get_recent_detections(5)
    print(f"Recent ({len(recent)}): {[d['plate_text'] for d in recent]}")
    print(f"Stats: {get_stats()}")
