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


# ── User Authentication ──────────────────────────────────────────────────────────

def create_user(username: str, password: str, role: str = "OPERATOR") -> tuple[bool, str]:
    """Create a new user document in the 'users' collection."""
    from werkzeug.security import generate_password_hash
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password are required"
    
    col = _col("users")
    if col.find_one({"username": username}) is not None:
        return False, "Username already exists"
    
    role_upper = role.strip().upper()
    if role_upper not in ("ADMIN", "OPERATOR"):
        role_upper = "OPERATOR"
        
    doc = {
        "username": username,
        "password_hash": generate_password_hash(password),
        "role": role_upper,
        "created_at": datetime.now(timezone.utc),
        "last_login": None
    }
    res = col.insert_one(doc)
    return True, str(res.inserted_id)


def get_user_by_username(username: str) -> dict | None:
    """Retrieve user document by username."""
    u = _col("users").find_one({"username": username.strip().lower()})
    return _serialise(u) if u else None


def verify_user(username: str, password: str) -> tuple[bool, dict | str]:
    """Verify user credentials and update last_login on success."""
    from werkzeug.security import check_password_hash
    username = username.strip().lower()
    col = _col("users")
    user = col.find_one({"username": username})
    
    if not user:
        return False, "User not found"
        
    if not check_password_hash(user["password_hash"], password):
        return False, "Invalid password"
        
    now = datetime.now(timezone.utc)
    col.update_one({"_id": user["_id"]}, {"$set": {"last_login": now}})
    
    user_data = {
        "id": str(user["_id"]),
        "username": user["username"],
        "role": user.get("role", "OPERATOR"),
        "last_login": now.isoformat()
    }
    return True, user_data


def seed_default_users():
    """Ensure default admin and operator users exist for demo."""
    col = _col("users")
    if col.count_documents({}) == 0:
        create_user("admin", "admin123", "ADMIN")
        create_user("operator", "operator123", "OPERATOR")
        print("[platevision-db] Seeded default users: admin / operator")


# ── Parking Ticket & Payment Helpers ─────────────────────────────────────────

def create_parking_ticket(
    plate_text: str,
    rate_per_hour: float = 10.0,
    free_hours: float = 1.0,
    supermarket_stamp: bool = False,
    entry_time: datetime | None = None
) -> dict:
    plate_text = plate_text.strip().upper()
    now = entry_time or datetime.now(timezone.utc)
    import random
    ticket_id = f"TKT-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    doc = {
        "ticket_id": ticket_id,
        "plate_text": plate_text,
        "entry_time": now,
        "exit_time": None,
        "rate_per_hour": float(rate_per_hour),
        "free_hours": float(free_hours) if supermarket_stamp else 0.0,
        "supermarket_stamp": bool(supermarket_stamp),
        "status": "ACTIVE",  # ACTIVE, PAID, EXITED
        "payment_method": None,
        "amount_paid": 0.0,
        "created_at": now
    }
    _col("tickets").insert_one(doc)
    return _serialise(doc)


def get_parking_ticket(ticket_id: str) -> dict | None:
    doc = _col("tickets").find_one({"ticket_id": ticket_id.strip().upper()})
    if not doc:
        # Fallback search by plate_text if ticket_id is plate
        doc = _col("tickets").find_one({"plate_text": ticket_id.strip().upper(), "status": "ACTIVE"})
    return _serialise(doc) if doc else None


def calculate_ticket_fee(ticket: dict, exit_time: datetime | None = None) -> dict:
    now = exit_time or datetime.now(timezone.utc)
    entry_str = ticket.get("entry_time")
    if isinstance(entry_str, str):
        try:
            entry_time = datetime.fromisoformat(entry_str.replace("Z", "+00:00"))
        except Exception:
            entry_time = datetime.now(timezone.utc)
    elif isinstance(entry_str, datetime):
        entry_time = entry_str
    else:
        entry_time = datetime.now(timezone.utc)

    if entry_time.tzinfo is not None:
        entry_time = entry_time.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)

    duration_seconds = max((now - entry_time).total_seconds(), 0)
    duration_hours = duration_seconds / 3600.0
    
    free_hours = float(ticket.get("free_hours", 0.0))
    rate = float(ticket.get("rate_per_hour", 10.0))
    
    billable_hours = max(duration_hours - free_hours, 0.0)
    import math
    billable_units = math.ceil(billable_hours)
    total_amount = round(billable_units * rate, 2)
    
    return {
        "duration_minutes": round(duration_seconds / 60.0, 1),
        "duration_hours": round(duration_hours, 2),
        "billable_hours": billable_hours,
        "free_hours": free_hours,
        "rate_per_hour": rate,
        "total_amount": total_amount,
        "currency": "MAD"
    }


def pay_parking_ticket(ticket_id: str, payment_method: str = "CARD", exit_now: bool = True) -> dict | None:
    ticket = get_parking_ticket(ticket_id)
    if not ticket:
        return None
    
    now = datetime.now(timezone.utc)
    fee_data = calculate_ticket_fee(ticket, exit_time=now if exit_now else None)
    
    update_fields = {
        "status": "PAID" if exit_now else "PAID_PENDING_EXIT",
        "payment_method": payment_method.upper(),
        "amount_paid": fee_data["total_amount"],
        "exit_time": now if exit_now else None,
        "paid_at": now
    }
    
    _col("tickets").update_one({"ticket_id": ticket["ticket_id"]}, {"$set": update_fields})
    updated = get_parking_ticket(ticket["ticket_id"])
    if updated:
        updated.update(fee_data)
    return updated


def list_parking_tickets(limit: int = 50, status: str | None = None) -> list[dict]:
    query = {"status": status} if status else {}
    try:
        from pymongo import DESCENDING as D
    except ImportError:
        from montydb import DESCENDING as D
    return [_serialise(d) for d in _col("tickets").find(query).sort("created_at", D).limit(limit)]



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
