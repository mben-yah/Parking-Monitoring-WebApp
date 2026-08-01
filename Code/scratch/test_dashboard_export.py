# -*- coding: utf-8 -*-
"""
Verification script for /api/dashboard/export endpoint in app.py
"""
import sys
import json
import csv
import io
from pathlib import Path

# Add Code directory to sys.path
code_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(code_dir))

from app import app

def run_dashboard_tests():
    app.config["TESTING"] = True
    client = app.test_client()

    with client.session_transaction() as sess:
        sess["user"] = {"username": "admin", "role": "ADMIN"}

    print("Running Dashboard Export Tests...\n" + "="*50)

    # Test 1: Full JSON Export
    res = client.get("/api/dashboard/export?format=json")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "attachment; filename=\"platevision_dashboard_all_" in res.headers.get("Content-Disposition", "")
    data = json.loads(res.get_data(as_text=True))
    assert "basic" in data and "peak_hours" in data and "sensors" in data
    print("✅ Test 1 Passed: /api/dashboard/export?format=json returned complete JSON analytics")

    # Test 2: Full CSV Export
    res = client.get("/api/dashboard/export?format=csv")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "attachment; filename=\"platevision_dashboard_all_" in res.headers.get("Content-Disposition", "")
    assert res.mimetype == "text/csv"
    csv_text = res.get_data(as_text=True)
    assert "PLATEVISION DASHBOARD & SENSOR HEALTH REPORT" in csv_text
    print("✅ Test 2 Passed: /api/dashboard/export?format=csv returned full CSV report")

    # Test 3: Sensors CSV Export
    res = client.get("/api/dashboard/export?format=csv&type=sensors")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    csv_reader = csv.DictReader(io.StringIO(res.get_data(as_text=True)))
    assert csv_reader.fieldnames == ["name", "type", "lot", "status", "uptime_pct", "downtime_mins", "last_active"]
    print("✅ Test 3 Passed: /api/dashboard/export?format=csv&type=sensors returned sensor table CSV")

    # Test 4: Peak Hours CSV Export
    res = client.get("/api/dashboard/export?format=csv&type=peak_hours")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    csv_reader = csv.DictReader(io.StringIO(res.get_data(as_text=True)))
    assert csv_reader.fieldnames == ["hour", "time_window", "detections_count"]
    rows = list(csv_reader)
    assert len(rows) == 24, f"Expected 24 hourly rows, got {len(rows)}"
    print("✅ Test 4 Passed: /api/dashboard/export?format=csv&type=peak_hours returned 24 hourly rows")

    print("\n" + "="*50 + "\nAll Dashboard Export tests passed successfully! 🎉")

if __name__ == "__main__":
    run_dashboard_tests()
