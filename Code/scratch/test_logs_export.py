# -*- coding: utf-8 -*-
"""
Verification script for /logs and /logs/export endpoints in app.py
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

def run_tests():
    app.config["TESTING"] = True
    client = app.test_client()

    # Authenticate session
    with client.session_transaction() as sess:
        sess["user"] = {"username": "admin", "role": "ADMIN"}

    print("Running Log Export Tests...\n" + "="*50)

    # Test 1: HTML Log Viewer
    res = client.get("/logs")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "PlateVision — Live Server Logs" in res.get_data(as_text=True), "HTML title missing"
    assert "Export TXT" in res.get_data(as_text=True), "Export TXT button missing in HTML"
    print("✅ Test 1 Passed: /logs HTML Log Viewer returned successfully")

    # Test 2: TXT Export
    res = client.get("/logs?format=txt")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "attachment; filename=\"platevision_log_" in res.headers.get("Content-Disposition", ""), "Content-Disposition invalid for TXT"
    assert res.mimetype == "text/plain", f"Expected text/plain, got {res.mimetype}"
    txt_content = res.get_data(as_text=True)
    print(f"✅ Test 2 Passed: /logs?format=txt returned {len(txt_content.splitlines())} lines of plain text log")

    # Test 3: JSON Export
    res = client.get("/logs?format=json")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "attachment; filename=\"platevision_log_" in res.headers.get("Content-Disposition", ""), "Content-Disposition invalid for JSON"
    assert res.mimetype == "application/json", f"Expected application/json, got {res.mimetype}"
    json_data = json.loads(res.get_data(as_text=True))
    assert isinstance(json_data, list), "JSON log export should return a list of items"
    if json_data:
        assert "timestamp" in json_data[0] and "level" in json_data[0] and "message" in json_data[0], "JSON schema invalid"
    print(f"✅ Test 3 Passed: /logs?format=json returned {len(json_data)} structured log entries")

    # Test 4: CSV Export
    res = client.get("/logs?format=csv")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "attachment; filename=\"platevision_log_" in res.headers.get("Content-Disposition", ""), "Content-Disposition invalid for CSV"
    assert res.mimetype == "text/csv", f"Expected text/csv, got {res.mimetype}"
    csv_text = res.get_data(as_text=True)
    csv_reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(csv_reader)
    assert csv_reader.fieldnames == ["timestamp", "level", "message"], "CSV fieldnames mismatch"
    print(f"✅ Test 4 Passed: /logs?format=csv returned {len(rows)} CSV log rows")

    # Test 5: Scoped & Line Count Filters
    res_50 = client.get("/logs?format=txt&n=50")
    res_full = client.get("/logs?format=txt&scope=full")
    lines_50 = res_50.get_data(as_text=True).splitlines()
    lines_full = res_full.get_data(as_text=True).splitlines()
    assert len(lines_50) <= 50, f"Expected <= 50 lines, got {len(lines_50)}"
    assert len(lines_full) >= len(lines_50), "Full log should be >= n=50"
    print(f"✅ Test 5 Passed: /logs?format=txt&n=50 ({len(lines_50)} lines) vs scope=full ({len(lines_full)} lines)")

    # Test 6: Direct /logs/export Endpoint
    res_exp = client.get("/logs/export?format=json")
    assert res_exp.status_code == 200, f"Expected 200, got {res_exp.status_code}"
    assert "attachment; filename=\"platevision_log_" in res_exp.headers.get("Content-Disposition", ""), "Content-Disposition invalid for /logs/export"
    print("✅ Test 6 Passed: /logs/export endpoint functions correctly")

    print("\n" + "="*50 + "\nAll Log Export tests passed successfully! 🎉")

if __name__ == "__main__":
    run_tests()
