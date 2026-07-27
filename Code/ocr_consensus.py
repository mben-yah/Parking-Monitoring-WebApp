# -*- coding: utf-8 -*-
"""
ocr_consensus.py
────────────────
OCR Filtering and Character-Level Majority Voting Module.

Functions:
1. has_digit(text): Checks if a string contains at least one numeric digit (0-9 or ٠-٩).
2. validate_and_filter_plate(text): Returns clean plate text if valid and contains a digit, else "".
3. character_level_vote(plate_reads): Performs position-by-position majority voting on aligned plate reads.
4. align_and_vote_plates(plate_list): Groups similar plate reads and applies character-level majority consensus.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import List, Dict, Any, Tuple


# Regex pattern to match Western (0-9) and Eastern Arabic (٠-٩) digits
DIGIT_PATTERN = re.compile(r"[\d٠-٩]")


def has_digit(text: str | None) -> bool:
    """
    Returns True if text contains at least one numeric digit (0-9 or ٠-٩).
    Records without any digits are invalid license plates and should be eliminated.
    """
    if not text:
        return False
    return bool(DIGIT_PATTERN.search(text))


def validate_and_filter_plate(text: str | None) -> str:
    """
    Cleans plate text and eliminates records that do not contain a digit.
    Returns cleaned text or "" if invalid.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Reject strings with no digits
    if not has_digit(cleaned):
        return ""
    # Reject strings shorter than 2 characters
    if len(cleaned) < 2:
        return ""
    return cleaned


def character_level_vote(plate_reads: List[str]) -> str:
    """
    Performs character-level majority voting across multiple reads of a plate.

    Example:
        Input: [
            "DG368HP11",
            "DG368HPI1",   # I vs 1 confusion
            "DG368HP11",
            "DB368HP11"    # B vs G confusion
        ]
        Output: "DG368HP11" (Majority per position)
    """
    # Filter out empty or non-digit strings first
    valid_reads = [r.strip().upper() for r in plate_reads if validate_and_filter_plate(r)]
    if not valid_reads:
        return ""
    if len(valid_reads) == 1:
        return valid_reads[0]

    # Find modal (most common) length
    lengths = [len(r) for r in valid_reads]
    target_len = Counter(lengths).most_common(1)[0][0]

    # Filter reads that match target length
    aligned_reads = [r for r in valid_reads if len(r) == target_len]
    if not aligned_reads:
        aligned_reads = valid_reads

    # Character-by-character majority vote
    consensus_chars = []
    max_len = max(len(r) for r in aligned_reads)

    for i in range(max_len):
        col_chars = []
        for read in aligned_reads:
            if i < len(read):
                c = read[i]
                col_chars.append(c)
        if col_chars:
            # Common character confusion normalization if tie
            most_common = Counter(col_chars).most_common()
            winner = most_common[0][0]
            # Handle tie-breaker favoring digits over ambiguous letters in digit positions if applicable
            consensus_chars.append(winner)

    consensus_str = "".join(consensus_chars)
    return validate_and_filter_plate(consensus_str) or aligned_reads[0]


def group_and_vote_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups detection events by plate similarity and applies character-level voting consensus.

    Each detection dict contains:
      - plate_text
      - timestamp_s / timestamp
      - confidence
      - bbox, annotated_b64, etc.

    Returns aggregated unique plate records with character-level voted plate_text.
    """
    if not detections:
        return []

    # Filter out non-digit detections early
    filtered_dets = [
        d for d in detections
        if validate_and_filter_plate(d.get("plate_text", ""))
    ]

    if not filtered_dets:
        return []

    # Group detections by normalized base string (ignoring 1/I, 0/O minor variations for grouping)
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for d in filtered_dets:
        raw_text = d["plate_text"].strip().upper()
        # Key for grouping: coarse key replacing ambiguous chars
        coarse_key = (
            raw_text.replace("I", "1")
                    .replace("O", "0")
                    .replace("Z", "2")
                    .replace("S", "5")
        )
        if coarse_key not in groups:
            groups[coarse_key] = []
        groups[coarse_key].append(d)

    # Compute character-level consensus for each group
    result_list = []
    for key, items in groups.items():
        reads = [item["plate_text"] for item in items]
        voted_text = character_level_vote(reads)

        if not voted_text:
            continue

        first_seen = min(item.get("timestamp_s", item.get("timestamp", 0)) for item in items)
        last_seen  = max(item.get("timestamp_s", item.get("timestamp", 0)) for item in items)
        best_conf  = max(item.get("confidence", 0.0) for item in items)
        best_b64   = next((item["annotated_b64"] for item in items if "annotated_b64" in item), None)

        result_list.append({
            "plate_text":   voted_text,
            "raw_reads":    reads,
            "count":        len(items),
            "first_seen":   first_seen,
            "last_seen":    last_seen,
            "confidence":   round(best_conf, 4),
            "annotated_b64": best_b64,
        })

    # Sort by detection count descending
    result_list.sort(key=lambda x: -x["count"])
    return result_list
