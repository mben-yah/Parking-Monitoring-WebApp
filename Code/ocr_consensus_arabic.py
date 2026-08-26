"""Improved OCR consensus module targeted for Arabic plates.

This module exposes functions to combine multiple OCR engine outputs (text + confidence)
and to apply Arabic-specific normalization and format priors.

It does not replace the existing ocr_consensus.py in the repo but provides a drop-in
`weighted_consensus` function that can be used by the Arabic pipeline.
"""
from Code.ocr_arabic.arabic_normalization import normalize_arabic_plate, confidence_weighted_vote


def weighted_consensus(reads, engine_weights=None):
    """
    reads: list of dicts [{ 'engine': 'paddle', 'text': '١٢٣٤', 'conf': 0.87 }, ...]
    engine_weights: optional dict mapping engine name -> multiplier weight
    Returns: normalized text + combined confidence
    """
    if engine_weights is None:
        engine_weights = {}
    candidates = []
    for r in reads:
        txt = r.get('text', '')
        conf = r.get('conf', 0.5)
        eng = r.get('engine')
        w = engine_weights.get(eng, 1.0)
        candidates.append((txt, conf * w))
    voted = confidence_weighted_vote(candidates)
    voted_norm = normalize_arabic_plate(voted)
    # estimate combined confidence as normalized weighted average
    total_w = sum([c[1] for c in candidates]) if candidates else 1.0
    avg_conf = sum([c[1] for c in candidates]) / total_w if total_w > 0 else 0.0
    return voted_norm, float(avg_conf)


if __name__ == '__main__':
    # quick demo
    reads = [
        {'engine': 'paddle', 'text': '١٢٣٤أ', 'conf': 0.9},
        {'engine': 'easyocr', 'text': '1234A', 'conf': 0.6},
    ]
    print(weighted_consensus(reads, engine_weights={'paddle': 1.2, 'easyocr': 0.8}))
