# Arabic normalization & confusion fixes

ARABIC_INDIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

COMMON_MAP = {
    # Arabic letter lookalikes to Latin/digits
    'أ': 'A', 'إ': 'I', 'آ': 'A',
    # Confusions: Arabic alef vs latin A/I, common OCR mistakes
}


def normalize_arabic_plate(s: str) -> str:
    """Normalize a plate string: map Arabic-indic digits to ASCII digits, remove diacritics,
    and apply common confusion substitutions.
    """
    if not s:
        return s
    out = []
    for ch in s:
        if ch in ARABIC_INDIC_DIGITS:
            out.append(ARABIC_INDIC_DIGITS[ch])
            continue
        if ch in COMMON_MAP:
            out.append(COMMON_MAP[ch])
            continue
        out.append(ch)
    # Post-process: remove spaces, punctuation commonly added by OCR
    res = ''.join(out)
    res = res.replace(' ', '').replace('-', '').replace('|', '')
    return res


def confidence_weighted_vote(candidates):
    """Given a list of (text, confidence) from different engines/frames,
    perform a simple character-level weighted vote.
    Returns the voted string.
    """
    if not candidates:
        return ""
    # Normalize all candidates to same length by padding with spaces on right
    texts = [normalize_arabic_plate(t[0]) for t in candidates]
    weights = [t[1] if t[1] is not None else 1.0 for t in candidates]
    max_len = max(len(t) for t in texts)
    import collections
    voted = []
    for i in range(max_len):
        counter = collections.Counter()
        for txt, w in zip(texts, weights):
            ch = txt[i] if i < len(txt) else ''
            counter[ch] += w
        best, _ = counter.most_common(1)[0]
        voted.append(best)
    return ''.join(voted).strip()
