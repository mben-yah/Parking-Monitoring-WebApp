
import sys; sys.path.insert(0, r'C:\Users\Mohamed Walid\Desktop\Internship\Code')
from arabic_ocr_pipeline import parse_morocco_plate, format_plate_display, REGIONAL_CODES

tests = [
    ('12345\u06280 6',   '\u0628',   '12345', '\u0628', 'Casablanca-Anfa'),
    ('99999\u06481',    '\u0648',   '99999', '\u0648', 'Rabat'),
    ('1234\u062744',    '\u062f',   '1234',  '\u062f', 'Errachidia'),
    ('678\u0647 53',    '\u0647',   '678',   '\u0647', 'Tétouan'),
    ('50000\u06406',    '\u0637',   '50000', '\u0637', 'Casablanca-Anfa'),
    ('42\u064a40',      '\u064a',   '42',    '\u064a', 'Tanger'),
]

# Rewrite with explicit arabic chars
tests2 = [
    ('12345' + '\u0628' + '6',    '\u0628',   '12345', '\u0628', 'Casablanca-Anfa'),
    ('99999' + '\u0648' + '1',    '\u0648',   '99999', '\u0648', 'Rabat'),
    ('1234'  + '\u062f' + '44',   '\u062f',   '1234',  '\u062f', 'Errachidia'),
    ('678'   + '\u0647' + '53',   '\u0647',   '678',   '\u0647', 'Tétouan'),
    ('50000' + '\u0637' + '6',    '\u0637',   '50000', '\u0637', 'Casablanca-Anfa'),
    ('42'    + '\u064a' + '40',   '\u064a',   '42',    '\u064a', 'Tanger'),
    # fallback: letter only in arabic_text
    ('12345 6',                   '\u0628',   '12345', '\u0628', 'Casablanca-Anfa'),
]

all_ok = True
for fast, ar, exp_seq, exp_letter, exp_city in tests2:
    p = parse_morocco_plate(fast, ar)
    disp = format_plate_display(p)
    ok = (p['left_seq'] == exp_seq and p['letter'] == exp_letter and p['city'] == exp_city)
    status = 'OK' if ok else 'FAIL'
    all_ok = all_ok and ok
    print(status, ' fast=', repr(fast), ' seq=', p['left_seq'],
          ' letter=', p['letter'], ' region=', p['region_code'],
          ' city=', p['city'], ' display=', disp)

print()
print('REGIONAL_CODES entries:', len(REGIONAL_CODES))
print('All tests passed:', all_ok)
