# -*- coding: utf-8 -*-
"""
elpd_prep.py  —  Fast COCO→YOLO prep for ELPD commercial dataset.
Copies images from G:\\ELPD and writes YOLO label files locally.
Run this ONCE; it reports progress every 50 images.
"""
import json, random, shutil, cv2, numpy as np
from pathlib import Path

ELPD_COCO   = Path(r"G:\ELPD\COCO")
ANN_JSON    = ELPD_COCO / "annotations" / "instances_Train.json"
IMG_SRC     = ELPD_COCO / "images" / "Train"
OUT_DATASET = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code\ELPD_Commercial")

VAL_RATIO = 0.15
SEED      = 42

# ── Parse COCO ─────────────────────────────────────────────────────────────────
print("Loading annotation JSON...", flush=True)
with open(ANN_JSON, encoding="utf-8") as f:
    coco = json.load(f)

id_to_fn   = {img["id"]: img["file_name"]          for img in coco["images"]}
id_to_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

img_anns: dict = {}
for ann in coco["annotations"]:
    img_anns.setdefault(ann["image_id"], []).append(ann)

on_disk   = {p.name for p in IMG_SRC.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"}}
valid_ids = [iid for iid, fn in id_to_fn.items() if fn in on_disk and iid in img_anns]
print(f"Valid annotated images on disk: {len(valid_ids)}", flush=True)

# ── Split ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(valid_ids)
n_val   = max(1, int(len(valid_ids) * VAL_RATIO))
val_ids = set(valid_ids[:n_val])
trn_ids = set(valid_ids[n_val:])

for split in ("train", "val"):
    for sub in ("images", "labels"):
        (OUT_DATASET / sub / split).mkdir(parents=True, exist_ok=True)

# ── Copy + label ───────────────────────────────────────────────────────────────
def write_split(ids, split):
    ids = list(ids)
    total = len(ids)
    for i, iid in enumerate(ids, 1):
        if i % 50 == 0 or i == total:
            print(f"  [{split}] {i}/{total}", flush=True)
        fn     = id_to_fn[iid]
        w, h   = id_to_dims[iid]
        src    = IMG_SRC / fn
        dst_i  = OUT_DATASET / "images" / split / fn
        dst_l  = OUT_DATASET / "labels" / split / (Path(fn).stem + ".txt")
        shutil.copy2(src, dst_i)
        lines = []
        for ann in img_anns[iid]:
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw/2) / w
            yc = (by + bh/2) / h
            nw = bw / w
            nh = bh / h
            xc,yc,nw,nh = [max(0.,min(1.,v)) for v in (xc,yc,nw,nh)]
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        dst_l.write_text("\n".join(lines))
    return total

print(f"\nCopying {len(trn_ids)} train images from G: drive...", flush=True)
n_trn = write_split(trn_ids, "train")
print(f"Copying {len(val_ids)} val images from G: drive...", flush=True)
n_val_w = write_split(val_ids, "val")

# ── Augmentation (brightness/contrast/noise) ──────────────────────────────────
print(f"\nAugmenting {n_trn} train images...", flush=True)
trn_img = OUT_DATASET / "images" / "train"
trn_lbl = OUT_DATASET / "labels" / "train"
orig_imgs = list(trn_img.glob("*.png")) + list(trn_img.glob("*.jpg"))
aug_count = 0
for img_p in orig_imgs:
    lbl_p = trn_lbl / (img_p.stem + ".txt")
    if not lbl_p.exists(): continue
    img = cv2.imread(str(img_p))
    if img is None: continue
    alpha = random.uniform(0.8, 1.25)
    beta  = random.randint(-30, 30)
    aug   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    choice = random.choice(["blur","noise","none"])
    if choice == "blur":
        k = random.choice([3,5])
        aug = cv2.GaussianBlur(aug,(k,k),0)
    elif choice == "noise":
        noise = np.random.normal(0,10,aug.shape).astype(np.uint8)
        aug = cv2.add(aug, noise)
    cv2.imwrite(str(trn_img / (img_p.stem+"_aug"+img_p.suffix)), aug)
    (trn_lbl / (img_p.stem+"_aug.txt")).write_text(lbl_p.read_text())
    aug_count += 1

total_trn = len(list(trn_img.glob("*")))
print(f"Augmentation done: {len(orig_imgs)} orig + {aug_count} aug = {total_trn} total", flush=True)

# ── YAML ───────────────────────────────────────────────────────────────────────
yaml_txt = f"""path: {OUT_DATASET.as_posix()}
train: images/train
val:   images/val
nc: 1
names:
  0: license_plate
"""
yaml_path = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code\elpd_commercial.yaml")
yaml_path.write_text(yaml_txt, encoding="utf-8")
print(f"\nYAML written: {yaml_path}", flush=True)
print(f"\nDONE — {total_trn} train | {n_val_w} val", flush=True)
