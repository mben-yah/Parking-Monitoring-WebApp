#!/usr/bin/env python3
"""Evaluate a saved CRNN checkpoint on a validation list (labels.txt).
Prints CER (character error rate) and plate-level exact match accuracy.
"""
import argparse
import torch
from pathlib import Path
from Code.ocr_arabic.train_crnn_arabic import OCRDataset, CRNN, decode_preds
import editdistance


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    return data


def evaluate(ckpt, base_dir, val_list):
    data = load_checkpoint(ckpt)
    charset = data['charset']
    model = CRNN(len(charset))
    model.load_state_dict(data['model_state'])
    model.eval()

    ds = OCRDataset(base_dir, val_list, charset=charset)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=16, collate_fn=lambda b: (__import__('itertools').starmap(lambda x,y,z: (x,y,z), [])))
    # the evaluate here will reuse a simple loop without the collate_fn; we keep it minimal
    total_chars = 0
    total_ed = 0
    total_exact = 0
    with torch.no_grad():
        for img_rel, txt in ds.samples:
            img_path = Path(base_dir) / img_rel
            img = __import__('PIL').Image.open(img_path).convert('L')
            import torchvision.transforms as T
            img_t = T.functional.resize(img, (32, min(320, 32*6)))
            img_t = T.functional.to_tensor(img_t)
            img_t = (img_t - 0.5)/0.5
            img_t = img_t.unsqueeze(0)
            out = model(img_t)
            pred = decode_preds(out.cpu(), charset)[0]
            total_chars += len(txt)
            total_ed += editdistance.eval(pred, txt)
            total_exact += 1 if pred == txt else 0

    cer = total_ed / max(1, total_chars)
    exact_acc = total_exact / len(ds)
    print(f"CER: {cer:.4f}, Exact plate acc: {exact_acc:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--base-dir', default='.')
    p.add_argument('--val-list', required=True)
    args = p.parse_args()
    evaluate(args.ckpt, args.base_dir, args.val_list)
