#!/usr/bin/env python3
"""Lightweight CRNN training script for Arabic license plate OCR.

This script implements a small CRNN (convolutional encoder + BiLSTM + CTC) and training loop.
It expects a `labels.txt` where each line is:
    relative/path/to/image.png\tTRANSCRIPTION

Notes:
- Designed to be a starting point. Tune the architecture, augmentations and optimizer for better accuracy.
- Uses CTC loss. Character set is built from training labels.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import editdistance


class OCRDataset(Dataset):
    def __init__(self, base_dir, list_file, charset=None, img_h=32, max_w=320):
        self.base = Path(base_dir)
        self.samples = []
        with open(list_file, 'r', encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln: continue
                img_rel, txt = ln.split('\t')
                self.samples.append((img_rel, txt))
        self.img_h = img_h
        self.max_w = max_w
        self.transforms = transforms.Compose([
            transforms.Resize((img_h, int(img_h*6))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if charset is None:
            charset = set()
            for _, t in self.samples:
                for ch in t:
                    charset.add(ch)
            self.charset = ''.join(sorted(charset))
        else:
            self.charset = charset
        self.char_to_idx = {c:i+1 for i,c in enumerate(self.charset)}  # 0 reserved for blank

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def __getitem__(self, idx):
        rel, txt = self.samples[idx]
        img_path = self.base / rel
        img = Image.open(img_path).convert('L')
        img = transforms.functional.resize(img, (self.img_h, min(self.max_w, int(self.img_h*6))))
        img = transforms.functional.to_tensor(img)
        img = (img - 0.5) / 0.5
        seq = self.encode_text(txt)
        return img, torch.tensor(seq, dtype=torch.long), txt


def collate_fn(batch):
    imgs, seqs, texts = zip(*batch)
    # pad images to same width
    max_w = max(i.shape[-1] for i in imgs)
    imgs_p = torch.zeros(len(imgs), 1, imgs[0].shape[-2], max_w)
    for i,img in enumerate(imgs):
        imgs_p[i,:, :, :img.shape[-1]] = img
    # concatenate seqs
    seq_lens = [len(s) for s in seqs]
    targets = torch.cat(seqs)
    return imgs_p, targets, torch.tensor(seq_lens, dtype=torch.long), texts


class CRNN(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_chars+1)  # +1 for blank

    def forward(self, x):
        # x: B, C, H, W
        b, c, h, w = x.size()
        out = self.cnn(x)  # B, C2, H2, W2
        b, c2, h2, w2 = out.size()
        # collapse height
        out = out.permute(0,3,1,2).contiguous()  # B, W2, C2, H2
        out = out.view(b, w2, c2*h2)  # B, T, feat
        out, _ = self.rnn(out)
        out = self.fc(out)  # B, T, classes
        out = out.log_softmax(2)
        return out


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = OCRDataset(args.base_dir, args.train_list)
    val_ds = OCRDataset(args.base_dir, args.val_list, charset=train_ds.charset)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = CRNN(len(train_ds.charset)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer = 1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for imgs, targets, seq_lens, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)  # B, T, C
            out = out.permute(1,0,2)  # T, B, C
            input_len = torch.full((out.size(1),), out.size(0), dtype=torch.long)
            loss = ctc(out, targets.to(device), input_len, seq_lens.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} train loss {avg:.4f}")
        # validate
        cer = validate(model, val_loader, device, train_ds.charset)
        print(f"Epoch {epoch} val CER {cer:.4f}")
        if cer < best_cer:
            best_cer = cer
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'charset': train_ds.charset}, os.path.join(args.save_dir, 'best.pt'))
    print("Training complete")


def decode_preds(log_probs, charset):
    # log_probs: B, T, C
    argmax = log_probs.argmax(2)  # B, T
    results = []
    for seq in argmax:
        prev = -1
        out = []
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                out.append(charset[idx-1])
            prev = idx
        results.append(''.join(out))
    return results


def validate(model, loader, device, charset):
    model.eval()
    tot_ed = 0
    tot_chars = 0
    with torch.no_grad():
        for imgs, targets, seq_lens, texts in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            decoded = decode_preds(out.cpu(), charset)
            for pred, gt in zip(decoded, texts):
                tot_ed += editdistance.eval(pred, gt)
                tot_chars += len(gt)
    return tot_ed / max(1, tot_chars)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='.')
    p.add_argument('--train-list', required=True)
    p.add_argument('--val-list', required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-dir', default='checkpoints/crnn_ar')
    args = p.parse_args()
    train(args)
