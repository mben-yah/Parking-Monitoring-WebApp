# -*- coding: utf-8 -*-
"""
elpd_train.py
─────────────
ELPD Commercial YOLOv8 Training with:
  - Per-epoch file logging  (logs/elpd_training.log)
  - ETA estimation
  - Resume from last checkpoint if available
"""
import time, sys, io, json
from pathlib import Path
from datetime import datetime, timedelta

# UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Users\Mohamed Walid\Desktop\Internship\Code")
YAML     = BASE / "elpd_commercial.yaml"
PROJECT  = BASE / "runs" / "detect" / "runs" / "detect"
RUN_NAME = "elpd_commercial_train1"
RUN_DIR  = PROJECT / RUN_NAME
LOG_DIR  = BASE / "logs"
LOG_FILE = LOG_DIR / "elpd_training.log"
EPOCHS   = 40
BATCH    = 16
IMGSZ    = 640

# Fine-tune from english_train33 unless a resume checkpoint exists
RESUME_PT = RUN_DIR / "weights" / "last.pt"
BASE_PT   = BASE / "runs" / "detect" / "runs" / "detect" / "english_train33" / "weights" / "best.pt"

LOG_DIR.mkdir(exist_ok=True)

def log(msg: str):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}]  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Determine starting weights ─────────────────────────────────────────────────
if RESUME_PT.exists():
    start_weights = str(RESUME_PT)
    log(f"Resuming from last checkpoint: {RESUME_PT}")
else:
    start_weights = str(BASE_PT)
    log(f"Starting fresh from: {BASE_PT}")

log(f"Dataset YAML  : {YAML}")
log(f"Output run    : {RUN_DIR}")
log(f"Epochs        : {EPOCHS}  |  Batch: {BATCH}  |  Imgsz: {IMGSZ}")
log("=" * 70)

# ── Custom callback for per-epoch logging + ETA ────────────────────────────────
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import default_callbacks

epoch_times: list[float] = []
train_start = time.time()

def on_train_epoch_end(trainer):
    epoch      = trainer.epoch + 1          # 1-indexed
    total_ep   = trainer.epochs
    t_now      = time.time()
    elapsed    = t_now - train_start

    # Track per-epoch wall time
    if len(epoch_times) < epoch:
        epoch_times.append(elapsed / epoch)

    avg_ep_s   = elapsed / epoch
    remaining  = avg_ep_s * (total_ep - epoch)
    eta        = datetime.now() + timedelta(seconds=remaining)

    # Pull latest metrics
    metrics = trainer.metrics or {}
    box_loss = getattr(trainer.loss_items, 'tolist', lambda: [None,None,None])()
    mAP50    = metrics.get("metrics/mAP50(B)",    "—")
    mAP5095  = metrics.get("metrics/mAP50-95(B)", "—")
    prec     = metrics.get("metrics/precision(B)", "—")
    rec      = metrics.get("metrics/recall(B)",    "—")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    log(
        f"Epoch {epoch:>2}/{total_ep}  |  "
        f"mAP50={fmt(mAP50)}  mAP50-95={fmt(mAP5095)}  "
        f"P={fmt(prec)}  R={fmt(rec)}  |  "
        f"Elapsed={str(timedelta(seconds=int(elapsed)))}  "
        f"ETA={eta.strftime('%H:%M:%S')}  (~{str(timedelta(seconds=int(remaining)))} left)"
    )

    # Write a JSON summary for easy machine reading
    summary = {
        "epoch": epoch, "total": total_ep,
        "mAP50": round(float(mAP50), 4) if isinstance(mAP50, float) else None,
        "mAP50_95": round(float(mAP5095), 4) if isinstance(mAP5095, float) else None,
        "elapsed_s": round(elapsed, 1),
        "eta": eta.isoformat(),
    }
    (LOG_DIR / "elpd_epoch_latest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

# ── Load model and attach callback ────────────────────────────────────────────
model = YOLO(start_weights)
model.add_callback("on_train_epoch_end", on_train_epoch_end)

log("Training started ...")
log("=" * 70)

results = model.train(
    data     = str(YAML),
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    project  = str(PROJECT),
    name     = RUN_NAME,
    exist_ok = True,
    verbose  = True,
    patience = 15,
)

# ── Final summary ──────────────────────────────────────────────────────────────
total_time = time.time() - train_start
best_pt    = RUN_DIR / "weights" / "best.pt"

log("=" * 70)
log("TRAINING COMPLETE!")
log(f"Total time    : {str(timedelta(seconds=int(total_time)))}")
log(f"Best weights  : {best_pt}")

final = {
    "status":      "complete",
    "total_time_s": round(total_time, 1),
    "best_weights": str(best_pt),
    "finished_at":  datetime.now().isoformat(),
}
(LOG_DIR / "elpd_training_final.json").write_text(
    json.dumps(final, indent=2), encoding="utf-8"
)
log("Log saved to: " + str(LOG_FILE))
