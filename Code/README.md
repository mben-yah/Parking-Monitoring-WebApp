# PlateVision — Automatic License Plate Recognition System

> **Intelligent parking surveillance platform** combining YOLOv8 plate detection, multi-engine OCR (PaddleOCR · EasyOCR), real-time livestream analysis, role-based authentication, MongoDB persistence, Docker containerization, and an analytics dashboard.

---

## Screenshots

### Homepage — Image Recognition
![Homepage — License Plate Recognition](screenshots/homepage.png)

### Live Stream — IP Webcam
![Livestream — IP Webcam mode with real-time detection feed](screenshots/livestream.png)

### Analytics & Sensor Health Dashboard
![Dashboard — Peak hours, sensor health, detection metrics](screenshots/dashboard.png)

### Supervisor — Client & Vehicle Authorization Manager
![Supervisor — Whitelist/Blacklist management and alert panel](screenshots/supervisor.png)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Datasets & Augmentation](#4-datasets--augmentation)
5. [Models & Training](#5-models--training)
6. [Core Modules](#6-core-modules)
7. [Web Application & Authentication](#7-web-application--authentication)
8. [Database Schema](#8-database-schema)
9. [OCR Filtering & Consensus Voting](#9-ocr-filtering--consensus-voting)
10. [Docker & Containerized Deployment](#10-docker--containerized-deployment)
11. [Installation & Local Setup](#11-installation--local-setup)
12. [API Reference](#12-api-reference)
13. [Configuration](#13-configuration)
14. [Results & Metrics](#14-results--metrics)
15. [Academic Reports](#15-academic-reports)

---

## 1. Project Overview

PlateVision is an end-to-end Automatic License Plate Recognition (ALPR / ANPR) system engineered for smart parking lot surveillance and access control.

| Feature | Description |
|---|---|
| **Image Mode** | Drag-and-drop single/batch images → instant plate detection + OCR |
| **Video Mode** | Upload video → frame-by-frame analysis with timeline, skip-slider & DB save toggle |
| **Livestream Mode** | Connect IP webcams or RTSP feeds → real-time SSE stream & intrusion alerts |
| **Dataset Selector** | Toggle between **Commercial ELPD (mAP 0.987)** and **Non-Commercial** models |
| **Authentication** | Role-based access control (`ADMIN` vs `OPERATOR`) with password hashing |
| **Supervisor Panel** | Vehicle Whitelist / Blacklist management with instant intrusion alert feeds |
| **Dashboard** | Peak hours traffic analysis, sensor health status (Uptime/Downtime), and plate trends |
| **DB Admin** | Paginated detection browser with search, model filtering, and single/batch deletion |

The system supports both **English/International** and **Moroccan/Arabic** license plates with dedicated detection models and specialized OCR normalization pipelines.

---

## 2. Architecture

```
+-----------------------------------------------------------------------+
|                             Browser (UI)                              |
|   login.html | index.html | video.html | livestream.html | db.html    |
|              dashboard.html | supervisor.html                         |
+-----------------------------------+-----------------------------------+
                                    | HTTP / SSE (Server-Sent Events)
+-----------------------------------v-----------------------------------+
|                      Flask Server  (app.py)                           |
|  * Auth Guard (@app.before_request)                                   |
|  * /predict  /predict_arabic  /predict_video  /stream_feed            |
|  * /api/auth/*  /api/parking/*  /api/dashboard/*                      |
+-------+---------------------------+-------------------+---------------+
        |                           |                   |
+-------v-------+        +----------v--------+  +-------v-----------+
| YOLOv8        |        |  OCR Engines      |  |  MongoDB /        |
| Detector      |        |  * EasyOCR (EN)   |  |  MontyDB fallback |
| (best.pt)     |        |  * PaddleOCR (AR) |  |  (mongodb_client) |
+-------+-------+        +----------+--------+  +-------------------+
        |                           |
        +-------------+-------------+
                      |
          +-----------v-----------+
          |   ocr_consensus.py    |
          |  * Digit filter       |
          |  * Character-level    |
          |    majority voting    |
          +-----------------------+
```

---

## 3. Directory Structure

```
Code/
├── app.py                          # Main Flask web server (Auth, API routes, SSE)
├── mongodb_client.py               # Unified Mongo/MontyDB client (Detections, Users, Alerts)
├── recognition_pipeline.py         # English & Commercial plate recognition pipeline
├── arabic_ocr_pipeline.py          # Arabic/Moroccan plate OCR pipeline
├── stream_manager.py           # Real-time IP camera / RTSP stream processing
├── ocr_consensus.py            # Digit filtering & character-level voting
├── alert_engine.py             # Blacklist match & intrusion alert engine
|
├── elpd_prep_v2.py                 # ELPD Commercial v2 augmentation & training script
├── elpd_commercial_v2.yaml         # YOLOv8 dataset configuration for ELPD v2
|
├── frontend/
│   ├── login.html                  # Dark glassmorphism authentication page (Sign In / Register)
│   ├── index.html                  # Image recognition homepage & dataset selector
│   ├── video.html                  # Video upload & timeline processing
│   ├── livestream.html             # Real-time IP webcam feed & alert overlay
│   ├── dashboard.html              # Peak hours & sensor health dashboard
│   ├── db.html                     # Detection database manager
│   └── supervisor.html             # Whitelist / Blacklist & alert supervisor
|
├── reports/
│   ├── chapitre3_conception_et_modelisation.md   # Chapitre 3 (French academic report)
│   └── rapport_validation_et_resultats.md        # Chapitre 4 (French academic report)
|
├── Dockerfile                      # Optimized Python 3.10 + PyTorch CPU image
├── docker-compose.yml              # Multi-container orchestration (App + MongoDB)
├── requirements.txt                # Python dependencies
├── .dockerignore                   # Build exclusion rules
|
├── logs/                           # Live training & API logs
├── platevision_db/                 # MontyDB persistent fallback storage
└── runs/detect/runs/detect/        # Trained YOLOv8 weight checkpoints
    ├── elpd_commercial_train2/     <- Best Commercial Model (mAP50 = 0.9870)
    └── english_train33/            <- Base English Model
```

---

## 4. Datasets & Augmentation

### 4.1 ELPD Commercial v2 Dataset (COCO Conversion)

| Property | Value |
|---|---|
| Source Directory | `archive\COCO` |
| Annotation Format | **COCO JSON** (`instances_Train.json`) |
| Total Images in JSON | 2,329 |
| Valid Annotated Images | **2,287** |
| Train / Val Split | **85% / 15%** (1,944 train / 343 val) |
| Augmentation Strategy | **Crop (70–90%)**, **Scale (75–125%)**, **Translation ($\pm 10\%$)** — *No colour/hue shifts* |
| **Final Augmented Train Set** | **7,735 images** |
| Validation Set | **343 images** |

### 4.2 English (AOLP Subset_LE) & Moroccan Datasets

- **AOLP Subset_LE**: Standard benchmark for English plate detection.
- **Moroccan License Plates**: Syntax format `Sequence | Arabic Letter | Region Code` (e.g., `12345 | أ | 06`).

---

## 5. Models & Training

### 5.1 Detection Models

All detection models are built on **YOLOv8n (Nano)** fine-tuned specifically for single-class license plate localization.

| Model / Run | Base Weights | Train Images | Epochs | mAP50 | mAP50-95 | Status |
|---|---|---|---|---|---|---|
| **`elpd_commercial_train2`** | `english_train33` | 7,735 | 40 (Checkpoint Epoch 8) | **0.9870 (98.7%)** | **0.8537** | **Active Commercial Model** |
| `english_train33` | `yolov8n.pt` | ~1,944 | 33 | 0.9420 | 0.7950 | Base English Model |

---

## 6. Core Modules

- **`app.py`**: Flask web application. Enforces `@app.before_request` authentication guard and manages multi-model dispatching based on `dataset_type` (`commercial`, `non_commercial`, `auto`).
- **`mongodb_client.py`**: Handles MongoDB and MontyDB fallback connection. Manages collections `detections`, `users`, `whitelist`, `blacklist`, `alerts`, and `devices`.
- **`ocr_consensus.py`**: Performs two-stage filtering:
  1. **Digit Filter (`has_digit`)**: Discards predictions containing zero digits.
  2. **Character-Level Majority Voting**: Position-by-position voting across multiple video frames.

---

## 7. Web Application & Authentication

### 🔑 Authentication & Role-Based Access Control

The web application enforces strict session-based authentication:
- **Unauthenticated Users**: Automatically redirected to `/login`.
- **`ADMIN` Role**: Full access to Whitelist/Blacklist CRUD, alert resolution, DB Admin (`/db`), and server diagnostics.
- **`OPERATOR` Role**: Access to live feeds, video analysis, and traffic monitoring.

#### Demo Credentials:
- 🛡️ **Administrator**: `admin` / `admin123`
- 👁️ **Operator**: `operator` / `operator123`

---

## 8. Database Schema

### `users`
```json
{
  "_id": "ObjectId",
  "username": "admin",
  "password_hash": "pbkdf2:sha256:...",
  "role": "ADMIN",
  "created_at": "ISODate",
  "last_login": "ISODate"
}
```

### `detections`
```json
{
  "_id": "ObjectId",
  "timestamp": "ISODate",
  "plate_text": "12345|A|06",
  "confidence": 0.987,
  "bbox": [x1, y1, x2, y2],
  "model_used": "elpd_commercial_train2",
  "source": "image",
  "source_url": "test.jpg"
}
```

---

## 9. OCR Filtering & Consensus Voting

```
[YOLOv8 Detection] -> Crop Plate Region -> Multi-OCR (EasyOCR / PaddleOCR)
                                                 |
                                     validate_and_filter_plate() (Digit Filter)
                                                 |
                                     Character-Level Voting (Across Frames)
                                                 |
                                     Final Clean Plate String -> DB & Alert Engine
```

---

## 10. Docker & Containerized Deployment

PlateVision is fully containerized for seamless deployment on Linux or Windows servers via Docker:

### Build and Run with Docker Compose:

```bash
# Build and launch both PlateVision Web App & MongoDB
docker-compose up --build -d
```

### Direct Docker Run:

```bash
docker build -t platevision:latest .
docker run -d -p 5000:5000 --name platevision_app platevision:latest
```

---

## 11. Installation & Local Setup

### Local Anaconda Setup:

```bash
# 1. Create and activate environment
conda create -n yolo_env python=3.10 -y
conda activate yolo_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch PlateVision
python app.py
```

Access the app in your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 12. API Reference

| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/login` | GET | - | Serves authentication UI |
| `/api/auth/login` | POST | `{username, password}` | Authenticates user & sets session |
| `/api/auth/me` | GET | - | Returns active session user |
| `/predict` | POST | `file`, `dataset_type` | Image plate detection & OCR |
| `/predict_video` | POST | `file`, `dataset_type`, `frame_skip` | Video SSE stream detection |
| `/api/parking/whitelist` | GET/POST | `{plate_text, owner}` | Manage authorized vehicles |
| `/api/parking/blacklist` | GET/POST | `{plate_text, reason}` | Manage banned vehicles |

---

## 13. Results & Metrics

### ELPD Commercial v2 Model (`elpd_commercial_train2`)

- **mAP50**: **0.9870 (98.70%)** 🔥
- **mAP50-95**: **0.8537**
- **Precision ($P$)**: **0.9399**
- **Recall ($R$)**: **0.9669**
- **Inference Speed**: ~35 ms / image (CPU)

---

## 14. Academic Reports

Complete French academic report chapters generated for thesis documentation are stored in the [`reports/`](file:///C:/Users/Mohamed%20Walid/Desktop/Internship/Code/reports) folder:

1. **Chapitre 3** : [`reports/chapitre3_conception_et_modelisation.md`](file:///C:/Users/Mohamed%20Walid/Desktop/Internship/Code/reports/chapitre3_conception_et_modelisation.md) — *Approches algorithmiques, modèles d'IA, augmentation de données et métriques d'évaluation.*
2. **Chapitre 4** : [`reports/rapport_validation_et_resultats.md`](file:///C:/Users/Mohamed%20Walid/Desktop/Internship/Code/reports/rapport_validation_et_resultats.md) — *Validation expérimentale, jeux de données, protocoles et discussion des résultats.*

---

## License

Copyright © 2026 Mohamed Walid Ben Yahia

All Rights Reserved.

This repository is provided for viewing purposes only. No permission is granted to use, copy, modify, distribute, or create derivative works without prior written permission.
