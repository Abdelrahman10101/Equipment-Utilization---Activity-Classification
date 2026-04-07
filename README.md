# 🦅 Eagle Vision — Equipment Utilization & Activity Intelligence

A real-time, microservices-based pipeline for **construction equipment monitoring** that combines
**instance segmentation, motion analysis, and temporal reasoning** to deliver highly accurate
utilization tracking and activity classification.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Kafka](https://img.shields.io/badge/Messaging-Apache%20Kafka-black)
![CV](https://img.shields.io/badge/CV-YOLOv11--Seg-purple)
![DB](https://img.shields.io/badge/Database-TimescaleDB-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🔥 What’s New (Key Upgrades)

### 🎯 Instance Segmentation (Pixel-Level Precision)

* Switched from **YOLO detection → YOLO segmentation (`yolo11m-seg.pt`)**
* Enables **pixel-accurate masks** instead of bounding boxes
* No additional training required (Ultralytics auto-downloads weights)

### 🎯 Mask-Aware Motion Analysis

* Optical flow computed **only inside object masks**
* Eliminates background noise (sky, ground, irrelevant motion)
* Region-based analysis respects **true object shape**

### 🎯 Temporal Intelligence

* Tracks motion history over ~30 frames
* Detects real behavioral patterns instead of frame-by-frame noise

Patterns detected:

* `down_then_up` → DIGGING
* `sustained_lateral` → SWINGING
* `up_then_release` → DUMPING
* `sustained_stillness` → WAITING
* `oscillating` → LOADING (vibration)

### 🎯 Two-Layer Activity Classification

1. **Primary Layer:** Temporal pattern detection (confidence ≥ 0.65)
2. **Fallback Layer:** Instant rule-based features

Both layers feed into smoothing and debouncing for stable predictions.

---

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌──────────┐     ┌────────────────────────┐     ┌──────────┐
│  Frame Producer  │────▶│  Kafka   │────▶│   CV Processor          │────▶│  Kafka   │
│  (Video → Kafka) │     │          │     │  Segmentation + Motion  │     │          │
└─────────────────┘     │ raw-     │     │  + Temporal Classifier  │     │equipment-│
                        │ frames   │     └────────────────────────┘     │events    │
                        └──────────┘                                   │annotated │
                                                                       │frames    │
                                                                       └────┬─────┘
                                                                            │
                                                          ┌─────────────────┼────────────────┐
                                                          │                 │                │
                                                   ┌──────▼──────┐  ┌──────▼────────┐       │
                                                   │   DB Sink   │  │  Dashboard     │       │
                                                   │ TimescaleDB │  │  (Streamlit)   │       │
                                                   └─────────────┘  └───────────────┘
```

---

## 🧩 Microservices

| Service        | Description                                                |
| -------------- | ---------------------------------------------------------- |
| Frame Producer | Extracts frames from videos → Kafka                        |
| CV Processor   | Segmentation + mask-aware motion + temporal classification |
| DB Sink        | Stores events in TimescaleDB                               |
| Dashboard      | Real-time monitoring UI                                    |

---

## 🚀 Quick Start

### Prerequisites

* Docker & Docker Compose (v2+)
* Sample `.mp4` videos in `sample_videos/`
* Optional: NVIDIA GPU

### Run

```bash
git clone <repo-url>
cd eagle-vision

cp /path/to/videos/*.mp4 sample_videos/

docker-compose up --build
```

Open dashboard:
[http://localhost:8501](http://localhost:8501)

---

## 🧠 Computer Vision Pipeline

### 1. Detection + Segmentation

* YOLOv11m-seg
* Instance-level masks
* ByteTrack for tracking

### 2. Mask-Aware Optical Flow

* Flow computed only within segmentation masks
* Removes background noise
* Improves motion accuracy

### 3. Region-Based Motion

Each object is split into:

* Upper → Arm / boom
* Middle → Cabin / swing
* Lower → Tracks / wheels

All regions are mask-filtered.

### 4. Temporal Pattern Detection

* Uses motion history (~30 frames)
* Recognizes behavior instead of isolated motion

### 5. Two-Layer Classification

* Temporal patterns (primary)
* Rule-based fallback
* Smoothed and debounced output

---

## 📦 Kafka Topics

| Topic            | Description                 |
| ---------------- | --------------------------- |
| raw-frames       | Input frames                |
| equipment-events | Activity + utilization data |
| annotated-frames | Visual output               |

---

## 📊 Event Payload

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "00:00:15.000",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "mask_temporal"
  }
}
```

---

## ⚙️ Configuration

| Variable              | Description         |
| --------------------- | ------------------- |
| TARGET_FPS            | Processing FPS      |
| CONFIDENCE_THRESHOLD  | Detection threshold |
| MOTION_THRESHOLD      | Flow sensitivity    |
| SLIDING_WINDOW_SIZE   | Temporal smoothing  |
| STATE_DEBOUNCE_FRAMES | Stability control   |

---

## 🧠 Why This Matters

Traditional systems:

* Bounding boxes
* Instant decisions
* High noise

Eagle Vision:

* Pixel-level understanding
* Temporal reasoning
* Behavior-aware classification

Result: **More accurate, stable, and realistic activity detection**

---

## 📁 Project Structure

```
eagle-vision/
├── docker-compose.yml
├── .env
├── db/
├── sample_videos/
├── services/
│   ├── frame_producer/
│   ├── cv_processor/
│   ├── db_sink/
│   └── dashboard/
└── models/
```

---

## 📝 License

Technical assessment / prototype project.
