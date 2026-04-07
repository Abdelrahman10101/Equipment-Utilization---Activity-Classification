# 🦅 Eagle Vision — Equipment Utilization & Activity Classification

A real-time, microservices-based pipeline that processes construction equipment video clips,
tracks utilization states (ACTIVE/INACTIVE), classifies work activities, and streams results
through Apache Kafka to a live Streamlit dashboard with TimescaleDB persistence.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Kafka](https://img.shields.io/badge/Messaging-Apache%20Kafka-black)
![CV](https://img.shields.io/badge/CV-YOLOv8-purple)
![DB](https://img.shields.io/badge/Database-TimescaleDB-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌──────────┐     ┌──────────────────┐     ┌──────────┐
│  Frame Producer  │────▶│          │────▶│   CV Processor    │────▶│          │
│  (Video → Kafka) │     │  Apache  │     │  YOLOv8 + Motion  │     │  Apache  │
└─────────────────┘     │  Kafka   │     │  + Activity Class  │     │  Kafka   │
                        │          │     └──────────────────┘     │          │
                        │ raw-     │                               │equipment-│
                        │ frames   │                               │events    │
                        └──────────┘                               │annotated-│
                                                                   │frames    │
                                                                   └────┬─────┘
                                                                        │
                                                          ┌─────────────┼──────────────┐
                                                          │             │              │
                                                   ┌──────▼──────┐  ┌──▼───────────┐   │
                                                   │   DB Sink   │  │  Dashboard   │   │
                                                   │  → Timescale│  │  (Streamlit) │   │
                                                   │    DB       │  │              │   │
                                                   └─────────────┘  └──────────────┘   │
```

### Microservices

| Service | Description | Port |
|---------|-------------|------|
| **Frame Producer** | Reads video files, extracts frames, publishes to Kafka | - |
| **CV Processor** | YOLOv8 detection + optical flow motion analysis + activity classification | - |
| **DB Sink** | Consumes events and batch-inserts into TimescaleDB | - |
| **Dashboard** | Streamlit real-time monitoring UI | 8501 |

### Infrastructure

| Service | Description | Port |
|---------|-------------|------|
| **Kafka** | Message broker (Confluent Platform) | 9092 / 29092 |
| **Zookeeper** | Kafka coordination | 2181 |
| **TimescaleDB** | Time-series database (PostgreSQL) | 5432 |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose** (v2+)
- **Sample Videos**: Place `.mp4` files in the `sample_videos/` folder
- **GPU** (optional): NVIDIA GPU for faster inference

### 1. Clone and add videos

```bash
git clone <repo-url>
cd eagle-vision

# Place your construction equipment videos here:
cp /path/to/your/videos/*.mp4 sample_videos/
```

### 2. Launch the pipeline

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 3. Open the dashboard

Navigate to **http://localhost:8501** in your browser.

### 4. Stop the pipeline

```bash
docker-compose down

# To also remove volumes (database data):
docker-compose down -v
```

---

## 🖥️ GPU Support

To enable NVIDIA GPU acceleration for the CV Processor, uncomment the GPU section
in `docker-compose.yml`:

```yaml
cv-processor:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

---

## 📦 Kafka Topics

| Topic | Producer | Consumer | Payload |
|-------|----------|----------|---------|
| `raw-frames` | Frame Producer | CV Processor | Base64 JPEG + metadata |
| `equipment-events` | CV Processor | DB Sink, Dashboard | JSON event payload |
| `annotated-frames` | CV Processor | Dashboard | Base64 annotated JPEG + states |

### Equipment Event Payload Format

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "00:00:15.000",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "utilization_percent": 83.3
  }
}
```

---

## 🧠 Computer Vision Pipeline

### Equipment Detection
- **YOLOv8m** (medium) for real-time object detection
- Supports COCO-pretrained (trucks) and custom-trained models (excavators, loaders)
- Built-in **ByteTrack** for multi-object tracking with persistent IDs

### Articulated Motion Analysis
Key challenge: Detecting ACTIVE state when only part of the machine moves.

**Solution: Region-Based Optical Flow**

```
┌─────────────────────┐
│   Upper Region      │  ← Arm/Boom movement
│   (top 45%)         │
├─────────────────────┤
│   Middle Region     │  ← Cab/Swing rotation
│   (middle 20%)      │
├─────────────────────┤
│   Lower Region      │  ← Track/Wheel movement
│   (bottom 35%)      │
└─────────────────────┘
+ Left/Right split for swing detection
```

1. Dense optical flow (Farnebäck) between consecutive frames
2. Extract motion magnitude per sub-region within each bounding box
3. If ANY sub-region exceeds threshold → **ACTIVE**
4. Motion source labeling: `arm_only`, `tracks_only`, `full_body`, `swing`

### Activity Classification

| Activity | Detection Pattern |
|----------|-------------------|
| **DIGGING** | Upper region active (downward), tracks stationary |
| **SWINGING/LOADING** | High lateral motion + upper/middle active |
| **DUMPING** | Upper region active (upward direction) |
| **WAITING** | All regions below threshold |

Temporal smoothing via sliding window prevents activity flickering.

---

## ⚙️ Configuration

All configuration is via environment variables in `.env` or `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_FPS` | 10 | Processing frame rate |
| `CONFIDENCE_THRESHOLD` | 0.4 | YOLO detection confidence |
| `MOTION_THRESHOLD` | 2.0 | General motion threshold |
| `ARM_MOTION_THRESHOLD` | 1.5 | Arm region threshold (more sensitive) |
| `TRACK_MOTION_THRESHOLD` | 2.5 | Track region threshold (less sensitive) |
| `STATE_DEBOUNCE_FRAMES` | 5 | Frames before state transition |
| `SLIDING_WINDOW_SIZE` | 10 | Activity classification window |

---

## 🗄️ Database Schema

TimescaleDB hypertable for time-series equipment events:

```sql
equipment_events (
    time, frame_id, equipment_id, equipment_class,
    current_state, current_activity, motion_source,
    bbox_x, bbox_y, bbox_w, bbox_h, confidence,
    total_tracked_seconds, total_active_seconds,
    total_idle_seconds, utilization_percent
)
```

Pre-built views:
- `equipment_latest` — Latest state per equipment
- `utilization_summary` — Aggregated utilization per equipment

---

## 📁 Project Structure

```
eagle-vision/
├── docker-compose.yml          # Full orchestration
├── .env                        # Configuration
├── db/
│   └── init.sql                # TimescaleDB schema
├── sample_videos/              # Input video files
├── services/
│   ├── frame_producer/         # Video → Kafka frames
│   ├── cv_processor/           # Detection + Motion + Classification
│   ├── db_sink/                # Kafka → TimescaleDB
│   └── dashboard/              # Streamlit UI
└── models/                     # Model weights (gitignored)
```

---

## 📝 License

This project is a technical assessment prototype.
