# Technical Write-Up: Design Decisions & Trade-offs

## 1. System Architecture

### Why Microservices over Monolith?

The decision to split the pipeline into four microservices (Frame Producer, CV Processor,
DB Sink, and Dashboard) was driven by:

1. **Independent Scaling**: The CV Processor is the bottleneck. In production, we could
   run multiple CV Processor instances within the same Kafka consumer group, each processing
   a partition of frames in parallel.

2. **Fault Isolation**: If the dashboard crashes, the CV processing continues. If the DB
   sink falls behind, events are buffered in Kafka. No single point of failure blocks the pipeline.

3. **Technology Independence**: Each service can use different base images. The CV Processor
   needs PyTorch/CUDA, while the DB Sink only needs psycopg2.

### Why Kafka?

Apache Kafka serves as the central nervous system because:
- **Decoupling**: Producers and consumers operate independently
- **Buffering**: Handles backpressure when CV processing is slower than frame ingestion
- **Replay**: Consumers can re-read from any offset for debugging or reprocessing
- **Multiple Consumers**: Both DB Sink and Dashboard consume from the same topic independently

### Trade-off: Frame Serialization

Sending full frames through Kafka as base64 JPEG is not ideal for production (high bandwidth).
Alternatives considered:
- **Shared filesystem**: Lower latency but introduces coupling
- **Redis Streams**: Better for frame data but adds another infrastructure component
- **Object store (S3/MinIO)**: Best for production, but over-engineered for a prototype

**Decision**: Base64 JPEG via Kafka is acceptable for prototype scope with LZ4 compression enabled.

---

## 2. Articulated Equipment Challenge

### Problem Statement

An excavator may be classified as INACTIVE if we only measure overall bounding box motion,
because the tracks are stationary while the arm is actively digging. This is a false negative
that would significantly skew utilization calculations.

### Solution: Region-Based Optical Flow Analysis

**Step 1: Dense Optical Flow (Farnebäck Method)**

We compute pixel-level motion vectors between consecutive grayscale frames using
OpenCV's `calcOpticalFlowFarneback`. This gives us a dense flow field F(x,y) = (dx, dy)
representing how each pixel moved between frames.

Why Farnebäck over alternatives:
- **Sparse KLT Tracking**: Only tracks corner features — misses arm motion on smooth surfaces
- **Background Subtraction**: Treats the entire machine as foreground, can't distinguish parts
- **Deep optical flow (RAFT)**: Too slow for real-time processing on CPU

**Step 2: Sub-Region Decomposition**

For each detected equipment bounding box, we divide it into regions:

```
┌─────────────────────┐
│   Upper (top 45%)   │  ← Arm/boom: uses lower threshold (1.5)
├─────────────────────┤     for higher sensitivity to arm motion
│   Middle (20%)      │  ← Cab/body: uses standard threshold (2.0)
├─────────────────────┤
│   Lower (bottom 35%)│  ← Tracks/wheels: uses higher threshold (2.5)
└─────────────────────┘     to filter out vibration/noise
```

Additionally, we split left/right for swing detection (lateral asymmetry).

**Step 3: Motion Source Classification**

- If upper region exceeds threshold but lower does not → `arm_only`
- If lower exceeds but upper does not → `tracks_only`
- If significant lateral asymmetry → `swing`
- If both upper and lower exceed → `full_body`

**Step 4: Temporal Smoothing**

A sliding window (10 frames) with majority voting prevents flicker between states.
The active threshold is set at 40% — if 4+ of the last 10 frames show motion, we
classify as ACTIVE.

### Trade-offs and Limitations

1. **Region boundaries are fixed ratios**: A custom model with instance segmentation
   (YOLOv8-seg) would provide pixel-precise equipment masks for more accurate sub-region
   analysis. We chose fixed ratios for prototype simplicity.

2. **Farneback is resolution-dependent**: Motion thresholds need tuning per resolution.
   Our defaults (1.5/2.0/2.5) are calibrated for 1280x720 input.

3. **Camera motion**: If the camera pans, ALL equipment will appear to have motion.
   In production, we'd add camera motion compensation using homography estimation.

---

## 3. Activity Classification

### Approach: Rule-Based Heuristics with Temporal Patterns

We chose rule-based classification over a trained ML classifier because:
- **No training data**: We don't have labeled activity datasets
- **Interpretability**: Rules are debuggable and tunable
- **Speed**: No additional model inference cost

### Classification Logic

**For Excavators:**
- **DIGGING**: Upper region active + downward motion direction (positive dy) + tracks stationary
- **SWINGING/LOADING**: High lateral asymmetry + middle/upper region active
- **DUMPING**: Upper region active + upward motion direction (negative dy)
- **WAITING**: All regions below threshold

**For Dump Trucks:**
- **DUMPING**: Upper region (bed) active + lower region stationary
- **TRAVELING**: Full body motion
- **SWINGING_LOADING**: Vibration pattern detected (high variance in motion magnitude)

### Temporal Smoothing

Activities are smoothed using a sliding window of the last 5 classifications with
majority voting. This prevents rapid flickering between activities during transitions.

### Known Limitations

1. **Digging vs. Swinging confusion**: When the arm moves laterally during digging,
   it may be classified as swinging. A trained temporal model (LSTM) would handle this better.

2. **Activity transitions**: The smoothing window introduces a 0.5-1 second delay
   in detecting activity changes.

---

## 4. State Management & Time Tracking

### Debounced State Transitions

To avoid false transitions from momentary motion/stillness, we require a state to
persist for 5 consecutive frames before transitioning. This means:
- At 10 FPS effective rate, state transitions have ~0.5 second latency
- Brief pauses (<0.5s) during active work won't count as idle time

### Time Calculation

```
utilization_percent = (total_active_seconds / total_tracked_seconds) × 100
```

Time accumulates based on the CURRENT state at each frame:
- While in ACTIVE state, delta_t is added to `total_active_seconds`
- While in INACTIVE state, delta_t is added to `total_idle_seconds`

---

## 5. Detection Model Considerations

### COCO-Pretrained YOLOv8

The baseline COCO model detects:
- `truck` (class 7) → mapped to `dump_truck`
- `bus` (class 5) → mapped to `dump_truck` (visually similar)

**Limitation**: COCO has no `excavator` class.

### Solutions for Excavator Detection

1. **Custom Roboflow model**: Swap the model weights with a construction-equipment-specific
   model from Roboflow Universe. The code automatically detects custom class names.

2. **Fine-tuning**: Fine-tune YOLOv8 on a construction equipment dataset.
   Many annotated datasets exist on Roboflow and Kaggle.

The code is designed to handle both — it dynamically maps model class names
to equipment types, supporting seamless model swapping.

---

## 6. Database Design

### Why TimescaleDB?

- **Time-series optimized**: Hypertables auto-partition by time for fast range queries
- **PostgreSQL compatible**: Full SQL query support, familiar tooling
- **Compression**: Native compression for older data
- **Continuous aggregates**: Could add real-time materialized views for dashboard queries

### Batch Inserts

The DB Sink batches events (50 per batch or every 2 seconds) to reduce write overhead.
Using `execute_values` from psycopg2 for efficient multi-row inserts.

---

## 7. Future Improvements

1. **Instance Segmentation**: Use YOLOv8-seg for pixel-precise equipment masks
2. **Keypoint detection**: Track excavator arm joints for better articulation analysis
3. **LSTM activity classifier**: Train on temporal motion feature sequences
4. **Camera motion compensation**: Homography-based stabilization
5. **Multi-camera support**: Track equipment across multiple camera views
6. **RTSP stream input**: Support live camera feeds instead of just video files
7. **Alert system**: Notify when utilization drops below threshold
8. **Model versioning**: MLflow integration for model management
