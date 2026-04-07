#!/usr/bin/env python3
"""
===============================================================================
 🦅 Eagle Vision — Kaggle GPU Pipeline
 Real-Time Equipment Utilization & Activity Classification
===============================================================================

 This is a SELF-CONTAINED version of the full Eagle Vision pipeline,
 designed to run on Kaggle with GPU/CPU/RAM resources.

 No Kafka, No Docker, No PostgreSQL — just direct video processing.

 What it does:
   1. Reads video files from input directory
   2. Detects construction equipment (YOLOv8 + ByteTrack)
   3. Analyzes motion via region-based optical flow
   4. Classifies activity (DIGGING, SWINGING, DUMPING, TRAVELING, WAITING)
   5. Tracks utilization state with debouncing
   6. Produces annotated output videos
   7. Generates CSV analytics + visual reports

 How to use on Kaggle:
   1. Create a new Kaggle notebook
   2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
   3. Upload your sample videos as a dataset
   4. Copy-paste this entire file into a notebook cell
   5. Update VIDEO_DIR and OUTPUT_DIR paths below
   6. Run!

===============================================================================
"""

# ── Cell 1: Install Dependencies ──
# !pip install ultralytics opencv-python-headless matplotlib pandas -q

import os
import sys
import glob
import time
import json
import logging
import csv
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter
from enum import Enum
from datetime import datetime

import cv2
import numpy as np
import torch

# Fix PyTorch 2.6+ Ultralytics compatibility
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **dict(k, weights_only=False))

from ultralytics import YOLO

# Optional: for charts in notebook
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EagleVision")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — EDIT THESE FOR YOUR KAGGLE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    All configuration in one place. Edit paths and thresholds here.
    """

    # ── Paths ──
    # Kaggle dataset path (update after uploading your videos)
    VIDEO_DIR = os.getenv("VIDEO_DIR", "/kaggle/input/sample-videos")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/kaggle/working/output_videos")
    EVENTS_CSV = os.getenv("EVENTS_CSV", "/kaggle/working/equipment_events.csv")
    SUMMARY_CSV = os.getenv("SUMMARY_CSV", "/kaggle/working/utilization_summary.csv")
    REPORT_DIR = os.getenv("REPORT_DIR", "/kaggle/working/reports")

    # ── YOLO11 (medium) — +6.6 mAP over v8s, fewer params ──
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11m.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))

    # Equipment class mapping from COCO to our labels
    COCO_EQUIPMENT_CLASSES = {
        7: "dump_truck",     # truck
        5: "dump_truck",     # bus (resembles large trucks)
        2: "vehicle",        # car
    }

    # ── Video Processing ──
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
    TARGET_FPS = int(os.getenv("TARGET_FPS", "8"))
    FRAME_RESIZE_WIDTH = int(os.getenv("FRAME_RESIZE_WIDTH", "1280"))
    FRAME_RESIZE_HEIGHT = int(os.getenv("FRAME_RESIZE_HEIGHT", "720"))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

    # ── Motion Analysis ──
    MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "2.0"))
    ARM_MOTION_THRESHOLD = float(os.getenv("ARM_MOTION_THRESHOLD", "1.5"))
    TRACK_MOTION_THRESHOLD = float(os.getenv("TRACK_MOTION_THRESHOLD", "2.5"))

    # ── State Management ──
    STATE_DEBOUNCE_FRAMES = int(os.getenv("STATE_DEBOUNCE_FRAMES", "5"))
    SLIDING_WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "10"))


# ═══════════════════════════════════════════════════════════════════════════════
#  EQUIPMENT DETECTOR (YOLOv8 + ByteTrack)
# ═══════════════════════════════════════════════════════════════════════════════

class EquipmentDetector:
    """Detects construction equipment using YOLOv8 with GPU acceleration."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.YOLO_MODEL
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD

        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Device: {self.device}")
        if self.device == "cuda":
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        self.class_names = self.model.names
        logger.info(f"Model classes: {self.class_names}")
        self._build_class_mapping()

    def _build_class_mapping(self):
        """Build mapping from model class IDs to equipment types."""
        self.equipment_mapping = {}

        custom_keywords = {
            "excavator": "excavator",
            "loader": "loader",
            "backhoe": "excavator",
            "dump_truck": "dump_truck",
            "dump truck": "dump_truck",
            "bulldozer": "bulldozer",
            "crane": "crane",
            "concrete_mixer": "concrete_mixer",
            "roller": "roller",
            "grader": "grader",
        }

        is_custom_model = False
        for class_id, class_name in self.class_names.items():
            name_lower = class_name.lower().replace(" ", "_")
            for keyword, equipment_type in custom_keywords.items():
                if keyword in name_lower:
                    self.equipment_mapping[class_id] = equipment_type
                    is_custom_model = True

        if not is_custom_model:
            self.equipment_mapping = Config.COCO_EQUIPMENT_CLASSES.copy()
            logger.info("Using COCO class mapping for equipment detection")
        else:
            logger.info(f"Using custom model mapping: {self.equipment_mapping}")

    def detect_with_tracking(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and track equipment using built-in ByteTrack."""
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())

                    if class_id not in self.equipment_mapping:
                        continue

                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].item())

                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].item())

                    equipment_class = self.equipment_mapping[class_id]

                    prefix_map = {
                        "excavator": "EX", "dump_truck": "DT",
                        "loader": "LD", "bulldozer": "BD",
                        "crane": "CR", "vehicle": "VH",
                        "concrete_mixer": "CM", "roller": "RL",
                        "grader": "GR",
                    }
                    prefix = prefix_map.get(equipment_class, "EQ")
                    equipment_id = f"{prefix}-{track_id:03d}" if track_id else f"{prefix}-UNK"

                    detections.append({
                        "bbox": bbox,
                        "class_id": class_id,
                        "equipment_class": equipment_class,
                        "confidence": confidence,
                        "track_id": track_id,
                        "equipment_id": equipment_id,
                    })

        return detections


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTION ANALYZER (Region-Based Optical Flow)
# ═══════════════════════════════════════════════════════════════════════════════

class MotionAnalyzer:
    """Analyzes motion within equipment bounding boxes using optical flow."""

    def __init__(self):
        self.prev_gray = None
        self.motion_threshold = Config.MOTION_THRESHOLD
        self.arm_threshold = Config.ARM_MOTION_THRESHOLD
        self.track_threshold = Config.TRACK_MOTION_THRESHOLD

        self.flow_params = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        }

        self.motion_history: Dict[str, deque] = {}
        self.history_size = Config.SLIDING_WINDOW_SIZE

        logger.info(
            f"MotionAnalyzer initialized — thresholds: "
            f"general={self.motion_threshold}, "
            f"arm={self.arm_threshold}, "
            f"track={self.track_threshold}"
        )

    def compute_optical_flow(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """Compute dense optical flow between current and previous frame."""
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray, None, **self.flow_params,
        )
        self.prev_gray = frame_gray.copy()
        return flow

    def analyze_equipment_motion(
        self, flow: np.ndarray, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Analyze motion for each detected equipment using region-based approach."""
        h, w = frame_shape[:2]

        for det in detections:
            bbox = det["bbox"]
            equipment_id = det.get("equipment_id", "UNK")

            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                det["motion"] = self._empty_motion()
                continue

            flow_roi = flow[y1:y2, x1:x2]
            motion_result = self._analyze_regions(
                flow_roi, det.get("equipment_class", "unknown"), x2 - x1, y2 - y1
            )

            if equipment_id not in self.motion_history:
                self.motion_history[equipment_id] = deque(maxlen=self.history_size)
            self.motion_history[equipment_id].append(motion_result)

            smoothed = self._smooth_motion(equipment_id)
            det["motion"] = smoothed

        return detections

    def _analyze_regions(
        self, flow_roi: np.ndarray, equipment_class: str, roi_w: int, roi_h: int,
    ) -> Dict[str, Any]:
        """
        Divide equipment ROI into sub-regions and analyze motion.

        Sub-regions:
        ┌─────────────────────┐
        │   Upper Region      │  ← Arm/Boom area (top 45%)
        ├─────────────────────┤
        │   Middle Region     │  ← Cab/Swing area (middle 20%)
        ├─────────────────────┤
        │   Lower Region      │  ← Track/Wheel area (bottom 35%)
        └─────────────────────┘
        """
        if flow_roi.size == 0:
            return self._empty_motion()

        mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])
        overall_magnitude = float(np.mean(mag))
        overall_max = float(np.max(mag))

        upper_end = int(roi_h * 0.45)
        middle_end = int(roi_h * 0.65)

        upper_mag = np.mean(mag[:upper_end, :]) if upper_end > 0 else 0
        middle_mag = np.mean(mag[upper_end:middle_end, :]) if middle_end > upper_end else 0
        lower_mag = np.mean(mag[middle_end:, :]) if middle_end < roi_h else 0

        mid_x = roi_w // 2
        left_mag = np.mean(mag[:, :mid_x]) if mid_x > 0 else 0
        right_mag = np.mean(mag[:, mid_x:]) if mid_x < roi_w else 0

        upper_flow = flow_roi[:upper_end, :]
        if upper_flow.size > 0:
            avg_dx = float(np.mean(upper_flow[..., 0]))
            avg_dy = float(np.mean(upper_flow[..., 1]))
        else:
            avg_dx, avg_dy = 0.0, 0.0

        lateral_asymmetry = abs(float(left_mag) - float(right_mag))

        is_upper_active = float(upper_mag) > self.arm_threshold
        is_middle_active = float(middle_mag) > self.motion_threshold
        is_lower_active = float(lower_mag) > self.track_threshold
        has_lateral_motion = lateral_asymmetry > self.motion_threshold

        is_active = is_upper_active or is_middle_active or is_lower_active or has_lateral_motion

        if is_active:
            if is_upper_active and not is_lower_active:
                motion_source = "arm_only"
            elif is_lower_active and not is_upper_active:
                motion_source = "tracks_only"
            elif has_lateral_motion and is_middle_active:
                motion_source = "swing"
            elif is_upper_active and is_lower_active:
                motion_source = "full_body"
            else:
                motion_source = "partial"
        else:
            motion_source = "none"

        return {
            "is_active": is_active,
            "motion_source": motion_source,
            "overall_magnitude": round(overall_magnitude, 3),
            "overall_max": round(float(overall_max), 3),
            "upper_magnitude": round(float(upper_mag), 3),
            "middle_magnitude": round(float(middle_mag), 3),
            "lower_magnitude": round(float(lower_mag), 3),
            "left_magnitude": round(float(left_mag), 3),
            "right_magnitude": round(float(right_mag), 3),
            "lateral_asymmetry": round(lateral_asymmetry, 3),
            "arm_direction": {"dx": round(avg_dx, 3), "dy": round(avg_dy, 3)},
            "region_active": {
                "upper": is_upper_active,
                "middle": is_middle_active,
                "lower": is_lower_active,
                "lateral": has_lateral_motion,
            },
        }

    def _smooth_motion(self, equipment_id: str) -> Dict[str, Any]:
        """Apply temporal smoothing over the motion history window."""
        history = self.motion_history.get(equipment_id, [])
        if not history:
            return self._empty_motion()
        latest = history[-1].copy()
        if len(history) < 3:
            return latest
        active_count = sum(1 for h in history if h["is_active"])
        latest["is_active"] = active_count > len(history) * 0.4
        sources = [h["motion_source"] for h in history if h["motion_source"] != "none"]
        if sources:
            latest["motion_source"] = Counter(sources).most_common(1)[0][0]
        elif not latest["is_active"]:
            latest["motion_source"] = "none"
        latest["overall_magnitude"] = round(
            np.mean([h["overall_magnitude"] for h in history]), 3
        )
        return latest

    def _empty_motion(self) -> Dict[str, Any]:
        return {
            "is_active": False, "motion_source": "none",
            "overall_magnitude": 0.0, "overall_max": 0.0,
            "upper_magnitude": 0.0, "middle_magnitude": 0.0,
            "lower_magnitude": 0.0, "left_magnitude": 0.0,
            "right_magnitude": 0.0, "lateral_asymmetry": 0.0,
            "arm_direction": {"dx": 0.0, "dy": 0.0},
            "region_active": {"upper": False, "middle": False, "lower": False, "lateral": False},
        }

    def reset(self):
        self.prev_gray = None
        self.motion_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVITY CLASSIFIER (Rule-Based + Temporal Smoothing)
# ═══════════════════════════════════════════════════════════════════════════════

class Activity(str, Enum):
    DIGGING = "DIGGING"
    SWINGING_LOADING = "SWINGING_LOADING"
    DUMPING = "DUMPING"
    WAITING = "WAITING"
    TRAVELING = "TRAVELING"
    IDLE = "IDLE"


class ActivityClassifier:
    """Rule-based activity classifier using motion analysis features."""

    def __init__(self):
        self.window_size = Config.SLIDING_WINDOW_SIZE
        self.feature_windows: Dict[str, deque] = {}
        self.activity_history: Dict[str, deque] = {}
        logger.info("ActivityClassifier initialized")

    def classify(
        self, equipment_id: str, equipment_class: str, motion: Dict[str, Any],
    ) -> Dict[str, Any]:
        if equipment_id not in self.feature_windows:
            self.feature_windows[equipment_id] = deque(maxlen=self.window_size)
            self.activity_history[equipment_id] = deque(maxlen=self.window_size)

        self.feature_windows[equipment_id].append(motion)

        if equipment_class in ("excavator", "backhoe"):
            activity, confidence = self._classify_excavator(equipment_id, motion)
        elif equipment_class == "dump_truck":
            activity, confidence = self._classify_dump_truck(equipment_id, motion)
        elif equipment_class in ("loader", "bulldozer"):
            activity, confidence = self._classify_loader(equipment_id, motion)
        else:
            activity, confidence = self._classify_generic(equipment_id, motion)

        self.activity_history[equipment_id].append(activity)
        smoothed_activity = self._smooth_activity(equipment_id, activity)

        return {
            "activity": smoothed_activity.value,
            "confidence": round(confidence, 2),
            "raw_activity": activity.value,
        }

    def _classify_excavator(self, equipment_id: str, motion: Dict[str, Any]) -> tuple:
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        middle = motion.get("middle_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)
        lateral = motion.get("lateral_asymmetry", 0)
        arm_dir = motion.get("arm_direction", {"dx": 0, "dy": 0})

        window = self.feature_windows[equipment_id]
        recent_dy_avg = 0
        if len(window) >= 3:
            recent_dy_avg = sum(
                w.get("arm_direction", {}).get("dy", 0) for w in window
            ) / len(window)

        if (lateral > Config.MOTION_THRESHOLD * 1.5 or source == "swing") and (
            upper > Config.ARM_MOTION_THRESHOLD or middle > Config.MOTION_THRESHOLD
        ):
            confidence = min(0.95, 0.6 + lateral * 0.1)
            return Activity.SWINGING_LOADING, confidence

        if source in ("arm_only", "partial") and arm_dir["dy"] > 0.3:
            confidence = min(0.95, 0.6 + upper * 0.05 + arm_dir["dy"] * 0.2)
            return Activity.DIGGING, confidence

        if upper > Config.ARM_MOTION_THRESHOLD * 2 and lower < Config.TRACK_MOTION_THRESHOLD:
            if recent_dy_avg > 0:
                return Activity.DIGGING, 0.7

        if source in ("arm_only", "partial") and arm_dir["dy"] < -0.3:
            confidence = min(0.9, 0.5 + abs(arm_dir["dy"]) * 0.2)
            return Activity.DUMPING, confidence

        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.8

        if source == "arm_only":
            return Activity.DIGGING, 0.5

        return Activity.WAITING, 0.4

    def _classify_dump_truck(self, equipment_id: str, motion: Dict[str, Any]) -> tuple:
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)

        if source == "arm_only" and upper > Config.ARM_MOTION_THRESHOLD:
            return Activity.DUMPING, 0.8

        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.85

        if motion.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 0.5:
            window = self.feature_windows[equipment_id]
            if len(window) >= 3:
                mags = [w.get("overall_magnitude", 0) for w in window]
                variance = sum((m - sum(mags)/len(mags))**2 for m in mags) / len(mags)
                if variance > 0.5:
                    return Activity.SWINGING_LOADING, 0.6

        return Activity.WAITING, 0.5

    def _classify_loader(self, equipment_id: str, motion: Dict[str, Any]) -> tuple:
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]

        if source == "arm_only":
            arm_dir = motion.get("arm_direction", {"dy": 0})
            if arm_dir["dy"] > 0:
                return Activity.DIGGING, 0.7
            else:
                return Activity.DUMPING, 0.7

        if source in ("full_body", "tracks_only"):
            return Activity.TRAVELING, 0.8

        return Activity.WAITING, 0.5

    def _classify_generic(self, equipment_id: str, motion: Dict[str, Any]) -> tuple:
        if not motion["is_active"]:
            return Activity.WAITING, 0.9
        if motion.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 2:
            return Activity.TRAVELING, 0.6
        return Activity.WAITING, 0.5

    def _smooth_activity(self, equipment_id: str, current: Activity) -> Activity:
        history = self.activity_history.get(equipment_id, [])
        if len(history) < 3:
            return current
        recent = list(history)[-5:]
        counts = Counter(recent)
        most_common = counts.most_common(1)[0][0]
        if counts[current] >= len(recent) * 0.4:
            return current
        return most_common

    def reset(self):
        self.feature_windows.clear()
        self.activity_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE MANAGER (Debouncing + Utilization Tracking)
# ═══════════════════════════════════════════════════════════════════════════════

class EquipmentState:
    """Tracks the state of a single piece of equipment."""

    def __init__(self, equipment_id: str, equipment_class: str):
        self.equipment_id = equipment_id
        self.equipment_class = equipment_class
        self.current_state = "INACTIVE"
        self.current_activity = "WAITING"
        self.motion_source = "none"
        self.total_tracked_seconds = 0.0
        self.total_active_seconds = 0.0
        self.total_idle_seconds = 0.0
        self.utilization_percent = 0.0
        self.last_update_time = None
        self.first_seen_time = None
        self.frame_count = 0
        self.pending_state = None
        self.pending_state_count = 0
        self.debounce_threshold = Config.STATE_DEBOUNCE_FRAMES
        self.transitions = []

    def update(
        self, is_active: bool, activity: str, motion_source: str,
        timestamp_sec: float, frame_id: int,
    ) -> Dict[str, Any]:
        self.frame_count += 1

        if self.first_seen_time is None:
            self.first_seen_time = timestamp_sec
            self.last_update_time = timestamp_sec

        dt = timestamp_sec - self.last_update_time
        if dt < 0:
            dt = 0
        self.last_update_time = timestamp_sec

        self.total_tracked_seconds = timestamp_sec - self.first_seen_time
        if self.total_tracked_seconds < 0:
            self.total_tracked_seconds = 0

        if self.current_state == "ACTIVE":
            self.total_active_seconds += dt
        else:
            self.total_idle_seconds += dt

        new_state = "ACTIVE" if is_active else "INACTIVE"
        self._debounce_state(new_state, timestamp_sec)

        if self.current_state == "ACTIVE":
            self.current_activity = activity
            self.motion_source = motion_source
        else:
            self.current_activity = "WAITING"
            self.motion_source = "none"

        if self.total_tracked_seconds > 0:
            self.utilization_percent = round(
                (self.total_active_seconds / self.total_tracked_seconds) * 100, 1
            )
        else:
            self.utilization_percent = 0.0

        return self.get_snapshot()

    def _debounce_state(self, new_state: str, timestamp_sec: float):
        if new_state == self.current_state:
            self.pending_state = None
            self.pending_state_count = 0
            return
        if new_state == self.pending_state:
            self.pending_state_count += 1
        else:
            self.pending_state = new_state
            self.pending_state_count = 1
        if self.pending_state_count >= self.debounce_threshold:
            old_state = self.current_state
            self.current_state = new_state
            self.pending_state = None
            self.pending_state_count = 0
            self.transitions.append({
                "from": old_state, "to": new_state,
                "timestamp_sec": timestamp_sec,
            })
            logger.debug(
                f"  {self.equipment_id}: {old_state} → {new_state} at t={timestamp_sec:.1f}s"
            )

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "equipment_class": self.equipment_class,
            "current_state": self.current_state,
            "current_activity": self.current_activity,
            "motion_source": self.motion_source,
            "total_tracked_seconds": round(self.total_tracked_seconds, 1),
            "total_active_seconds": round(self.total_active_seconds, 1),
            "total_idle_seconds": round(self.total_idle_seconds, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "frame_count": self.frame_count,
        }


class StateManager:
    """Manages state for all tracked equipment."""

    def __init__(self):
        self.equipment_states: Dict[str, EquipmentState] = {}
        logger.info("StateManager initialized")

    def update_equipment(
        self, equipment_id: str, equipment_class: str, is_active: bool,
        activity: str, motion_source: str, timestamp_sec: float, frame_id: int,
    ) -> Dict[str, Any]:
        if equipment_id not in self.equipment_states:
            self.equipment_states[equipment_id] = EquipmentState(equipment_id, equipment_class)
            logger.info(f"New equipment tracked: {equipment_id} ({equipment_class})")

        return self.equipment_states[equipment_id].update(
            is_active=is_active, activity=activity,
            motion_source=motion_source, timestamp_sec=timestamp_sec,
            frame_id=frame_id,
        )

    def get_summary(self) -> Dict[str, Any]:
        if not self.equipment_states:
            return {"total_equipment": 0}

        total_active = sum(s.total_active_seconds for s in self.equipment_states.values())
        total_idle = sum(s.total_idle_seconds for s in self.equipment_states.values())
        total_tracked = sum(s.total_tracked_seconds for s in self.equipment_states.values())

        return {
            "total_equipment": len(self.equipment_states),
            "total_active_seconds": round(total_active, 1),
            "total_idle_seconds": round(total_idle, 1),
            "total_tracked_seconds": round(total_tracked, 1),
            "overall_utilization_percent": round(
                (total_active / total_tracked) * 100, 1
            ) if total_tracked > 0 else 0,
            "equipment": {
                eid: state.get_snapshot()
                for eid, state in self.equipment_states.items()
            },
        }

    def reset(self):
        self.equipment_states.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "ACTIVE": (0, 255, 0),        # Green
    "INACTIVE": (0, 0, 255),      # Red
    "DIGGING": (0, 200, 255),     # Orange
    "SWINGING_LOADING": (255, 200, 0),  # Cyan
    "DUMPING": (200, 0, 255),     # Purple
    "TRAVELING": (255, 255, 0),   # Yellow
    "WAITING": (128, 128, 128),   # Gray
    "IDLE": (128, 128, 128),
}


def draw_annotations(
    frame: np.ndarray, detections: list, states: dict,
) -> np.ndarray:
    """Draw bounding boxes, labels, and status on frame."""
    annotated = frame.copy()

    for det in detections:
        bbox = det["bbox"]
        equipment_id = det.get("equipment_id", "UNK")
        equipment_class = det.get("equipment_class", "unknown")

        state_info = states.get(equipment_id, {})
        current_state = state_info.get("current_state", "INACTIVE")
        activity = state_info.get("current_activity", "WAITING")
        utilization = state_info.get("utilization_percent", 0)
        motion_source = state_info.get("motion_source", "none")

        color = COLORS.get(current_state, (255, 255, 255))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        thickness = 3 if current_state == "ACTIVE" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label_lines = [
            f"{equipment_id} | {equipment_class}",
            f"State: {current_state} | {activity}",
            f"Util: {utilization:.1f}% | Motion: {motion_source}",
        ]

        y_offset = y1 - 10
        for i, line in enumerate(reversed(label_lines)):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            y_pos = y_offset - i * 22
            cv2.rectangle(
                annotated, (x1, y_pos - 15),
                (x1 + text_size[0] + 8, y_pos + 5), color, -1,
            )
            cv2.putText(
                annotated, line, (x1 + 4, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # Motion magnitude bar
        motion = det.get("motion", {})
        mag = motion.get("overall_magnitude", 0)
        bar_width = min(int(mag * 20), x2 - x1)
        cv2.rectangle(
            annotated, (x1, y2 + 2), (x1 + bar_width, y2 + 8),
            COLORS.get(activity, (255, 255, 255)), -1,
        )

    _draw_global_overlay(annotated, states)
    return annotated


def _draw_global_overlay(frame: np.ndarray, states: dict):
    """Draw global statistics overlay in top-right corner."""
    if not states:
        return
    h, w = frame.shape[:2]
    active_count = sum(1 for s in states.values() if s.get("current_state") == "ACTIVE")
    total_count = len(states)

    overlay_lines = [
        f"Equipment: {total_count} tracked",
        f"Active: {active_count} | Idle: {total_count - active_count}",
    ]

    x_start = w - 300
    y_start = 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 10, 5), (w - 5, y_start + len(overlay_lines) * 25 + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(overlay_lines):
        cv2.putText(
            frame, line, (x_start, y_start + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV EVENT LOGGER (replaces Kafka + TimescaleDB)
# ═══════════════════════════════════════════════════════════════════════════════

class EventLogger:
    """Logs equipment events to CSV file (replaces Kafka + DB pipeline)."""

    CSV_HEADERS = [
        "timestamp", "frame_id", "video_name", "equipment_id", "equipment_class",
        "current_state", "current_activity", "motion_source", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "total_tracked_seconds", "total_active_seconds", "total_idle_seconds",
        "utilization_percent",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.file = open(csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.CSV_HEADERS)
        self.writer.writeheader()
        self.event_count = 0
        logger.info(f"EventLogger: writing to {csv_path}")

    def log_event(
        self, frame_id: int, video_name: str, timestamp: str,
        det: Dict, state_snapshot: Dict,
    ):
        """Log a single equipment event."""
        bbox = det["bbox"]
        row = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "video_name": video_name,
            "equipment_id": det.get("equipment_id", "UNK"),
            "equipment_class": det.get("equipment_class", "unknown"),
            "current_state": state_snapshot.get("current_state", "INACTIVE"),
            "current_activity": state_snapshot.get("current_activity", "WAITING"),
            "motion_source": state_snapshot.get("motion_source", "none"),
            "confidence": det.get("confidence", 0),
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2] - bbox[0],
            "bbox_h": bbox[3] - bbox[1],
            "total_tracked_seconds": state_snapshot.get("total_tracked_seconds", 0),
            "total_active_seconds": state_snapshot.get("total_active_seconds", 0),
            "total_idle_seconds": state_snapshot.get("total_idle_seconds", 0),
            "utilization_percent": state_snapshot.get("utilization_percent", 0),
        }
        self.writer.writerow(row)
        self.event_count += 1

    def close(self):
        self.file.close()
        logger.info(f"EventLogger: {self.event_count} events written to {self.csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR (replaces Streamlit Dashboard)
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates visual utilization reports (replaces the Streamlit dashboard)."""

    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def generate_summary_csv(self, state_manager: StateManager, video_name: str):
        """Write per-equipment utilization summary to CSV."""
        summary = state_manager.get_summary()
        csv_path = os.path.join(self.report_dir, f"{video_name}_summary.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "equipment_id", "equipment_class", "current_state",
                "total_tracked_seconds", "total_active_seconds",
                "total_idle_seconds", "utilization_percent", "frame_count",
            ])
            for eid, state in summary.get("equipment", {}).items():
                writer.writerow([
                    eid, state["equipment_class"], state["current_state"],
                    state["total_tracked_seconds"], state["total_active_seconds"],
                    state["total_idle_seconds"], state["utilization_percent"],
                    state["frame_count"],
                ])

        logger.info(f"Summary CSV: {csv_path}")
        return csv_path

    def generate_charts(self, events_csv: str, video_name: str):
        """Generate utilization charts from event CSV."""
        if not HAS_PLOTTING:
            logger.warning("matplotlib/pandas not available, skipping charts")
            return

        try:
            df = pd.read_csv(events_csv)
            if df.empty:
                return

            # Filter for this video
            vdf = df[df["video_name"] == video_name] if "video_name" in df.columns else df
            if vdf.empty:
                return

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"🦅 Eagle Vision — {video_name}", fontsize=16, fontweight="bold")

            # 1. Utilization over time per equipment
            ax = axes[0, 0]
            for eid in vdf["equipment_id"].unique():
                edf = vdf[vdf["equipment_id"] == eid]
                ax.plot(edf["frame_id"], edf["utilization_percent"], label=eid, linewidth=1.5)
            ax.set_title("Utilization Over Time")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Utilization %")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 2. State distribution per equipment
            ax = axes[0, 1]
            state_counts = vdf.groupby(["equipment_id", "current_state"]).size().unstack(fill_value=0)
            state_counts.plot(kind="bar", ax=ax, colormap="Set2")
            ax.set_title("State Distribution")
            ax.set_xlabel("Equipment")
            ax.set_ylabel("Frame Count")
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=8)

            # 3. Activity distribution per equipment
            ax = axes[1, 0]
            activity_counts = vdf.groupby(["equipment_id", "current_activity"]).size().unstack(fill_value=0)
            activity_counts.plot(kind="bar", ax=ax, colormap="tab10")
            ax.set_title("Activity Distribution")
            ax.set_xlabel("Equipment")
            ax.set_ylabel("Frame Count")
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=8)

            # 4. Final utilization summary (horizontal bar)
            ax = axes[1, 1]
            summary = vdf.groupby("equipment_id")["utilization_percent"].last().sort_values()
            colors = ["#00ff88" if v > 50 else "#ff4444" for v in summary.values]
            summary.plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Final Utilization %")
            ax.set_xlabel("Utilization %")
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            chart_path = os.path.join(self.report_dir, f"{video_name}_charts.png")
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Charts saved: {chart_path}")

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")

    def print_summary(self, state_manager: StateManager, video_name: str):
        """Print a formatted utilization summary to console."""
        summary = state_manager.get_summary()

        print("\n" + "=" * 70)
        print(f"  🦅 EAGLE VISION — UTILIZATION REPORT")
        print(f"  Video: {video_name}")
        print("=" * 70)
        print(f"  Total Equipment Tracked: {summary['total_equipment']}")

        if summary["total_equipment"] > 0:
            print(f"  Overall Utilization:     {summary['overall_utilization_percent']}%")
            print(f"  Total Active Time:       {summary['total_active_seconds']}s")
            print(f"  Total Idle Time:         {summary['total_idle_seconds']}s")
            print(f"  Total Tracked Time:      {summary['total_tracked_seconds']}s")
            print("-" * 70)
            print(f"  {'Equipment':<12} {'Class':<14} {'Active(s)':<12} {'Idle(s)':<12} {'Util %':<10}")
            print("-" * 70)
            for eid, state in summary.get("equipment", {}).items():
                print(
                    f"  {eid:<12} {state['equipment_class']:<14} "
                    f"{state['total_active_seconds']:<12} "
                    f"{state['total_idle_seconds']:<12} "
                    f"{state['utilization_percent']:<10}"
                )
        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_files(video_dir: str) -> List[str]:
    """Find all video files in the specified directory."""
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        # Also check case variants
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    video_files = list(set(video_files))  # deduplicate
    video_files.sort()
    return video_files


def process_video(
    video_path: str,
    detector: EquipmentDetector,
    motion_analyzer: MotionAnalyzer,
    activity_classifier: ActivityClassifier,
    state_manager: StateManager,
    event_logger: EventLogger,
    output_dir: str,
):
    """Process a single video through the full pipeline."""
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {video_name}")
    logger.info(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    logger.info(
        f"  Video: {width}x{height}, {original_fps:.1f} FPS, "
        f"{total_frames} frames, {duration:.1f}s"
    )

    # Setup video writer for annotated output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    target_size = (Config.FRAME_RESIZE_WIDTH, Config.FRAME_RESIZE_HEIGHT)
    writer = cv2.VideoWriter(output_path, fourcc, Config.TARGET_FPS, target_size)

    # Calculate frame skip
    frame_skip = max(1, int(original_fps / Config.TARGET_FPS)) if original_fps > 0 else Config.FRAME_SKIP

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Reset per-video state
    motion_analyzer.reset()
    activity_classifier.reset()
    state_manager.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to match target FPS
        if frame_count % frame_skip != 0:
            continue

        # Resize frame
        frame_resized = cv2.resize(frame, target_size)

        # Calculate timestamp
        timestamp_sec = frame_count / original_fps if original_fps > 0 else 0
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = timestamp_sec % 60
        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        # ── Step 1: Detect equipment ──
        detections = detector.detect_with_tracking(frame_resized)

        # ── Step 2: Compute optical flow ──
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        flow = motion_analyzer.compute_optical_flow(frame_gray)

        if flow is not None and detections:
            # ── Step 3: Analyze motion per equipment ──
            detections = motion_analyzer.analyze_equipment_motion(
                flow, detections, frame_resized.shape
            )

            # ── Step 4: Classify activity & update state ──
            all_states = {}
            for det in detections:
                equipment_id = det.get("equipment_id", "UNK")
                equipment_class = det.get("equipment_class", "unknown")
                motion = det.get("motion", {})

                activity_result = activity_classifier.classify(
                    equipment_id, equipment_class, motion
                )

                state_snapshot = state_manager.update_equipment(
                    equipment_id=equipment_id,
                    equipment_class=equipment_class,
                    is_active=motion.get("is_active", False),
                    activity=activity_result["activity"],
                    motion_source=motion.get("motion_source", "none"),
                    timestamp_sec=timestamp_sec,
                    frame_id=frame_count,
                )

                all_states[equipment_id] = state_snapshot

                # Log event to CSV (replaces Kafka → DB pipeline)
                event_logger.log_event(
                    frame_id=frame_count,
                    video_name=video_name,
                    timestamp=timestamp_str,
                    det=det,
                    state_snapshot=state_snapshot,
                )

            # ── Step 5: Draw annotations & write frame ──
            annotated_frame = draw_annotations(frame_resized, detections, all_states)
            writer.write(annotated_frame)
        else:
            # No detections or first frame — write original
            writer.write(frame_resized)

        processed_count += 1
        if processed_count % 50 == 0:
            elapsed = time.time() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Processed {processed_count} frames ({fps:.1f} FPS) — "
                f"Tracking {len(state_manager.equipment_states)} equipment "
                f"[{frame_count}/{total_frames}]"
            )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    fps = processed_count / elapsed if elapsed > 0 else 0

    logger.info(f"\n  ✅ Finished: {video_name}")
    logger.info(f"     Processed {processed_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
    logger.info(f"     Output: {output_path}")

    return video_name


def main():
    """
    ═══════════════════════════════════════════════════════════
     EAGLE VISION — KAGGLE GPU PIPELINE
    ═══════════════════════════════════════════════════════════
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🦅  EAGLE VISION — Kaggle GPU Pipeline                  ║
    ║  Real-Time Equipment Utilization & Activity Classification║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # System info
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    logger.info(f"OpenCV: {cv2.__version__}")

    # Find videos
    video_dir = Config.VIDEO_DIR
    logger.info(f"\nLooking for videos in: {video_dir}")

    video_files = get_video_files(video_dir)
    if not video_files:
        logger.error(f"❌ No video files found in {video_dir}")
        logger.info("   Please upload videos to a Kaggle dataset and update VIDEO_DIR")
        logger.info("   Supported formats: .mp4, .avi, .mov, .mkv, .webm")
        return

    logger.info(f"Found {len(video_files)} video(s):")
    for vf in video_files:
        logger.info(f"  📹 {os.path.basename(vf)}")

    # Initialize components
    logger.info("\nInitializing pipeline components...")
    detector = EquipmentDetector()
    motion_analyzer = MotionAnalyzer()
    activity_classifier = ActivityClassifier()
    state_manager = StateManager()
    event_logger = EventLogger(Config.EVENTS_CSV)
    report_generator = ReportGenerator(Config.REPORT_DIR)

    # Process each video
    total_start = time.time()

    for idx, video_path in enumerate(video_files):
        video_name = process_video(
            video_path=video_path,
            detector=detector,
            motion_analyzer=motion_analyzer,
            activity_classifier=activity_classifier,
            state_manager=state_manager,
            event_logger=event_logger,
            output_dir=Config.OUTPUT_DIR,
        )

        if video_name:
            # Print summary
            report_generator.print_summary(state_manager, video_name)

            # Generate per-video summary CSV
            report_generator.generate_summary_csv(state_manager, os.path.splitext(video_name)[0])

    # Close event logger
    event_logger.close()

    # Generate charts from all events
    logger.info("\n📊 Generating analytics reports...")
    for vf in video_files:
        vname = os.path.basename(vf)
        report_generator.generate_charts(Config.EVENTS_CSV, vname)

    total_elapsed = time.time() - total_start
    logger.info(f"\n🎉 All done! Total time: {total_elapsed:.1f}s")
    logger.info(f"   📁 Output videos: {Config.OUTPUT_DIR}")
    logger.info(f"   📊 Events CSV:    {Config.EVENTS_CSV}")
    logger.info(f"   📈 Reports:       {Config.REPORT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
