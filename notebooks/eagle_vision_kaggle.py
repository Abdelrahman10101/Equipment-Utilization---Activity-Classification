# 🦅 Eagle Vision — Kaggle GPU Notebook
# 
# This notebook allows you to run the CV Processor on Kaggle's free GPU.
# It processes a video file directly (without Kafka) and saves results.
#
# HOW TO USE:
# 1. Upload this file as a Kaggle notebook
# 2. Upload your video file(s) to Kaggle as a dataset
# 3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
# 4. Run all cells
#
# The output will be:
# - Annotated video with bounding boxes
# - CSV file with all equipment events
# - Utilization summary

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install ultralytics opencv-python-headless numpy

# %% [markdown]
# ## 2. Import Modules

# %%
import os
import sys
import cv2
import json
import time
import base64
import numpy as np
import csv
from collections import deque, Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 3. Configuration

# %%
class Config:
    """Configuration for standalone CV processing."""
    YOLO_MODEL = "yolov8m.pt"
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    MOTION_THRESHOLD = 2.0
    ARM_MOTION_THRESHOLD = 1.5
    TRACK_MOTION_THRESHOLD = 2.5
    STATE_DEBOUNCE_FRAMES = 5
    SLIDING_WINDOW_SIZE = 10
    TARGET_FPS = 10
    
    # Output
    OUTPUT_DIR = "/kaggle/working/output"
    
    # COCO class mapping
    COCO_EQUIPMENT_CLASSES = {
        7: "dump_truck",
        5: "dump_truck",
        2: "vehicle",
    }

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## 4. Core CV Components
# 
# Copy of the microservice components adapted for standalone use.

# %%
# ── Equipment Detector ──

from ultralytics import YOLO

class EquipmentDetector:
    def __init__(self):
        print(f"Loading YOLOv8 model: {Config.YOLO_MODEL}")
        self.model = YOLO(Config.YOLO_MODEL)
        self.class_names = self.model.names
        self._build_class_mapping()
        print(f"Model loaded. Classes: {self.class_names}")
    
    def _build_class_mapping(self):
        self.equipment_mapping = {}
        custom_keywords = {
            "excavator": "excavator", "loader": "loader",
            "backhoe": "excavator", "dump_truck": "dump_truck",
            "dump truck": "dump_truck", "bulldozer": "bulldozer",
            "crane": "crane",
        }
        
        is_custom = False
        for class_id, class_name in self.class_names.items():
            name_lower = class_name.lower().replace(" ", "_")
            for kw, eq_type in custom_keywords.items():
                if kw in name_lower:
                    self.equipment_mapping[class_id] = eq_type
                    is_custom = True
        
        if not is_custom:
            self.equipment_mapping = Config.COCO_EQUIPMENT_CLASSES.copy()
            print("Using COCO class mapping")
        else:
            print(f"Using custom mapping: {self.equipment_mapping}")
    
    def detect_with_tracking(self, frame):
        results = self.model.track(
            frame, conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD, tracker="bytetrack.yaml",
            persist=True, verbose=False,
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
                    track_id = int(boxes.id[i].item()) if boxes.id is not None else None
                    
                    eq_class = self.equipment_mapping[class_id]
                    prefix_map = {"excavator": "EX", "dump_truck": "DT", "loader": "LD",
                                  "bulldozer": "BD", "crane": "CR", "vehicle": "VH"}
                    prefix = prefix_map.get(eq_class, "EQ")
                    eq_id = f"{prefix}-{track_id:03d}" if track_id else f"{prefix}-UNK"
                    
                    detections.append({
                        "bbox": bbox, "class_id": class_id,
                        "equipment_class": eq_class, "confidence": confidence,
                        "track_id": track_id, "equipment_id": eq_id,
                    })
        return detections

# %%
# ── Motion Analyzer ──

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None
        self.motion_history = {}
        self.flow_params = {
            "pyr_scale": 0.5, "levels": 3, "winsize": 15,
            "iterations": 3, "poly_n": 5, "poly_sigma": 1.2,
            "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        }
    
    def compute_optical_flow(self, frame_gray):
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame_gray, None, **self.flow_params)
        self.prev_gray = frame_gray.copy()
        return flow
    
    def analyze_equipment_motion(self, flow, detections, frame_shape):
        h, w = frame_shape[:2]
        for det in detections:
            bbox = det["bbox"]
            eq_id = det.get("equipment_id", "UNK")
            
            x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
            
            if x2 <= x1 or y2 <= y1:
                det["motion"] = self._empty()
                continue
            
            flow_roi = flow[y1:y2, x1:x2]
            motion = self._analyze_regions(flow_roi, x2-x1, y2-y1)
            
            if eq_id not in self.motion_history:
                self.motion_history[eq_id] = deque(maxlen=Config.SLIDING_WINDOW_SIZE)
            self.motion_history[eq_id].append(motion)
            
            det["motion"] = self._smooth(eq_id)
        return detections
    
    def _analyze_regions(self, flow_roi, roi_w, roi_h):
        if flow_roi.size == 0:
            return self._empty()
        
        mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])
        overall_mag = float(np.mean(mag))
        
        upper_end = int(roi_h * 0.45)
        middle_end = int(roi_h * 0.65)
        
        upper_mag = float(np.mean(mag[:upper_end, :])) if upper_end > 0 else 0
        middle_mag = float(np.mean(mag[upper_end:middle_end, :])) if middle_end > upper_end else 0
        lower_mag = float(np.mean(mag[middle_end:, :])) if middle_end < roi_h else 0
        
        mid_x = roi_w // 2
        left_mag = float(np.mean(mag[:, :mid_x])) if mid_x > 0 else 0
        right_mag = float(np.mean(mag[:, mid_x:])) if mid_x < roi_w else 0
        lateral = abs(left_mag - right_mag)
        
        upper_flow = flow_roi[:upper_end, :]
        avg_dx = float(np.mean(upper_flow[..., 0])) if upper_flow.size > 0 else 0
        avg_dy = float(np.mean(upper_flow[..., 1])) if upper_flow.size > 0 else 0
        
        is_upper = upper_mag > Config.ARM_MOTION_THRESHOLD
        is_middle = middle_mag > Config.MOTION_THRESHOLD
        is_lower = lower_mag > Config.TRACK_MOTION_THRESHOLD
        has_lateral = lateral > Config.MOTION_THRESHOLD
        
        is_active = is_upper or is_middle or is_lower or has_lateral
        
        if is_active:
            if is_upper and not is_lower: source = "arm_only"
            elif is_lower and not is_upper: source = "tracks_only"
            elif has_lateral and is_middle: source = "swing"
            elif is_upper and is_lower: source = "full_body"
            else: source = "partial"
        else:
            source = "none"
        
        return {
            "is_active": is_active, "motion_source": source,
            "overall_magnitude": round(overall_mag, 3),
            "upper_magnitude": round(upper_mag, 3),
            "middle_magnitude": round(middle_mag, 3),
            "lower_magnitude": round(lower_mag, 3),
            "lateral_asymmetry": round(lateral, 3),
            "arm_direction": {"dx": round(avg_dx, 3), "dy": round(avg_dy, 3)},
            "region_active": {"upper": is_upper, "middle": is_middle, "lower": is_lower, "lateral": has_lateral},
        }
    
    def _smooth(self, eq_id):
        history = list(self.motion_history.get(eq_id, []))
        if not history: return self._empty()
        latest = history[-1].copy()
        if len(history) < 3: return latest
        
        active_count = sum(1 for h in history if h["is_active"])
        latest["is_active"] = active_count > len(history) * 0.4
        
        sources = [h["motion_source"] for h in history if h["motion_source"] != "none"]
        if sources:
            latest["motion_source"] = Counter(sources).most_common(1)[0][0]
        elif not latest["is_active"]:
            latest["motion_source"] = "none"
        
        return latest
    
    def _empty(self):
        return {
            "is_active": False, "motion_source": "none",
            "overall_magnitude": 0, "upper_magnitude": 0,
            "middle_magnitude": 0, "lower_magnitude": 0,
            "lateral_asymmetry": 0, "arm_direction": {"dx": 0, "dy": 0},
            "region_active": {"upper": False, "middle": False, "lower": False, "lateral": False},
        }

# %%
# ── Activity Classifier ──

class ActivityClassifier:
    def __init__(self):
        self.feature_windows = {}
        self.activity_history = {}
    
    def classify(self, eq_id, eq_class, motion):
        if eq_id not in self.feature_windows:
            self.feature_windows[eq_id] = deque(maxlen=Config.SLIDING_WINDOW_SIZE)
            self.activity_history[eq_id] = deque(maxlen=Config.SLIDING_WINDOW_SIZE)
        
        self.feature_windows[eq_id].append(motion)
        
        if eq_class in ("excavator", "backhoe"):
            activity, conf = self._classify_excavator(eq_id, motion)
        elif eq_class == "dump_truck":
            activity, conf = self._classify_dump_truck(eq_id, motion)
        else:
            activity, conf = self._classify_generic(motion)
        
        self.activity_history[eq_id].append(activity)
        smoothed = self._smooth(eq_id, activity)
        
        return {"activity": smoothed, "confidence": round(conf, 2)}
    
    def _classify_excavator(self, eq_id, m):
        if not m["is_active"]: return "WAITING", 0.9
        
        source = m["motion_source"]
        upper = m.get("upper_magnitude", 0)
        lateral = m.get("lateral_asymmetry", 0)
        arm_dy = m.get("arm_direction", {}).get("dy", 0)
        lower = m.get("lower_magnitude", 0)
        
        if lateral > Config.MOTION_THRESHOLD * 1.5 or source == "swing":
            return "SWINGING_LOADING", min(0.95, 0.6 + lateral * 0.1)
        if source in ("arm_only", "partial") and arm_dy > 0.3:
            return "DIGGING", min(0.95, 0.6 + upper * 0.05)
        if source in ("arm_only", "partial") and arm_dy < -0.3:
            return "DUMPING", min(0.9, 0.5 + abs(arm_dy) * 0.2)
        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return "TRAVELING", 0.8
        if source == "arm_only":
            return "DIGGING", 0.5
        return "WAITING", 0.4
    
    def _classify_dump_truck(self, eq_id, m):
        if not m["is_active"]: return "WAITING", 0.9
        source = m["motion_source"]
        upper = m.get("upper_magnitude", 0)
        lower = m.get("lower_magnitude", 0)
        
        if source == "arm_only" and upper > Config.ARM_MOTION_THRESHOLD:
            return "DUMPING", 0.8
        if source in ("full_body", "tracks_only"):
            return "TRAVELING", 0.85
        return "WAITING", 0.5
    
    def _classify_generic(self, m):
        if not m["is_active"]: return "WAITING", 0.9
        if m.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 2:
            return "TRAVELING", 0.6
        return "WAITING", 0.5
    
    def _smooth(self, eq_id, current):
        history = list(self.activity_history.get(eq_id, []))
        if len(history) < 3: return current
        recent = history[-5:]
        counts = Counter(recent)
        if counts[current] >= len(recent) * 0.4: return current
        return counts.most_common(1)[0][0]

# %%
# ── State Manager ──

class EquipmentState:
    def __init__(self, eq_id, eq_class):
        self.equipment_id = eq_id
        self.equipment_class = eq_class
        self.current_state = "INACTIVE"
        self.current_activity = "WAITING"
        self.motion_source = "none"
        self.total_tracked = 0.0
        self.total_active = 0.0
        self.total_idle = 0.0
        self.utilization = 0.0
        self.first_seen = None
        self.last_update = None
        self.pending_state = None
        self.pending_count = 0
    
    def update(self, is_active, activity, motion_source, ts_sec):
        if self.first_seen is None:
            self.first_seen = ts_sec
            self.last_update = ts_sec
        
        dt = max(0, ts_sec - self.last_update)
        self.last_update = ts_sec
        self.total_tracked = ts_sec - self.first_seen
        
        if self.current_state == "ACTIVE": self.total_active += dt
        else: self.total_idle += dt
        
        new_state = "ACTIVE" if is_active else "INACTIVE"
        if new_state != self.current_state:
            if new_state == self.pending_state:
                self.pending_count += 1
            else:
                self.pending_state = new_state
                self.pending_count = 1
            if self.pending_count >= Config.STATE_DEBOUNCE_FRAMES:
                self.current_state = new_state
                self.pending_state = None
                self.pending_count = 0
        else:
            self.pending_state = None
            self.pending_count = 0
        
        if self.current_state == "ACTIVE":
            self.current_activity = activity
            self.motion_source = motion_source
        else:
            self.current_activity = "WAITING"
            self.motion_source = "none"
        
        if self.total_tracked > 0:
            self.utilization = round((self.total_active / self.total_tracked) * 100, 1)
    
    def snapshot(self):
        return {
            "equipment_id": self.equipment_id,
            "equipment_class": self.equipment_class,
            "current_state": self.current_state,
            "current_activity": self.current_activity,
            "motion_source": self.motion_source,
            "total_tracked_seconds": round(self.total_tracked, 1),
            "total_active_seconds": round(self.total_active, 1),
            "total_idle_seconds": round(self.total_idle, 1),
            "utilization_percent": self.utilization,
        }

class StateManager:
    def __init__(self):
        self.states = {}
    
    def update(self, eq_id, eq_class, is_active, activity, motion_source, ts_sec):
        if eq_id not in self.states:
            self.states[eq_id] = EquipmentState(eq_id, eq_class)
        self.states[eq_id].update(is_active, activity, motion_source, ts_sec)
        return self.states[eq_id].snapshot()

# %% [markdown]
# ## 5. Visualization

# %%
COLORS = {
    "ACTIVE": (0, 255, 0), "INACTIVE": (0, 0, 255),
    "DIGGING": (0, 200, 255), "SWINGING_LOADING": (255, 200, 0),
    "DUMPING": (200, 0, 255), "TRAVELING": (255, 255, 0),
    "WAITING": (128, 128, 128),
}

def draw_annotations(frame, detections, states):
    annotated = frame.copy()
    for det in detections:
        bbox = det["bbox"]
        eq_id = det.get("equipment_id", "UNK")
        state = states.get(eq_id, {})
        current_state = state.get("current_state", "INACTIVE")
        activity = state.get("current_activity", "WAITING")
        util = state.get("utilization_percent", 0)
        ms = state.get("motion_source", "none")
        
        color = COLORS.get(current_state, (255, 255, 255))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        thick = 3 if current_state == "ACTIVE" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        
        labels = [
            f"{eq_id} | {det.get('equipment_class', '?')}",
            f"{current_state} | {activity}",
            f"Util: {util:.1f}% | Motion: {ms}",
        ]
        
        y_off = y1 - 10
        for i, line in enumerate(reversed(labels)):
            ts = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            yp = y_off - i * 22
            cv2.rectangle(annotated, (x1, yp - 15), (x1+ts[0]+8, yp+5), color, -1)
            cv2.putText(annotated, line, (x1+4, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    
    return annotated

# %% [markdown]
# ## 6. Process Video

# %%
def process_video(video_path, output_dir=Config.OUTPUT_DIR):
    """Process a single video file and output annotated video + CSV results."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {w}x{h}, {fps:.1f} FPS, {total_frames} frames, {total_frames/fps:.1f}s")
    
    # Initialize components
    detector = EquipmentDetector()
    motion_analyzer = MotionAnalyzer()
    classifier = ActivityClassifier()
    state_manager = StateManager()
    
    # Output setup
    video_name = Path(video_path).stem
    out_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    out_csv_path = os.path.join(output_dir, f"{video_name}_events.csv")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_skip = max(1, int(fps / Config.TARGET_FPS))
    out_fps = fps / frame_skip
    out = cv2.VideoWriter(out_video_path, fourcc, out_fps, (w, h))
    
    # CSV writer
    csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "timestamp", "equipment_id", "equipment_class",
        "current_state", "current_activity", "motion_source",
        "total_tracked_sec", "total_active_sec", "total_idle_sec",
        "utilization_pct", "confidence",
    ])
    
    frame_count = 0
    processed_count = 0
    all_events = []
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        processed_count += 1
        ts_sec = frame_count / fps if fps > 0 else 0
        ts_str = f"{int(ts_sec//3600):02d}:{int((ts_sec%3600)//60):02d}:{ts_sec%60:06.3f}"
        
        # Detect
        detections = detector.detect_with_tracking(frame)
        
        # Motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = motion_analyzer.compute_optical_flow(gray)
        
        all_states = {}
        if flow is not None and detections:
            detections = motion_analyzer.analyze_equipment_motion(flow, detections, frame.shape)
            
            for det in detections:
                eq_id = det.get("equipment_id", "UNK")
                eq_class = det.get("equipment_class", "unknown")
                motion = det.get("motion", {})
                
                act_result = classifier.classify(eq_id, eq_class, motion)
                snap = state_manager.update(
                    eq_id, eq_class,
                    motion.get("is_active", False),
                    act_result["activity"],
                    motion.get("motion_source", "none"),
                    ts_sec,
                )
                all_states[eq_id] = snap
                
                # Write CSV row
                csv_writer.writerow([
                    frame_count, ts_str, eq_id, eq_class,
                    snap["current_state"], snap["current_activity"],
                    snap["motion_source"],
                    snap["total_tracked_seconds"], snap["total_active_seconds"],
                    snap["total_idle_seconds"], snap["utilization_percent"],
                    det.get("confidence", 0),
                ])
                
                all_events.append({
                    "frame_id": frame_count,
                    "equipment_id": eq_id,
                    "equipment_class": eq_class,
                    "timestamp": ts_str,
                    "utilization": {
                        "current_state": snap["current_state"],
                        "current_activity": snap["current_activity"],
                        "motion_source": snap["motion_source"],
                    },
                    "time_analytics": {
                        "total_tracked_seconds": snap["total_tracked_seconds"],
                        "total_active_seconds": snap["total_active_seconds"],
                        "total_idle_seconds": snap["total_idle_seconds"],
                        "utilization_percent": snap["utilization_percent"],
                    },
                })
        
        # Draw annotations and write frame
        annotated = draw_annotations(frame, detections, all_states)
        out.write(annotated)
        
        if processed_count % 50 == 0:
            elapsed = time.time() - start_time
            proc_fps = processed_count / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_count}/{total_frames} | "
                  f"{processed_count} processed | "
                  f"{proc_fps:.1f} FPS | "
                  f"Equipment: {len(state_manager.states)}")
    
    cap.release()
    out.release()
    csv_file.close()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {processed_count} frames in {elapsed:.1f}s ({processed_count/elapsed:.1f} FPS)")
    print(f"\nOutput files:")
    print(f"  Video: {out_video_path}")
    print(f"  CSV:   {out_csv_path}")
    
    print(f"\n{'='*60}")
    print(f"UTILIZATION SUMMARY")
    print(f"{'='*60}")
    for eq_id, state in state_manager.states.items():
        s = state.snapshot()
        print(f"  {eq_id} ({s['equipment_class']}):")
        print(f"    State: {s['current_state']} | Activity: {s['current_activity']}")
        print(f"    Active: {s['total_active_seconds']:.1f}s | Idle: {s['total_idle_seconds']:.1f}s")
        print(f"    Utilization: {s['utilization_percent']:.1f}%")
    print(f"{'='*60}")
    
    # Save JSON events
    events_path = os.path.join(output_dir, f"{video_name}_events.json")
    with open(events_path, "w") as f:
        json.dump(all_events, f, indent=2)
    print(f"  JSON:  {events_path}")
    
    return all_events

# %% [markdown]
# ## 7. Run Processing
# 
# **Update the video path below to point to your uploaded video.**

# %%
# ── UPDATE THIS PATH ──
# If you uploaded videos as a Kaggle dataset, the path will be like:
# /kaggle/input/your-dataset-name/video.mp4

VIDEO_PATH = "/kaggle/input/construction-videos/excavator_digging.mp4"  # <-- UPDATE THIS

# Check if file exists
if os.path.exists(VIDEO_PATH):
    events = process_video(VIDEO_PATH)
else:
    print(f"Video not found at: {VIDEO_PATH}")
    print("\nAvailable files in /kaggle/input/:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    print("\nPlease update VIDEO_PATH above.")

# %% [markdown]
# ## 8. Process All Videos in a Directory

# %%
def process_all_videos(input_dir="/kaggle/input"):
    """Find and process all video files."""
    import glob
    
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v}")
    
    all_results = {}
    for v in videos:
        events = process_video(v)
        all_results[v] = events
    
    return all_results

# Uncomment to process all videos:
# all_results = process_all_videos()

# %% [markdown]
# ## 9. Display Results

# %%
from IPython.display import display, HTML
import glob

output_files = glob.glob(os.path.join(Config.OUTPUT_DIR, "*"))
if output_files:
    print("Output files generated:")
    for f in output_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")
else:
    print("No output files yet. Run the processing cells above first.")
