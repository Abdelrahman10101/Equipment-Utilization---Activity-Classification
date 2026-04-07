"""
Motion Analyzer

Implements region-based optical flow analysis to detect articulated motion
in construction equipment. This is key for correctly classifying an excavator
as ACTIVE when only its arm is moving while the tracks are stationary.

Approach:
1. Compute dense optical flow (Farnebäck) between consecutive frames
2. For each detected equipment bbox, divide into sub-regions
3. Calculate motion magnitude per sub-region
4. Determine ACTIVE/INACTIVE state and motion source
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

import cv2
import numpy as np

from config import Config

logger = logging.getLogger("MotionAnalyzer")


class MotionAnalyzer:
    """
    Analyzes motion within equipment bounding boxes using optical flow.
    Handles articulated motion by dividing equipment into sub-regions.
    """

    def __init__(self):
        self.prev_gray = None
        self.motion_threshold = Config.MOTION_THRESHOLD
        self.arm_threshold = Config.ARM_MOTION_THRESHOLD
        self.track_threshold = Config.TRACK_MOTION_THRESHOLD

        # Optical flow parameters (Farnebäck)
        self.flow_params = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        }

        # Motion history per equipment (for temporal smoothing)
        self.motion_history: Dict[str, deque] = {}
        self.history_size = Config.SLIDING_WINDOW_SIZE

        logger.info(
            f"MotionAnalyzer initialized — thresholds: "
            f"general={self.motion_threshold}, "
            f"arm={self.arm_threshold}, "
            f"track={self.track_threshold}"
        )

    def compute_optical_flow(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense optical flow between current and previous frame.

        Returns:
            Optical flow array (H, W, 2) with dx, dy components, or None if first frame
        """
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            frame_gray,
            None,
            **self.flow_params,
        )

        self.prev_gray = frame_gray.copy()
        return flow

    def analyze_equipment_motion(
        self,
        flow: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Analyze motion for each detected equipment using region-based approach.

        Args:
            flow: Dense optical flow (H, W, 2)
            detections: List of equipment detections with bbox
            frame_shape: (height, width) of the frame

        Returns:
            Enriched detections with motion analysis results
        """
        h, w = frame_shape[:2]

        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            equipment_id = det.get("equipment_id", "UNK")
            equipment_class = det.get("equipment_class", "unknown")

            # Clamp bbox to frame boundaries
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                det["motion"] = self._empty_motion()
                continue

            # Extract flow within bounding box
            flow_roi = flow[y1:y2, x1:x2]

            # Compute region-based motion analysis
            motion_result = self._analyze_regions(
                flow_roi, equipment_class, x2 - x1, y2 - y1
            )

            # Update motion history for temporal smoothing
            if equipment_id not in self.motion_history:
                self.motion_history[equipment_id] = deque(maxlen=self.history_size)
            self.motion_history[equipment_id].append(motion_result)

            # Apply temporal smoothing
            smoothed = self._smooth_motion(equipment_id)
            det["motion"] = smoothed

        return detections

    def _analyze_regions(
        self,
        flow_roi: np.ndarray,
        equipment_class: str,
        roi_w: int,
        roi_h: int,
    ) -> Dict[str, Any]:
        """
        Divide equipment ROI into sub-regions and analyze motion in each.

        Sub-regions for excavator-type equipment:
        ┌─────────────────────┐
        │   Upper Region      │  ← Arm/Boom area
        │   (top 45%)         │
        ├─────────────────────┤
        │   Middle Region     │  ← Cab/Swing area
        │   (middle 20%)      │
        ├─────────────────────┤
        │   Lower Region      │  ← Track/Wheel area
        │   (bottom 35%)      │
        └─────────────────────┘

        Also split left/right for detecting swing rotation.
        """
        if flow_roi.size == 0:
            return self._empty_motion()

        # Compute flow magnitude and angle
        mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])

        # Overall motion
        overall_magnitude = float(np.mean(mag))
        overall_max = float(np.max(mag))

        # --- Region-based analysis ---
        # Vertical regions
        upper_end = int(roi_h * 0.45)
        middle_end = int(roi_h * 0.65)

        upper_mag = np.mean(mag[:upper_end, :]) if upper_end > 0 else 0
        middle_mag = np.mean(mag[upper_end:middle_end, :]) if middle_end > upper_end else 0
        lower_mag = np.mean(mag[middle_end:, :]) if middle_end < roi_h else 0

        # Horizontal regions (for swing detection)
        mid_x = roi_w // 2
        left_mag = np.mean(mag[:, :mid_x]) if mid_x > 0 else 0
        right_mag = np.mean(mag[:, mid_x:]) if mid_x < roi_w else 0

        # Direction analysis for activity classification
        # Average flow direction in upper region (for arm movement direction)
        upper_flow = flow_roi[:upper_end, :]
        if upper_flow.size > 0:
            avg_dx = float(np.mean(upper_flow[..., 0]))
            avg_dy = float(np.mean(upper_flow[..., 1]))
        else:
            avg_dx, avg_dy = 0.0, 0.0

        # Lateral motion asymmetry (indicates swing/rotation)
        lateral_asymmetry = abs(float(left_mag) - float(right_mag))

        # Determine motion state
        is_upper_active = float(upper_mag) > self.arm_threshold
        is_middle_active = float(middle_mag) > self.motion_threshold
        is_lower_active = float(lower_mag) > self.track_threshold
        has_lateral_motion = lateral_asymmetry > self.motion_threshold

        # Determine overall state and motion source
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
        """
        Apply temporal smoothing over the motion history window.
        Uses majority voting for state and averaging for magnitudes.
        """
        history = self.motion_history.get(equipment_id, [])
        if not history:
            return self._empty_motion()

        latest = history[-1].copy()

        if len(history) < 3:
            return latest

        # Majority voting for is_active (debouncing)
        active_count = sum(1 for h in history if h["is_active"])
        latest["is_active"] = active_count > len(history) * 0.4  # 40% threshold

        # Most common motion source
        sources = [h["motion_source"] for h in history if h["motion_source"] != "none"]
        if sources:
            from collections import Counter
            latest["motion_source"] = Counter(sources).most_common(1)[0][0]
        elif not latest["is_active"]:
            latest["motion_source"] = "none"

        # Average magnitudes
        latest["overall_magnitude"] = round(
            np.mean([h["overall_magnitude"] for h in history]), 3
        )

        return latest

    def _empty_motion(self) -> Dict[str, Any]:
        """Return empty motion result for edge cases."""
        return {
            "is_active": False,
            "motion_source": "none",
            "overall_magnitude": 0.0,
            "overall_max": 0.0,
            "upper_magnitude": 0.0,
            "middle_magnitude": 0.0,
            "lower_magnitude": 0.0,
            "left_magnitude": 0.0,
            "right_magnitude": 0.0,
            "lateral_asymmetry": 0.0,
            "arm_direction": {"dx": 0.0, "dy": 0.0},
            "region_active": {
                "upper": False,
                "middle": False,
                "lower": False,
                "lateral": False,
            },
        }

    def reset(self):
        """Reset state for a new video."""
        self.prev_gray = None
        self.motion_history.clear()
