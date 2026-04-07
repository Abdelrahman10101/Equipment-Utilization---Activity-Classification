"""
Motion Analyzer — Segmentation-Enhanced

Uses YOLO11m-seg instance segmentation masks combined with dense optical flow
to accurately detect articulated motion in construction equipment.

Key Improvement Over Bounding-Box Approach:
    Optical flow is computed ONLY within the equipment's pixel mask, eliminating
    background motion (wind, other machines, camera shake) that pollutes the
    bounding-box region. The mask is then split into vertical sub-regions to
    separate arm/boom motion from track/wheel motion.

How It Handles Articulated Motion:
    1. Segmentation mask isolates the equipment pixels precisely
    2. Mask is divided into upper (arm/boom), middle (cab), lower (tracks)
    3. Optical flow magnitude is computed per masked sub-region
    4. If upper region has flow but lower doesn't → arm-only motion → ACTIVE

Approach:
    1. Compute dense optical flow (Farnebäck) between consecutive frames
    2. For each detected equipment, apply the segmentation mask
    3. Compute flow ONLY within masked pixels per sub-region
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
    Analyzes motion within equipment segmentation masks using optical flow.
    Handles articulated motion by dividing the mask into sub-regions.
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

        # Temporal flow direction history for activity pattern detection
        self.flow_direction_history: Dict[str, deque] = {}
        self.direction_history_size = 30  # ~2-3 seconds at 10-15 FPS

        logger.info(
            f"MotionAnalyzer (seg-enhanced) initialized — thresholds: "
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
        Analyze motion for each detected equipment using segmentation masks
        and region-based optical flow.

        If a segmentation mask is available (from YOLO-seg), it's used to
        restrict flow analysis to equipment pixels only. Otherwise, falls
        back to bounding-box based analysis.

        Args:
            flow: Dense optical flow (H, W, 2)
            detections: List of equipment detections with bbox and optional mask
            frame_shape: (height, width) of the frame

        Returns:
            Enriched detections with motion analysis results
        """
        h, w = frame_shape[:2]

        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            equipment_id = det.get("equipment_id", "UNK")
            equipment_class = det.get("equipment_class", "unknown")
            mask = det.get("mask", None)  # Segmentation mask (if available)

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

            # If segmentation mask is available, use it
            mask_roi = None
            if mask is not None:
                mask_roi = mask[y1:y2, x1:x2]

            # Compute region-based motion analysis (mask-aware)
            motion_result = self._analyze_regions(
                flow_roi, equipment_class, x2 - x1, y2 - y1, mask_roi
            )

            # Update motion history for temporal smoothing
            if equipment_id not in self.motion_history:
                self.motion_history[equipment_id] = deque(maxlen=self.history_size)
            self.motion_history[equipment_id].append(motion_result)

            # Update flow direction history for temporal pattern detection
            if equipment_id not in self.flow_direction_history:
                self.flow_direction_history[equipment_id] = deque(
                    maxlen=self.direction_history_size
                )
            self.flow_direction_history[equipment_id].append({
                "upper_dy": motion_result["arm_direction"]["dy"],
                "upper_dx": motion_result["arm_direction"]["dx"],
                "upper_mag": motion_result["upper_magnitude"],
                "lateral_asym": motion_result["lateral_asymmetry"],
                "lower_mag": motion_result["lower_magnitude"],
                "is_active": motion_result["is_active"],
            })

            # Detect temporal flow patterns (for activity classification)
            temporal_patterns = self._detect_temporal_patterns(equipment_id)
            motion_result["temporal_patterns"] = temporal_patterns

            # Apply temporal smoothing
            smoothed = self._smooth_motion(equipment_id)
            smoothed["temporal_patterns"] = temporal_patterns
            det["motion"] = smoothed

        return detections

    def _analyze_regions(
        self,
        flow_roi: np.ndarray,
        equipment_class: str,
        roi_w: int,
        roi_h: int,
        mask_roi: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Divide equipment ROI into sub-regions and analyze motion in each.
        When a segmentation mask is provided, only count pixels within the mask.

        Sub-regions for excavator-type equipment:
        ┌─────────────────────┐
        │   Upper Region      │  ← Arm/Boom area
        │   (top 45%)         │     Flow computed only within mask pixels
        ├─────────────────────┤
        │   Middle Region     │  ← Cab/Swing area
        │   (middle 20%)      │
        ├─────────────────────┤
        │   Lower Region      │  ← Track/Wheel area
        │   (bottom 35%)      │
        └─────────────────────┘

        Segmentation mask improvement:
            Without mask: flow includes background (sky, ground, other objects)
            With mask:    flow ONLY from equipment pixels → cleaner signal
        """
        if flow_roi.size == 0:
            return self._empty_motion()

        # Compute flow magnitude and angle
        mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])

        # Apply mask if available — zero out non-equipment pixels
        if mask_roi is not None and mask_roi.shape == mag.shape:
            # Ensure mask is binary
            binary_mask = (mask_roi > 0).astype(np.float32)
            mag_masked = mag * binary_mask
            # Count valid pixels per region
            has_mask = True
        else:
            mag_masked = mag
            binary_mask = None
            has_mask = False

        # Overall motion (masked)
        if has_mask:
            mask_pixel_count = np.sum(binary_mask)
            if mask_pixel_count > 0:
                overall_magnitude = float(np.sum(mag_masked) / mask_pixel_count)
                overall_max = float(np.max(mag_masked))
            else:
                overall_magnitude = 0.0
                overall_max = 0.0
        else:
            overall_magnitude = float(np.mean(mag))
            overall_max = float(np.max(mag))

        # --- Region-based analysis ---
        upper_end = int(roi_h * 0.45)
        middle_end = int(roi_h * 0.65)

        # Compute per-region magnitudes (mask-aware)
        upper_mag = self._region_magnitude(mag_masked, binary_mask, 0, upper_end)
        middle_mag = self._region_magnitude(mag_masked, binary_mask, upper_end, middle_end)
        lower_mag = self._region_magnitude(mag_masked, binary_mask, middle_end, roi_h)

        # Horizontal regions (for swing detection)
        mid_x = roi_w // 2
        left_mag = self._region_magnitude_horizontal(mag_masked, binary_mask, 0, mid_x)
        right_mag = self._region_magnitude_horizontal(mag_masked, binary_mask, mid_x, roi_w)

        # Direction analysis for arm movement
        upper_flow = flow_roi[:upper_end, :]
        if has_mask and binary_mask is not None:
            upper_mask = binary_mask[:upper_end, :]
        else:
            upper_mask = None

        if upper_flow.size > 0:
            if upper_mask is not None:
                upper_pixels = np.sum(upper_mask)
                if upper_pixels > 0:
                    avg_dx = float(np.sum(upper_flow[..., 0] * upper_mask) / upper_pixels)
                    avg_dy = float(np.sum(upper_flow[..., 1] * upper_mask) / upper_pixels)
                else:
                    avg_dx, avg_dy = 0.0, 0.0
            else:
                avg_dx = float(np.mean(upper_flow[..., 0]))
                avg_dy = float(np.mean(upper_flow[..., 1]))
        else:
            avg_dx, avg_dy = 0.0, 0.0

        # Lateral motion asymmetry (indicates swing/rotation)
        lateral_asymmetry = abs(float(left_mag) - float(right_mag))

        # Compute mask coverage ratio per region (how much of the region is equipment)
        if has_mask and binary_mask is not None:
            upper_coverage = self._region_coverage(binary_mask, 0, upper_end)
            lower_coverage = self._region_coverage(binary_mask, middle_end, roi_h)
        else:
            upper_coverage = 1.0
            lower_coverage = 1.0

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
            "has_segmentation_mask": has_mask,
            "upper_coverage": round(upper_coverage, 3),
            "lower_coverage": round(lower_coverage, 3),
        }

    def _region_magnitude(
        self,
        mag: np.ndarray,
        mask: Optional[np.ndarray],
        y_start: int,
        y_end: int,
    ) -> float:
        """Compute mean flow magnitude in a vertical region, respecting mask."""
        if y_end <= y_start:
            return 0.0
        region = mag[y_start:y_end, :]
        if region.size == 0:
            return 0.0
        if mask is not None:
            mask_region = mask[y_start:y_end, :]
            pixel_count = np.sum(mask_region)
            if pixel_count > 0:
                return float(np.sum(region) / pixel_count)
            return 0.0
        return float(np.mean(region))

    def _region_magnitude_horizontal(
        self,
        mag: np.ndarray,
        mask: Optional[np.ndarray],
        x_start: int,
        x_end: int,
    ) -> float:
        """Compute mean flow magnitude in a horizontal region, respecting mask."""
        if x_end <= x_start:
            return 0.0
        region = mag[:, x_start:x_end]
        if region.size == 0:
            return 0.0
        if mask is not None:
            mask_region = mask[:, x_start:x_end]
            pixel_count = np.sum(mask_region)
            if pixel_count > 0:
                return float(np.sum(region) / pixel_count)
            return 0.0
        return float(np.mean(region))

    def _region_coverage(
        self,
        mask: np.ndarray,
        y_start: int,
        y_end: int,
    ) -> float:
        """Compute what fraction of a region is covered by the mask."""
        if y_end <= y_start:
            return 0.0
        region = mask[y_start:y_end, :]
        if region.size == 0:
            return 0.0
        return float(np.sum(region) / region.size)

    def _detect_temporal_patterns(self, equipment_id: str) -> Dict[str, Any]:
        """
        Analyze flow direction history to detect temporal motion patterns
        that correspond to specific activities.

        Patterns detected:
            - down_then_up:   Digging cycle (arm goes down to scoop, comes back up)
            - sustained_lateral: Swinging (consistent lateral motion indicating rotation)
            - up_then_release: Dumping (arm goes up, bucket tips, material drops)
            - no_motion:      Waiting (sustained low motion)
            - oscillating:    Loading/vibration (being loaded by another machine)

        Returns:
            Dict with pattern detection results and confidence scores
        """
        history = self.flow_direction_history.get(equipment_id, [])

        if len(history) < 5:
            return {
                "dominant_pattern": "insufficient_data",
                "confidence": 0.0,
                "down_then_up": False,
                "sustained_lateral": False,
                "up_then_release": False,
                "sustained_stillness": False,
                "oscillating": False,
            }

        recent = list(history)
        dy_values = [h["upper_dy"] for h in recent]
        dx_values = [h["upper_dx"] for h in recent]
        upper_mags = [h["upper_mag"] for h in recent]
        lateral_asym = [h["lateral_asym"] for h in recent]
        lower_mags = [h["lower_mag"] for h in recent]
        active_flags = [h["is_active"] for h in recent]

        patterns = {}

        # --- PATTERN 1: Down-then-Up (DIGGING) ---
        # Look for a sequence where dy goes positive (down) then negative (up)
        down_then_up = self._detect_down_up_cycle(dy_values, upper_mags)
        patterns["down_then_up"] = down_then_up

        # --- PATTERN 2: Sustained Lateral Motion (SWINGING/LOADING) ---
        # Consistent high lateral asymmetry or consistent dx direction
        recent_lateral = lateral_asym[-min(10, len(lateral_asym)):]
        recent_dx = dx_values[-min(10, len(dx_values)):]
        sustained_lateral = (
            np.mean(recent_lateral) > self.motion_threshold * 1.2
            or (
                abs(np.mean(recent_dx)) > 0.5
                and np.std(np.sign(recent_dx)) < 0.6  # consistent direction
            )
        )
        patterns["sustained_lateral"] = bool(sustained_lateral)

        # --- PATTERN 3: Up-then-Release (DUMPING) ---
        # dy goes negative (arm up) then sudden magnitude drop (release)
        up_then_release = self._detect_up_release_cycle(dy_values, upper_mags)
        patterns["up_then_release"] = up_then_release

        # --- PATTERN 4: Sustained Stillness (WAITING) ---
        # Low motion in all regions for sustained period
        recent_active = active_flags[-min(10, len(active_flags)):]
        sustained_stillness = sum(recent_active) < len(recent_active) * 0.2
        patterns["sustained_stillness"] = bool(sustained_stillness)

        # --- PATTERN 5: Oscillating (LOADING vibration) ---
        # Rapid alternation of motion magnitude (being loaded)
        recent_mags = upper_mags[-min(10, len(upper_mags)):]
        if len(recent_mags) >= 5:
            mag_diffs = np.diff(recent_mags)
            sign_changes = np.sum(np.diff(np.sign(mag_diffs)) != 0)
            oscillating = sign_changes > len(mag_diffs) * 0.5 and np.mean(recent_mags) > 0.5
        else:
            oscillating = False
        patterns["oscillating"] = bool(oscillating)

        # Determine dominant pattern with confidence
        dominant, confidence = self._determine_dominant_pattern(patterns, recent)

        return {
            "dominant_pattern": dominant,
            "confidence": round(confidence, 2),
            **patterns,
        }

    def _detect_down_up_cycle(
        self, dy_values: list, upper_mags: list
    ) -> bool:
        """
        Detect a digging cycle: arm moves down (dy > 0) then up (dy < 0).
        Requires significant upper region motion during both phases.
        """
        if len(dy_values) < 8:
            return False

        recent_dy = dy_values[-min(15, len(dy_values)):]
        recent_mag = upper_mags[-min(15, len(upper_mags)):]

        # Find sign transitions in dy
        has_positive_phase = any(d > 0.3 for d in recent_dy)
        has_negative_phase = any(d < -0.3 for d in recent_dy)
        has_motion = np.mean(recent_mag) > self.arm_threshold * 0.5

        if has_positive_phase and has_negative_phase and has_motion:
            # Check for actual transition (not just noise)
            signs = [1 if d > 0.2 else (-1 if d < -0.2 else 0) for d in recent_dy]
            transitions = sum(
                1 for i in range(1, len(signs))
                if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0
            )
            return transitions >= 1

        return False

    def _detect_up_release_cycle(
        self, dy_values: list, upper_mags: list
    ) -> bool:
        """
        Detect a dumping cycle: arm moves up (dy < 0) then sudden motion drop.
        """
        if len(dy_values) < 8:
            return False

        recent_dy = dy_values[-min(15, len(dy_values)):]
        recent_mag = upper_mags[-min(15, len(upper_mags)):]

        # Look for consistent upward motion followed by magnitude drop
        mid = len(recent_dy) // 2
        first_half_dy = recent_dy[:mid]
        second_half_mag = recent_mag[mid:]

        upward_phase = np.mean(first_half_dy) < -0.3 if first_half_dy else False
        release_phase = (
            len(second_half_mag) > 2
            and np.mean(second_half_mag[:2]) > self.arm_threshold
            and np.mean(second_half_mag[-2:]) < self.arm_threshold * 0.5
        ) if second_half_mag else False

        return bool(upward_phase and release_phase)

    def _determine_dominant_pattern(
        self, patterns: Dict[str, bool], recent_history: list
    ) -> Tuple[str, float]:
        """Determine the most likely activity pattern from detected patterns."""
        if patterns["sustained_stillness"]:
            return "waiting", 0.9

        active_patterns = [k for k, v in patterns.items() if v and k != "sustained_stillness"]

        if not active_patterns:
            # Check if there's any motion at all
            avg_mag = np.mean([h["upper_mag"] for h in recent_history[-5:]])
            if avg_mag > self.arm_threshold:
                return "active_unclassified", 0.4
            return "waiting", 0.6

        # Priority order: down_then_up > sustained_lateral > up_then_release > oscillating
        if patterns["down_then_up"]:
            return "digging_cycle", 0.75
        if patterns["sustained_lateral"]:
            return "swinging", 0.70
        if patterns["up_then_release"]:
            return "dumping_cycle", 0.70
        if patterns["oscillating"]:
            return "loading_vibration", 0.60

        return "active_unclassified", 0.4

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
            "has_segmentation_mask": False,
            "upper_coverage": 0.0,
            "lower_coverage": 0.0,
            "temporal_patterns": {
                "dominant_pattern": "insufficient_data",
                "confidence": 0.0,
                "down_then_up": False,
                "sustained_lateral": False,
                "up_then_release": False,
                "sustained_stillness": False,
                "oscillating": False,
            },
        }

    def get_temporal_patterns(self, equipment_id: str) -> Dict[str, Any]:
        """Get the current temporal flow patterns for an equipment."""
        return self._detect_temporal_patterns(equipment_id)

    def reset(self):
        """Reset state for a new video."""
        self.prev_gray = None
        self.motion_history.clear()
        self.flow_direction_history.clear()
